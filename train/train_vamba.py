from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from hf_mtask_trainer import HfMultiTaskTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
import torch
import os
import wandb
import regex as re
from train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from conversation import conv_qwen2 as default_conv, conv_templates
from train.data import load_data_from_config, set_default_image_token, set_default_video_token, set_mmodal_token_sep, set_ignore_index
from pathlib import Path
from typing import Optional
from pathlib import Path

os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

@dataclass
class DataArguments:
    max_img_seq_len: Optional[int] = field(
        metadata={"help": "The maximum number of image sequence length after tokenization. Sequences longer "
                          "than this will be truncated.", "default": 1024, "required": False},
        default=20000,
    )
    max_txt_seq_len: Optional[int] = field(
        metadata={"help": "The maximum number of text sequence length after tokenization. Sequences longer "
                          "than this will be truncated.", "default": 1024, "required": False},
        default=1024,
    )
    data_config_file: Optional[str] = field(
        metadata={"help": "Pretrained config name or path if not the same as model_name", "default": None, "required": False},
        default=None,
    )
    dataset_balancing: Optional[bool] = field(
        metadata={"help": "Whether to balance the dataset", "default": True, "required": False},
        default=True,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models", "default": "llava-hf/llava-1.5-7b-hf", "required": False},
        default="llava-hf/llava-1.5-7b-hf",
    )
    trainable_modules: Optional[str] = field(
        metadata={"help": "The modules to train", "default": "all", "required": False},
        default="all",
    )
    lora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use LoRA", "default": False, "required": False},
        default=False,
    )
    qlora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use QLoRA", "default": False, "required": False},
        default=False,
    )
    dora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use Dora", "default": False, "required": False},
        default=True,
    )
    lora_r: Optional[int] = field(
        metadata={"help": "LoRA r", "default": 128, "required": False},
        default=128,
    )
    lora_alpha: Optional[float] = field(
        metadata={"help": "LoRA alpha", "default": 256, "required": False},
        default=256,
    )
    lora_dropout: Optional[float] = field(
        metadata={"help": "LoRA dropout", "default": 0.05, "required": False},
        default=0.05,
    )
    lora_bias: Optional[str] = field(
        metadata={"help": "LoRA bias", "default": 'none', "required": False},
        default='none',
    )
    attn_implementation: Optional[str] = field(
        metadata={"help": "The attention implementation to use", "default": "flash_attention_2", "required": False},
        default="flash_attention_2",
    )
    max_image_size: Optional[str] = field(
        metadata={"help": "The maximum image size", "default": "(1080,1920)", "required": False},
        default="(1080,1920)",
    )
    conv_template : Optional[str] = field(
        metadata={"help": "The conversation template to use", "default": None, "required": False},
        default=None,
    )
    init_cross_attn_weights_from_self_attn : Optional[bool] = field(
        metadata={"help": "Whether to initialize cross attention weights from self attention layers", "default": False, "required": False},
        default=False,
    )
    enable_vision_mamba_mixer : Optional[bool] = field(
        metadata={"help": "Whether to enable mamba mixer layers in LLM decoder", "default": False, "required": False},
        default=False,
    )
    use_alpha_balancing : Optional[bool] = field(
        metadata={"help": "Whether to use alpha balancing", "default": False, "required": False},
        default=False,
    )
    cross_attn_use_position_embeddings : Optional[bool] = field(
        metadata={"help": "Whether to use position embeddings in cross attention layers", "default": False, "required": False},
        default=False,
    )
    cross_attn_use_layer_norm : Optional[bool] = field(
        metadata={"help": "Whether to use layer norms for image hidden states", "default": False, "required": False},
        default=False,
    )
    distill_coeff : Optional[float] = field(
        metadata={"help": "Coefficient for distillation loss", "default": 0.0, "required": False},
        default=0.0,
    )
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model(model_args, training_args):
    print("Loading model...")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    
    if model_args.qlora_enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tower"],
        )
    else:
        bnb_config = None
        
    from transformers import AutoTokenizer
    from models.vamba_mamba2.configuration_qwen2_vl import Qwen2VLConfig
    from models.vamba_mamba2.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from models.vamba_mamba2.processing_qwen2_vl import Qwen2VLProcessor
    from models.vamba_mamba2.image_processing_qwen2_vl import Qwen2VLImageProcessor
    config = Qwen2VLConfig.from_pretrained(model_args.model_name_or_path,)
    config.vision_mamba_mixer_config.enable = model_args.enable_vision_mamba_mixer
    config.use_alpha_balancing = model_args.use_alpha_balancing
    config.cross_attn_use_position_embeddings = model_args.cross_attn_use_position_embeddings
    config.cross_attn_use_layer_norm = model_args.cross_attn_use_layer_norm
    config.distill_coeff = model_args.distill_coeff
    config._attn_implementation = model_args.attn_implementation
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        config=config,
        quantization_config=bnb_config if model_args.qlora_enabled else None,
    )
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    processor = Qwen2VLProcessor(image_processor, tokenizer, model_config=model.config)

    if model_args.init_cross_attn_weights_from_self_attn:
        print("Initializing cross attention weights from self attention layers...")
        model.init_cross_attn_from_self_attn()
        print("Successfully initialized cross attention weights from self attention layers")

    if model_args.trainable_modules != "all":
        trainable_modules = [x.strip() for x in model_args.trainable_modules.split(",")]
        print("Trainable modules:", trainable_modules)
        for name, param in model.named_parameters():
            if any([x in name for x in trainable_modules]):
                print("Tuning", name)
            else:
                param.requires_grad = False
        if ("cross_attn" in trainable_modules or "vision_mixer" in trainable_modules) and len(trainable_modules) == 1:
            model.enable_input_require_grads()
        elif "cross_attn" in trainable_modules and "vision_mixer" in trainable_modules and len(trainable_modules) == 2:
            model.enable_input_require_grads()
    else:
        print("Tuning all modules")
    print("Successfully loaded model from:", model_args.model_name_or_path)

    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if model_args.lora_enabled:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
            use_dora=model_args.dora_enabled,
        )
        print("Adding LoRA adapters...")
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        print("Successfully added LoRA adapters")
    
    set_default_image_token("<|vision_start|><|vision_end|>")
    set_default_video_token("<|vision_start|><|vision_end|>")
    set_mmodal_token_sep("")
    set_ignore_index(-100)
    return model, processor

def main(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    training_args.output_dir = Path(training_args.output_dir) / model_args.model_name_or_path.split("/")[-1] / training_args.run_name
    
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(training_args.output_dir)
    training_args.remove_unused_columns = False
    data_args.is_master_worker = training_args.local_rank in [-1, 0]
    from train.data import Collator
    data_args.extra_collator_func = Collator._right_pad_inputs_with_attention_mask
    
    if not training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = True
    if training_args.resume_from_checkpoint == True:
        # search for the latest checkpoint
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [x for x in all_checkpoints if (x / "trainer_state.json").exists() and not x.name.endswith("final")]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
    
    model, processor = load_model(model_args, training_args)

    data_args.model_patch_size = processor.image_processor.patch_size
    data_args.temporal_patch_size = processor.image_processor.temporal_patch_size
    
    if model_args.conv_template:
        data_args.conv_format = conv_templates[model_args.conv_template] 
    else:
        data_args.conv_format = default_conv
    print("Using conversation template:", data_args.conv_format)
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        raise ValueError("Data config file is required")

    trainer = HfMultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor,
    )
    if trainer.is_world_process_zero():
        print("Training arguments:")
        print(training_args)
        print("Data arguments:")
        print(data_args)
        print("Model arguments:")
        print(model_args)
    if training_args.do_train:
        print("Training model...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # save
        final_checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-final')
        if model_args.lora_enabled:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), model_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(final_checkpoint_dir)
                model.save_pretrained(final_checkpoint_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(final_checkpoint_dir, 'non_lora_trainables.bin'))
        else:
            trainer.save_model(output_dir=final_checkpoint_dir)
        processor.save_pretrained(final_checkpoint_dir)
    if training_args.do_predict:
        print("Predicting...")
        trainer.predict(test_dataset)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    main(training_args, data_args, model_args)