import os
import torch
from typing import List, Optional
from transformers import AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from models.vamba_mamba2.configuration_qwen2_vl import Qwen2VLConfig
from models.vamba_mamba2.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from models.vamba_mamba2.processing_qwen2_vl import Qwen2VLProcessor
from models.vamba_mamba2.image_processing_qwen2_vl import Qwen2VLImageProcessor
import pdb
from train.conversation import conv_templates
from tools.chat_utils import load_media_data_image, load_media_data_video, load_identity

from calflops import calculate_flops


def load_image_or_video(image_or_video, model, processor):
    _type = image_or_video["type"]
    content = image_or_video["content"]
    metadata = image_or_video.get("metadata", {})

    if _type == "image":
        load_func = load_media_data_image
    elif _type == "video":
        load_func = load_media_data_video
    elif _type == "pil_image" or _type == "pil_video":
        load_func = load_identity
    else:
        raise ValueError(f"Unknown type: {_type}")
    return load_func(content, model, processor, **metadata)

class Vamba():
    def __init__(self, model_path="TIGER-Lab/Vamba-Qwen2-VL-7B", device="cuda") -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        attn_implementation = "flash_attention_2"
        print(f"Using {attn_implementation} for attention implementation")

        config = Qwen2VLConfig.from_pretrained(model_path)
        # config.vision_mamba_mixer_config.use_mem_eff_path = False
        config._attn_implementation = attn_implementation
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).to(device).eval()
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        chat_template = tokenizer.chat_template
        self.processor = Qwen2VLProcessor(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template)
        self.patch_size = image_processor.patch_size
        self.conv = conv_templates["qwen2"]
        self.terminators = [
            self.processor.tokenizer.eos_token_id,
        ]
        
    def __call__(self, inputs: List[dict], generation_config: Optional[dict] = None) -> str:
        images = [x for x in inputs if x["type"] == "image" or x["type"] == "video" or x["type"] == "pil_image" or x["type"] == "pil_video"]
        assert len(images) == 1, "only support 1 input image/video"
        images = images[0]

        # should keep (tokenizer)
        text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
        if "<video> " in text_prompt:
            text_prompt = text_prompt.replace("<video> ", "<|vision_start|><|vision_end|>")
        elif "<image> " in text_prompt:
            text_prompt = text_prompt.replace("<image> ", "<|vision_start|><|vision_end|>")
        conv = self.conv.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        input_images = load_image_or_video(images, self.patch_size, self.processor)

        do_resize = False if images.get("metadata", True) else images["metadata"].get("do_resize", False)
        inputs = self.processor(text=prompt, images=input_images, do_resize=do_resize, return_tensors="pt")
        """
        Inputs:
        input_ids: torch.Size([1, 1324]
        attention_mask: torch.Size([1, 1324]
        pixel_values_videos: torch.Size([5184, 1176])
        video_grid_thw:  [4, 36, 36]
        """
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_config = generation_config if generation_config is not None else {}
        if "max_new_tokens" not in generation_config:
            generation_config["max_new_tokens"] = 512
        if "eos_token_id" not in generation_config:
            generation_config["eos_token_id"] = self.terminators

        # all_kwargs = {}
        # all_kwargs.update(inputs)
        # all_kwargs.update(generation_config)
        # flops, macs, params = calculate_flops(
        #     model=self.model,
        #     kwargs=all_kwargs,
        #     forward_mode="generate"
        # )
        # print(f"{flops}, {macs}, {params}")
        generate_ids = self.model.generate(**inputs, **generation_config)
        generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return generated_text

    
if __name__ == "__main__":
    model = Vamba(model_path="TIGER-Lab/Vamba-Qwen2-VL-7B")
    print("############################################################################################")
    test_input = [
        {
            "type": "video",
            "content": "assets/magic.mp4",
            "metadata": {
                "video_num_frames": 128,
                "video_sample_type": "middle",
                "img_longest_edge": 640,
                "img_shortest_edge": 256,
            }
        },
        {
            "type": "text",
            "content": "<video> Describe the magic trick."
        }
    ]
    print(model(test_input))
    print("############################################################################################")
    test_input = [
        {
            "type": "image",
            "content": "assets/old_man.png",
            "metadata": {}
        },
        {
            "type": "text",
            "content": "<image> Describe this image."
        }
    ]
    print(model(test_input))