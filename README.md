# Vamba

This repo contains code for [Vamba](https://arxiv.org/abs/TODO), a hybrid Mamba-Transformer model that leverages cross-attention layers and Mamba-2 blocks for efficient hour-long video understanding.

[**🌐 Homepage**](https://tiger-ai-lab.github.io/Vamba/) | [**📖 arXiv**](https://arxiv.org/abs/TODO) | [**💻 GitHub**](https://github.com/TIGER-AI-Lab/Vamba) | [**🤗 Model**](https://huggingface.co/TIGER-Lab/Vamba-Qwen2-VL-7B)

## Install
Please use the following commands to install the required packages:
```bash
conda env create -f environment.yaml
conda activate vamba
pip install flash-attn --no-build-isolation
```
## Model Inference
```bash
cd Vamba
export PYTHONPATH=.
python tools/vamba_chat.py
```


## Model Training
1. Modify the data configuration files under `train/data_configs/` to point to the correct paths of the datasets. You should refer to [CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds), [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose), [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) and [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) for preparing the training datasets.
2. Follow the commands below to train Vamba model:
```bash
# pretraining
bash scripts/pretrain_vamba.sh

# instruction-tuning
bash scripts/sft_vamba.sh
```

## Evaluation
Use the scripts under `eval/` to evaluate Vamba models. For example, to evaluate Video-MME, use the command:
```
cd Vamba
export PYTHONPATH=.
python eval/eval_videomme.py --model_type vamba --model_name_or_path TIGER-Lab/Vamba-Qwen2-VL-7B --num_frames 512 --data_dir <path_to_videomme_data>
```

## Vamba Model Architecture
<p align="center">
<img src="https://tiger-ai-lab.github.io/Vamba/static/images/vamba_main.png" width="900">
</p>

The main computation overhead in the transformer-based LMMs comes from the quadratic complexity of the self-attention in the video tokens. To overcome this issue, we design a hybrid Mamba Transformer architecture to process text and video tokens differently. The key idea of our method is to split the expensive self-attention operation over the entire video and text token sequence into two more efficient components. Since video tokens typically dominate the sequence while text tokens remain few, we maintain the self-attention mechanism exclusively for the text tokens and eliminate it for the video tokens. Instead, we add cross-attention layers that use text tokens as queries and video tokens as keys and values. In the meantime, we propose employing Mamba blocks to effectively process the video tokens.



## Citation
If you find our paper useful, please cite us with
```
coming soon
```
