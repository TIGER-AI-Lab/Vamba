import fire
import json
from pathlib import Path
from eval.utils_hourvideo import HourVideoDataset
from tqdm import tqdm

def main(
    model_type="vamba",
    model_name_or_path="TIGER-Lab/Vamba-Qwen2-VL-7B",
    data_dir="/path/to/datasets/HourVideo/videos",
    json_path="/path/to/datasets/HourVideo/dev_v1.0_annotations.json",
    num_frames=128,
    img_shortest_edge=256,
    img_longest_edge=360,
    max_img_seq_len=16500,
    do_resize=True,
    results_dir="./output/eval/hourvideo",
    overwrite=False,
    # generation config
    max_new_tokens=512,
    do_sample=True,
    top_k=None,
    top_p=0.9,
    temperature=1,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "vamba":
        from tools.vamba_chat import Vamba
        model = Vamba(model_name_or_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    sample_config = {
        'num_frames': num_frames,
        'sample_type': 'uniform',
        'model_patch_size': model.patch_size,
        'img_shortest_edge': img_shortest_edge,
        'img_longest_edge': img_longest_edge,
        'max_img_seq_len': max_img_seq_len,
        'do_resize': do_resize,
    }

    dataset = HourVideoDataset(
        data_dir,
        json_path,
        sample_config=sample_config,
    )

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }

    core_data = {}

    model_save_path = "/".join(model_name_or_path.split("/")[-2:])
    results_file = Path(results_dir) / f"{num_frames}frames" / f"{model_save_path}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    if results_file.exists() and not overwrite:
        with open(results_file, "r") as rf:
            try:
                core_data = json.load(rf)
            except json.JSONDecodeError:
                core_data = {}
    
    for i in tqdm(range(len(core_data), len(dataset))):
        data = dataset[i]
        uid = data['uid']
        if uid in core_data:
            continue
        images = data["video"]
        questions = data["questions"]
        video_answers = {uid:{"benchmark_dataset":[]}}
        
        for question in questions:
            text = '\n'.join([
                'Select the best answer to the following multiple-choice question based on the video.',
                question['question'],
                question['mcq_test'],
                'Respond with only the letter (A, B, C, D, or E) of the correct option.',
            ])

            messages = [
                {
                    "type": "pil_video",
                    "content": images
                },
                {
                    "type": "text",
                    "content": f"<video> {text}",
                }
            ]
            
            response = model(messages, generation_config)
            response = response.lower()
            if model_type == "llava_mini":
                question["response"] = response
        
            if "the answer is" in response:
                response = response.split("the answer is")[-1].strip()
            elif "answer:" in response:
                response = response.split("answer:")[-1].strip()
            elif "the option is" in response:
                response = response.split("the option is ")[-1].strip()
            for char in response:
                if char.isalpha():
                    response = char
                    break
            try:
                question["predicted_answer_label"] = response[0].upper()
            except Exception as e:
                question["predicted_answer_label"] = 'Z'
            
            question["correct"] = question["predicted_answer_label"][0] == question["correct_answer_label"] or question["predicted_answer_label"][0] == question["correct_answer_label"].lower() if len(question["predicted_answer_label"]) > 0 else False

            video_answers[uid]["benchmark_dataset"].append(question)
        
        core_data[uid] = video_answers[uid]
    
        with open(results_file, "w") as wf:
            json.dump(core_data, wf, indent=4)
        
        
if __name__ == "__main__":
    fire.Fire(main)