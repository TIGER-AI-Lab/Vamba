import fire
import json
from pathlib import Path
from eval.utils_mvbench import MVBench_dataset
from tqdm import tqdm

def main(
    model_type="vamba",
    model_name_or_path="TIGER-Lab/Vamba-Qwen2-VL-7B",
    data_dir="/path/to/datasets/MVBench",
    num_frames=512,
    img_shortest_edge=256,
    img_longest_edge=480,
    max_img_seq_len=20000,
    do_resize=True,
    results_dir="./output/eval/mvbench",
    overwrite=False,
    # generation config
    max_new_tokens=512,
    do_sample=False,
    top_k=None,
    top_p=0.9,
    temperature=0.6,
):
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    json_dir = data_dir / "json"
    video_dir = data_dir / "video"
    results_dir.mkdir(parents=True, exist_ok=True)
    

    data_list = {
        "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", "perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", "perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
        "Character Order": ("character_order.json", "perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
    }
    
    for subtask in data_list:
        # add json_dir prefix and video_dir prefix
        json_path, video_path, video_type, has_start_end = data_list[subtask]
        video_path = str(video_dir / video_path)
        data_list[subtask] = (json_path, video_path, video_type, has_start_end)
    
    if model_type == "vamba":
        from tools.vamba_chat import Vamba
        model = Vamba(model_name_or_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }

    dataset = MVBench_dataset(
        json_dir,
        data_list,
        num_segments=num_frames,
        model_patch_size=model.patch_size,
        img_shortest_edge=img_shortest_edge,
        img_longest_edge=img_longest_edge,
        max_img_seq_len=max_img_seq_len,
        do_resize=do_resize,
    )
    assert len(dataset) == 4000, f"len(dataset) = {len(dataset)}"
        
    core_data = []

    model_save_path = "/".join(model_name_or_path.split("/")[-2:])
    results_file = Path(results_dir) / f"{num_frames}frames" / f"{model_save_path}.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    if results_file.exists() and not overwrite:
        with open(results_file, "r") as rf:
            for line in rf:
                core_data.append(json.loads(line))
    else:
        with open(results_file, "w") as wf:
            pass
    
    for i in tqdm(range(len(core_data), 4000)):
        data = dataset[i] 
        di = {"question": data["question"], "answer": data["answer"][0], "task_type": data["task_type"]}
        
        images = data["video"]
        question = data["question"]
        messages = [
            {
                "type": "pil_video",
                "content": images
            },
            {
                "type": "text",
                "content": f"<video> {question}",
            }
        ]
        response = model(messages, generation_config)
        response = response.lower()
        
        di["outputs"] = response
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
        di["correct"] = response[0] == data["answer"][0] or response[0] == data["answer"][0].lower() if len(response) > 0 else False
        
        with open(results_file, "a") as wf:
            json.dump(di, wf)
            wf.write("\n")
        core_data.append(di)
        
    # print accuracy
    task_type_dict = {}
    for item in core_data:
        task_type = item["task_type"]
        if task_type not in task_type_dict:
            task_type_dict[task_type] = {"correct": 0, "total": 0}
        task_type_dict[task_type]["total"] += 1
        if item["correct"]:
            task_type_dict[task_type]["correct"] += 1
    for task_type in task_type_dict:
        print(f"Task Type: {task_type}")
        print(f"Accuracy: {task_type_dict[task_type]['correct']} / {task_type_dict[task_type]['total']:.4f} = {task_type_dict[task_type]['correct'] / task_type_dict[task_type]['total']:.4f}")
        print()
    all_correct = sum([task_type_dict[task_type]["correct"] for task_type in task_type_dict])
    all_total = sum([task_type_dict[task_type]["total"] for task_type in task_type_dict])
    print(f"Overall Accuracy: {all_correct} / {all_total:.4f} = {all_correct / all_total:.4f}")
        
        
if __name__ == "__main__":
    fire.Fire(main)