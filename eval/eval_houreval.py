import fire
import json
from pathlib import Path
from eval.utils_videomme import VideoMMEDataset
from eval.utils_longvideobench import LongVideoBenchDataset
from eval.utils_mlvu import MLVUDataset
from tqdm import tqdm


def main(
    model_type="vamba",
    model_name_or_path="TIGER-Lab/Vamba-Qwen2-VL-7B",
    videomme_data_dir="/path/to/datasets/videomme",
    videomme_frames_dir=None,
    mlvu_data_dir="/path/to/datasets/MLVU/MLVU",
    longvideobench_data_dir="/path/to/datasets/LongVideoBench",
    num_frames=512,
    img_shortest_edge=256,
    img_longest_edge=480,
    max_img_seq_len=120000,
    do_resize=True,
    use_subtitle=False,
    results_dir="./output/eval/houreval",
    overwrite=False,
    # generation config
    max_new_tokens=512,
    do_sample=False,
    top_k=None,
    top_p=0.9,
    temperature=0.6,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "vamba":
        from tools.vamba_chat import Vamba
        model = Vamba(model_name_or_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    sample_config = {
        'filtered': True,
        'num_frames': num_frames,
        'sample_type': 'uniform',
        'model_patch_size': model.patch_size,
        'img_shortest_edge': img_shortest_edge,
        'img_longest_edge': img_longest_edge,
        'max_img_seq_len': max_img_seq_len,
        'do_resize': do_resize,
    }

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }

    dataset_scores = []
    for dataset_name in ['videomme', 'mlvu', 'longvideobench']:
        if dataset_name == 'videomme':
            dataset = VideoMMEDataset(
                videomme_data_dir,
                frames_path=videomme_frames_dir,
                sample_config=sample_config,
                use_subtitle=use_subtitle
            )
        elif dataset_name == 'mlvu':
            dataset = MLVUDataset(
                mlvu_data_dir,          
                sample_config=sample_config,
            )
        elif dataset_name == 'longvideobench':
            dataset = LongVideoBenchDataset(
                longvideobench_data_dir,
                sample_config=sample_config,
            )
        else:
            raise NotImplementedError("This daraset is not supported yet")

        core_data = {}

        model_save_path = "/".join(model_name_or_path.split("/")[-2:])
        results_file = Path(results_dir) / f"{num_frames}frames" / f"{model_save_path}" / f"{dataset_name}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
    
        if results_file.exists() and not overwrite:
            with open(results_file, "r") as rf:
                try:
                    core_data = json.load(rf)
                except json.JSONDecodeError:
                    core_data = {}
    
        for i in tqdm(range(len(core_data), len(dataset))):
            data = dataset[i]
            video_path = data['video_path']
            if video_path in core_data:
                continue

            if model_type == "kangaroo":
                images = (data["video"], data["durations"])
            else:
                images = data["video"]

            questions = data["questions"]
            video_answers = []   

            for question in questions:
                text = question["text"]
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
                question["correct"] = response[0] == question["answer"] or response[0] == question["answer"].lower() if len(response) > 0 else False

                video_answers.append(question)
            
            core_data[video_path] = video_answers
        
            with open(results_file, "w") as wf:
                json.dump(core_data, wf, indent=4)
            
        all_questions = []
        for answers in core_data.values():
            all_questions.extend(answers) 
        # print accuracy
        result = {"correct": 0, "total": 0}
        for item in all_questions:
            result["total"] += 1
            if item["correct"]:
                result["correct"] += 1
        all_correct = result["correct"]
        all_total = result["total"]
        print(f"{dataset_name} Overall Accuracy: {all_correct} / {all_total:.4f} = {all_correct / all_total:.4f}")
        dataset_scores.append({"correct":all_correct, "total":all_total})
    
    all_correct = sum([dataset_score['correct'] for dataset_score in dataset_scores])
    all_total = sum([dataset_score['total'] for dataset_score in dataset_scores])
    print(f"Overall Accuracy: {all_correct} / {all_total:.4f} = {all_correct / all_total:.4f}")
      
if __name__ == "__main__":
    fire.Fire(main)