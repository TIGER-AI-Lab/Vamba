import fire
import json
from pathlib import Path
from eval.utils_nextqa import NextQADataset
from tqdm import tqdm

def main(
    model_type="longva",
    model_name_or_path="/cpfs/data/user/weiming/checkpoints/lmms-lab/LongVA-7B",
    data_dir="/map-vepfs/weiming/datasets/NExTQA",
    num_frames=512,
    img_shortest_edge=256,
    img_longest_edge=480,
    max_img_seq_len=16500,
    do_resize=False,
    results_dir="./output/eval/nextqa",
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
        'num_frames': num_frames,
        'sample_type': 'uniform',
        'model_patch_size': model.patch_size,
        'img_shortest_edge': img_shortest_edge,
        'img_longest_edge': img_longest_edge,
        'max_img_seq_len': max_img_seq_len,
        'do_resize': do_resize,
    }

    dataset = NextQADataset(
        data_dir,
        sample_config=sample_config,
    )

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }

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
    
    for i in tqdm(range(len(core_data), len(dataset))):
        data = dataset[i]
        meta = data["meta"]
        di = {
            "question": data["text"],
            "answer": meta["answer"],
            "answer_text": meta["answer_text"],
            "type": meta["type"],
        }
        
        images = data["video"]
        question = data["text"]
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
        di["response"] = response
        
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
        di["correct"] = response[0] == di["answer"] or response[0] == di["answer"].lower() if len(response) > 0 else False
        
        with open(results_file, "a") as wf:
            json.dump(di, wf)
            wf.write("\n")
        core_data.append(di)
        
    # print accuracy
    question_type_dict = {}
    for item in core_data:
        question_type = item["type"]
        if question_type not in question_type_dict:
            question_type_dict[question_type] = {"correct": 0, "total": 0}
        question_type_dict[question_type]["total"] += 1
        if item["correct"]:
            question_type_dict[question_type]["correct"] += 1
    for question_type in question_type_dict:
        print(f"Question Type: {question_type}")
        print(f"Accuracy: {question_type_dict[question_type]['correct']} / {question_type_dict[question_type]['total']:.4f} = {question_type_dict[question_type]['correct'] / question_type_dict[question_type]['total']:.4f}")
        print()
    all_correct = sum([1 if item["correct"] else 0 for item in core_data])
    all_total = len(core_data)
    print(f"Overall Accuracy: {all_correct} / {all_total:.4f} = {all_correct / all_total:.4f}")
        
        
if __name__ == "__main__":
    fire.Fire(main)