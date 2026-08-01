import re
import torch
from tqdm import tqdm

def generate_answers(conf, model, tokenizer, dataset):
    dataset_length = conf["dataset"]["eval_length"]
    
    input_ids = torch.tensor(list(dataset["test"]["input_ids"]))
    attention_mask = torch.tensor(list(dataset["test"]["attention_mask"]))

    if dataset_length > 100:
        texts = []
        assert (dataset_length / 100).is_integer()
        num_batches = int(dataset_length / 100)
        for b in tqdm(range(num_batches)):
            input_sub_tensor = input_ids[(b * 100):((b+1) * 100), :]
            attention_sub_tensor = attention_mask[(b * 100):((b+1) * 100), :]
            batch_texts = batch_generate(model, tokenizer, input_sub_tensor, attention_sub_tensor)
            texts.extend(batch_texts)

    else:
        texts = batch_generate(model, tokenizer, input_ids, attention_mask)

    _, solutions = ground_truth_answers(dataset)
    return texts, solutions


def batch_generate(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=400,
            do_sample=False,
        )

    texts = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )

    return texts


def ground_truth_answers(dataset):
    problems = list(dataset["test"]["problem"])
    solutions = list(dataset["test"]["solution"])
    return problems, solutions


def extract_ground_truth(conf, solution: str):
    solution = solution.strip()
    if conf["eval"]["dataset"] == "gsm8k":
        assert "####" in solution
        solution_parts = solution.split("####")
    elif conf["eval"]["dataset"] == "meta-math":
        assert "The answer is:" in solution
        solution_parts = solution.split("The answer is:")
    else: raise NotImplementedError
    final_part = solution_parts[-1]
    assert final_part != ""
    return final_part.strip()


def process_decimals(value: str):
    if "," in value: 
        return float(value.replace(",",""))
    else: 
        return float(value)



def extract_prediction(response: str):
    ANSWER_PATTERN = re.compile(r"(?:####|The answer is:)\s*\$?(-?\d[\d,]*(?:\.\d+)?)")
    NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

    match = ANSWER_PATTERN.search(response)
    if match is not None:
        return process_decimals(match.group(1))

    if "####" in response and "The answer is:" not in response:
        before_separator = response.split("####")[0]
        numbers = NUMBER_PATTERN.findall(before_separator)
        if numbers:
            return process_decimals(numbers[-1])

    return None


def extract_floats(conf, response: str, solution: str, ground_truth = None):
    """
    response: generated response by the model
    solution: ground truth response in the dataset
    answer_value: ground truth value from extract_ground_truth to be translated into float()
    """
    if solution != None:
        assert ground_truth == None
        answer_value = extract_ground_truth(conf, solution)
        ground_truth = process_decimals(answer_value)
    else: 
        assert conf == None
        ground_truth = float(ground_truth)

    prediction = extract_prediction(response)
    if prediction is None:
        print(response)
    return prediction, ground_truth