import re
import ast
import json
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset

from src.poet.dataset import retrieve_dataset
from src.poet.directories import write_feature_storage_dir
from src.poet.compare_answers import extract_floats


def system_prompt(conf):
    max_words = conf["interpretability"]["max_words_description"]
  
    template = f"""
    You are a technical analyst identifying latent semantic features in mathematical text.
    
    All examples come from mathematics. Do NOT describe features as “mathematical” or “math-related”.

    Your task is to infer the most specific shared semantic role, operation, or function that is common to all examples.

    Constraints:
    - Describe the latent concept, not surface tokens or model behavior.
    - Your description must distinguish these spans from other mathematical text.
    - Do not mention tokens, models, or activations.
    - Prefer functional roles within proofs, derivations, or definitions.
    - Avoid generic abstractions like “reasoning”, “formal logic”, or “symbolic manipulation” unless unavoidable.
    - Output exactly one sentence (max {max_words} words).
    - If no specific shared role exists, reply exactly: "No coherent concept."
    - Only reply "No coherent concept." if the spans do not share a specific semantic function beyond being mathematical.
    """
    return template


def select_context_window(conf, tokenized_example, token_idx):
    window = conf["interpretability"]["window_leading_trailing_tokens"]
    start = max(0, token_idx - window)
    end = min(len(tokenized_example), token_idx + window + 1)
    span_ids = tokenized_example[start:end]
    return span_ids


def gemma_decode(gemma_tokenizer, span_ids, skip_special_tokens = False):

    spans = gemma_tokenizer.batch_decode(
            span_ids,
            skip_special_tokens=skip_special_tokens
        )
    return spans
    

def retrieve_feature_examples(conf, heap):
    num_examples = conf["interpretability"]["num_spans_per_feature"]
    if len(heap) < num_examples: 
        return None
    
    assert heap[0][0] <= heap[-1][0] # assert increasing
    
    test_set_list = []
    test_set_idx_set = set()
    # heap.reverse()
    random.shuffle(heap)
    for ftr in heap:
        _, test_set_idx, token_idx = ftr
        if test_set_idx not in test_set_idx_set:
            test_set_list.append((test_set_idx, token_idx))
        test_set_idx_set.add(test_set_idx)
        if len(test_set_list) == num_examples:
            break
    
    assert len(test_set_list) == len(list(test_set_idx_set))
    if len(test_set_list) < num_examples:
        return None
    
    return test_set_list


def build_feature_prompt(
    gemma_tokenizer,
    tokenized_spans,
):
    spans = gemma_decode(gemma_tokenizer, tokenized_spans)

    spans_block = "\n".join(
        f"{i+1}. {span}" for i, span in enumerate(spans)
    )

    prompt = "Spans:\n" + spans_block + "\n\nAnswer:"    
        
    return prompt, spans_block


def write_messages(prompt, system):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return messages


def retrieve_testset_tokenizer(conf):
    gemma_tokenizer = AutoTokenizer.from_pretrained(conf["model"]["name"])

    dataset = retrieve_dataset(conf, gemma_tokenizer, split="test")
    testset = dataset["test"]
    return testset, gemma_tokenizer


def tokenize_spans(conf, dataset, test_set_list):
    examples = []
    for (test_set_idx, _) in test_set_list:
        example = dataset[test_set_idx]["input_ids"]
        examples.append(example)

    tokenized_spans = []
    for loop_idx, (_, token_idx) in enumerate(test_set_list):
        span_ids = select_context_window(conf, examples[loop_idx], token_idx)
        tokenized_spans.append(span_ids)
    
    return tokenized_spans


def spans_for_feature(conf, ftr_index):
    dataset, gemma_tokenizer = retrieve_testset_tokenizer(conf)

    feature_dict = retrieve_feature_dict(conf)
    heap = feature_dict[ftr_index]
    test_set_list = retrieve_feature_examples(conf, heap)
    if test_set_list == None:
        return f"Not enough examples exist for this feature!\nWe require contexts from at least {conf['interpretability']['num_spans_per_feature']} different observation points."
    
    tokenized_spans = tokenize_spans(conf, dataset, test_set_list)
    _, spans_block = build_feature_prompt(gemma_tokenizer, tokenized_spans)
    return spans_block


def create_prompt_dataset(conf):
    feature_dict = retrieve_feature_dict(conf)

    random.seed(19012026)
    shuffled_features = sorted(feature_dict.keys()).copy()
    random.shuffle(shuffled_features)

    dataset, gemma_tokenizer = retrieve_testset_tokenizer(conf)

    system = system_prompt(conf)

    messages_list = []
    final_feature_indices = []
    for ftr_index in shuffled_features:
        heap = feature_dict[ftr_index]
        test_set_list = retrieve_feature_examples(conf, heap)
        if test_set_list == None: 
            continue
        # examples = []
        # for (test_set_idx, _) in test_set_list:
        #     example = dataset[test_set_idx]["input_ids"]
        #     examples.append(example)

        # tokenized_spans = []
        # for loop_idx, (_, token_idx) in enumerate(test_set_list):
        #     span_ids = select_context_window(conf, examples[loop_idx], token_idx)
        #     tokenized_spans.append(span_ids)
        tokenized_spans = tokenize_spans(conf, dataset, test_set_list)
        prompt, _ = build_feature_prompt(gemma_tokenizer, tokenized_spans)
        message = write_messages(prompt, system)
        messages_list.append(message)
        final_feature_indices.append(ftr_index)
        if len(messages_list) == conf["interpretability"]["num_eval_features"]:
            break
    
    prompt_dataset = Dataset.from_dict({"chat": messages_list})
    return prompt_dataset, final_feature_indices


def retrieve_feature_dict(conf):
    json_dir = write_feature_storage_dir(conf)
    json_path = json_dir + "/feature_dict.jsonl"
    
    feature_dict = {}

    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)

            ftr_index = data["feature_idx"]
            entries = data["entries"]

            feature_dict[ftr_index] = [tuple(e) for e in entries]
    
    return feature_dict


def retrieve_feature_counts(conf):
    json_dir = write_feature_storage_dir(conf)
    json_path = json_dir + "/feature_counter.jsonl"
    
    feature_counts = {}

    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)

            ftr_index = data["feature_idx"]
            count = data["count"]

            feature_counts[ftr_index] = count
    
    return feature_counts


def retrieve_feature_explanations(conf):
    json_dir = write_feature_storage_dir(conf)
    json_path = json_dir + "/feature_explanations.jsonl"
    
    feature_explanations = {}

    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)

            ftr_index = data["feature_idx"]
            explanation = data["explanation"]

            feature_explanations[ftr_index] = explanation
    
    return feature_explanations


def dataset_to_inputs(llama_tokenizer, prompt_dataset):
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "left"


    texts = [
        llama_tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        for chat in prompt_dataset["chat"]
    ]

    llama_enc = llama_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False
    )

    return llama_enc


def add_to_explanations_dictionary(model, llama_tokenizer, input_ids_tensor, attention_mask_tensor, final_feature_indices, explanations_dict):
    terminators = [
        llama_tokenizer.eos_token_id,
        llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_new_tokens=24,
            eos_token_id=terminators,
            do_sample=False,
        )
    
    assert len(final_feature_indices) == input_ids_tensor.shape[0]

    for loop_idx, ftr_index in enumerate(final_feature_indices):
        prompt_len = attention_mask_tensor[loop_idx].sum().item()
        response = outputs[loop_idx][prompt_len:]

        decoded = llama_tokenizer.decode(response, skip_special_tokens=False)
        decoded = decoded.split("<|eot_id|><|eot_id|>")[0]
        marker = "assistant<|end_header_id|>" # "Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        letter_idx = decoded.find(marker)
        if letter_idx != -1:
            decoded = decoded[letter_idx + len(marker):]
        
        decoded = decoded.strip()

        while decoded[-len("<|eot_id|>"):] == "<|eot_id|>":
            decoded = decoded[:-len("<|eot_id|>")]


        decoded = decoded.strip()

        explanations_dict[ftr_index] = decoded

    return explanations_dict


########## EMBEDDINGS.PY ##########
def relevant_prefixes():
    prefixes = [
        "The spans describe",
        "The latent concept shared among these spans is", 
        "The spans share",
        "The shared semantic role is",
        "The common latent concept is",
    ]
    return prefixes


def remove_prefix(text):
    prefixes = relevant_prefixes()

    t = text.lower().strip()
    for p in prefixes:
        if t.startswith(p):
            return t[len(p):].strip()
    return t



########## INTERPRETABILITY_SCORE.PY ##########
def system_prompt_score(conf):
    correct_snippets = conf["interpretability"]["correct_snippets"]
    template = f"""
    You are evaluating the interpretability of an explanation with respect to a set of text snippets.

    You will be given:
    - A list of 8 labeled text snippets (1, 2, 3, 4, 5, 6, 7, 8)
    - A single explanation describing a specific concept, feature, or pattern

    Your task is to identify which snippet the explanation applies to MOST STRONGLY.
    Exactly {correct_snippets} of the provided snippets is correct.

    Guidelines:
    - Select ONLY the {correct_snippets} snippet for which the explanation is clearly and directly applicable.
    - Do NOT select snippets where the explanation is only weakly, indirectly, or ambiguously related.
    - You must output exactly one label.
    - Do not explain your reasoning.

    Output format requirements (MANDATORY):
    - Output must be a Python-style list of snippet labels.
    - Example valid outputs: [1], [5]
    - Do NOT include any additional text, punctuation, or explanation.
    - Do NOT include quotes around labels.
    - Do NOT include reasoning or commentary.

    If the explanation best fits snippet 1, your entire output must be:
    [1]
    """
    return template


def dataset_score(conf):
    feature_dict = retrieve_feature_dict(conf)
    feature_explanations = retrieve_feature_explanations(conf)

    num_total_snippets = conf["interpretability"]["total_snippets"]
    num_correct_snippets = conf["interpretability"]["correct_snippets"]
    num_incorrect_snippets = num_total_snippets - num_correct_snippets
    
    random.seed(23012026)
    shuffled_features = sorted(feature_explanations.keys()).copy()
    random.shuffle(shuffled_features)

    dataset, gemma_tokenizer = retrieve_testset_tokenizer(conf)

    decoy_indices = list(range(len(dataset)))
    
    random.shuffle(decoy_indices)
    # decoy_indices = random.sample(range(len(dataset)), 5000)

    explanation_decoy_dict = {}

    prefixes = relevant_prefixes()

    total_evaluations = conf["interpretability"]["total_evaluations"]

    feature_indices_with_prefix = []
    for ftr_index in shuffled_features:
        explanation = feature_explanations[ftr_index]
        for p in prefixes:
            if p in explanation:
                feature_indices_with_prefix.append(ftr_index)
                break
        if len(feature_indices_with_prefix) == total_evaluations:
            break

    assert isinstance(feature_indices_with_prefix, list)
    assert len(feature_indices_with_prefix) == total_evaluations
    idx_in_shuffled_features = 0
    correct_idx = feature_indices_with_prefix[idx_in_shuffled_features]
    explanation_decoy_dict[correct_idx] = []
    for test_set_idx in decoy_indices:
        input_ids = dataset[test_set_idx]["input_ids"]
        assert isinstance(input_ids, list)
        if correct_idx not in input_ids:
            explanation_decoy_dict[correct_idx] += [test_set_idx]
        if len(explanation_decoy_dict[correct_idx]) == num_incorrect_snippets:
            idx_in_shuffled_features += 1
            if idx_in_shuffled_features < total_evaluations:
                correct_idx = feature_indices_with_prefix[idx_in_shuffled_features]
                explanation_decoy_dict[correct_idx] = []
        if idx_in_shuffled_features == total_evaluations:
            break

    assert len(explanation_decoy_dict) == total_evaluations
    correct_indices_dict = {}
    all_examples_dict = {}
    for ftr_index in explanation_decoy_dict.keys():
        heap = feature_dict[ftr_index]
        assert heap[0][0] <= heap[-1][0]
        entries = heap[-num_correct_snippets] # random.sample(heap, num_correct_snippets)
        if num_correct_snippets > 1: correct_examples = [test_set_idx for (_, test_set_idx, _) in entries]
        else: correct_examples = [entries[1]]
        all_examples = explanation_decoy_dict[ftr_index] + correct_examples
        random.shuffle(all_examples)
        assert len(all_examples) == num_total_snippets
        all_examples_dict[ftr_index] = all_examples
        correct_indices = [all_examples.index(ex) for ex in correct_examples]
        correct_indices_dict[ftr_index] = correct_indices
        assert len(correct_indices_dict[ftr_index]) == num_correct_snippets

    messages_list = []
    system = system_prompt_score(conf)
    assert len(all_examples_dict) == total_evaluations
    for ftr_index in all_examples_dict.keys():
        all_examples = all_examples_dict[ftr_index]
        all_tokenized_examples = [dataset[test_set_idx]["input_ids"] for test_set_idx in all_examples]
        natural_language_examples = gemma_decode(gemma_tokenizer, all_tokenized_examples, skip_special_tokens=True)
        assert len(natural_language_examples) == num_total_snippets
        language_blocks = "\n".join(
            f"{i+1}. {nl}" for i, nl in enumerate(natural_language_examples)
        )
        explanation = feature_explanations[ftr_index]
        prompt = "Text snippets:\n" + language_blocks + "\n\nExplanation:\n" + explanation + "\n\nAnswer:"

        message = write_messages(prompt, system)
        messages_list.append(message)
    
    assert len(messages_list) == len(list(all_examples_dict.keys()))

    prompt_dataset = Dataset.from_dict({
        "chat": messages_list,
        "feature_index": list(all_examples_dict.keys()),
        })
    return prompt_dataset, correct_indices_dict


def add_to_accuracy_dictionary(conf, model, llama_tokenizer, input_ids_tensor, attention_mask_tensor, correct_indices_dict, accuracy_dict):
    terminators = [
        llama_tokenizer.eos_token_id,
        llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    num_correct_snippets = conf["interpretability"]["correct_snippets"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=False,
        )
    
    assert len(correct_indices_dict) == input_ids_tensor.shape[0]

    batch_overlap = 0

    for loop_idx, ftr_index in enumerate(correct_indices_dict):
        prompt_len = attention_mask_tensor[loop_idx].sum().item()
        
        response = outputs[loop_idx][prompt_len:]

        decoded = llama_tokenizer.decode(response, skip_special_tokens=False)
        
        marker = "assistant<|end_header_id|>" # "Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        letter_idx = decoded.find(marker)
        if letter_idx != -1:
            decoded = decoded[letter_idx + len(marker):]
        
        decoded = decoded.split("<|eot_id|><|eot_id|>")[0]
        decoded = decoded.strip()
        
        # if loop_idx == 1: 
        #     test_decode = llama_tokenizer.decode(outputs[loop_idx], skip_special_tokens=False)
        #     print(test_decode)

        while decoded[-len("<|eot_id|>"):] == "<|eot_id|>":
            decoded = decoded[:-len("<|eot_id|>")]

        pattern = r"\[\s*\d+\s*\]" # r"\[\s*\d+,\s*\d+\s*\]" # r"\[(?:\d+?:,\s*\d+)*)?\]$"

        choices = re.findall(pattern, decoded)
        if len(choices) != 1:
            print(ftr_index)
            print(decoded, flush = True)
            choices = ["[-100]"]
        assert len(choices) == 1
        choice = choices[0]

        choice = choice.strip()
        choice_list = ast.literal_eval(choice)
        assert isinstance(choice_list, list)
        assert len(choice_list) == num_correct_snippets
        correct_indices = correct_indices_dict[ftr_index]
        assert len(correct_indices) == num_correct_snippets

        choice_list = [c-1 for c in choice_list]
        overlap = len(list(set(choice_list) & set(correct_indices)))
        accuracy_dict[ftr_index] = {
            "choice_list": choice_list,
            "correct_indices": correct_indices,
            "overlap": overlap
        }
        batch_overlap += overlap
    
    return accuracy_dict, batch_overlap


def jerrys_dataset(first_name):
    dataset_dict = {
        "0": {"sentence": [f"{first_name} uses 10 weight plates each weighing 30 pounds on an exercise machine. This exercise machine uses special technology to make the weights 20% heavier on the lowering portion. How heavy did the weights feel when being lowered?"], "ground_truth": 260,},
        "1": {"sentence": [f"{first_name} sold clips to 48 of his friends in April, and then he sold half as many clips in May. How many clips did {first_name} sell altogether in April and May?"], "ground_truth": 72,},
        "2": {"sentence": [f"{first_name} writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"], "ground_truth": 624,},
        "3": {"sentence": [f"{first_name} is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"], "ground_truth": 48,},
        "4": {"sentence": [f"{first_name} created a care package to send to his brother, who was away at boarding school. {first_name} placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?"], "ground_truth": 16,},
        "5": {"sentence": [f"{first_name} can read 8 pages of a book in 20 minutes. How many hours will it take him to read 120 pages?"], "ground_truth": 5,},
        "6": {"sentence": [f"{first_name} creates a media empire. He creates a movie for $2000. Each DVD cost $6 to make. He sells it for 2.5 times that much. He sells 500 movies a day for 5 days a week. How much profit does he make in 20 weeks?"], "ground_truth": 448000,},
        "7": {"sentence": [f"{first_name} is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?"], "ground_truth": 5,},
        "8": {"sentence": [f"{first_name} buys 10 packs of magic cards. Each pack has 20 cards and 1/4 of those cards are uncommon. How many uncommon cards did he get?"], "ground_truth": 50,},
        "9": {"sentence": [f"{first_name} took 9 pills a day for 14 days. Of these 9 pills, 4 pills cost $1.50 each, and the other pills each cost $5.50 more. How much did he spend in total on the pills?"], "ground_truth": 41,},
        "10": {"sentence": [f"{first_name} went to a shop to buy some groceries. He bought some bread for $2, butter for $3, and juice for two times the price of the bread. He had $15 for his shopping. How much money did {first_name} have left?"], "ground_truth": 6,},
        "11": {"sentence": [f"{first_name} has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does {first_name} have in total?"], "ground_truth": 15,},
        "12": {"sentence": [f"{first_name} has five more roommates than twice as many as Bob. If Bob has 10 roommates, how many roommates does {first_name} have?"], "ground_truth": 25,},
        "13": {"sentence": [f"{first_name} owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away?"], "ground_truth": 10,},
        "14": {"sentence": [f"{first_name} eats 1 apple a day for two weeks. Over the next three weeks, he eats the same number of apples as the total of the first two weeks. Over the next two weeks, he eats 3 apples a day. Over these 7 weeks, how many apples does he average a week?"], "ground_truth": 10,},
        "15": {"sentence": [f"{first_name} bought 2 soft drinks for$ 4 each and 5 candy bars. He spent a total of 28 dollars. How much did each candy bar cost?"], "ground_truth": 4,},
        "16": {"sentence": [f"{first_name} has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?"], "ground_truth": 160,},
        "17": {"sentence": [f"{first_name} is throwing a huge Christmas party. He invites 30 people. Everyone attends the party, and half of the guests bring a plus one (one other person). He plans to serve a 3-course meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?"], "ground_truth": 135,},
        "18": {"sentence": [f"{first_name} volunteers at a shelter twice a month for 3 hours at a time. How many hours does he volunteer per year?"], "ground_truth": 72,},
        "19": {"sentence": [f"{first_name} puts $25 in his piggy bank every month for 2 years to save up for a vacation. He had to spend $400 from his piggy bank savings last week to repair his car. How many dollars are left in his piggy bank?"], "ground_truth": 200,},
        "20": {"sentence": [f"{first_name} has 16 toy cars, and the number of cars he has increases by 50% every year. How many toy cars will {first_name} have in three years?"], "ground_truth": 54,},
        "21": {"sentence": [f"{first_name} just turned 12 and started playing the piano. His friend Sheila told him about the 10,000-hour rule which says, after 10,000 hours of practice, you become an expert or master in your field. If {first_name} wants to become a piano expert before he is 20, how many hours a day will he need to practice if he practices every day, Monday – Friday, and takes two weeks off for vacation each year?"], "ground_truth": 5,},
        "22": {"sentence": [f"In 6 months Bella and {first_name} will be celebrating their 4th anniversary. How many months ago did they celebrate their 2nd anniversary?"], "ground_truth": 18,},
        "23": {"sentence": [f"{first_name} has some coins. He has 2 more quarters than nickels and 4 more dimes than quarters. If he has 6 nickels, how much money does he have?"], "ground_truth": 350,},
        "24": {"sentence": [f"{first_name} starts exercising at home during quarantine. To start, he decides to do 3 sets of 15 push-ups each. Near the end of the third set, he gets tired and does 5 fewer push-ups. How many push-ups did he do in total?"], "ground_truth": 40,},
        "25": {"sentence": [f"{first_name} likes to collect model trains. He asks for one for every birthday of his, and asks for two each Christmas. {first_name} always gets the gifts he asks for, and asks for these same gifts every year for 5 years. At the end of the 5 years, his parents give him double the number of trains he already has. How many trains does {first_name} have now?"], "ground_truth": 45,},
        "26": {"sentence": [f"{first_name} has $5000. He spends $2800 on a new motorcycle, and then spends half of what's left on a concert ticket. {first_name} then loses a fourth of what he has left. How much money does he have left?"], "ground_truth": 825,},
        "27": {"sentence": [f"{first_name} has 7 one-dollar bills, 4 five-dollar bills, 2 ten-dollar bills, and 1 twenty-dollar bill. He goes to buy peanuts, which cost $3 a pound. He buys what he wants and has $4 in change. He plans to eat the peanuts all in one week. How many pounds does he eat on average per day?"], "ground_truth": 3,},
        "28": {"sentence": [f"{first_name} has a terrible toothache and decides to buy some painkillers from the store. He picks up a bottle of 50 pills and takes them home. He takes 2 pills each day three times a day for the first 2 days, before cutting this amount in half for the next 3 days. On the sixth day, he takes a final 2 pills in the morning and ends up feeling better. How many pills are left in the bottle?"], "ground_truth": 27,},
        "29": {"sentence": [f"{first_name} picked a handful of dandelion puffs. He gave 3 to his mom, another 3 to his sister, 5 to his grandmother, and 2 to his dog. Then, he divided the remaining dandelion puffs equally among his 3 friends. How many dandelion puffs did each friend receive if he originally picked 40 dandelion puffs?"], "ground_truth": 9,},
    }
    return dataset_dict


def name_map():
    name_dict = {
        "Mike": 11681, # Michael
        "Jason": 59158,
        "Jacob": 42983, # Jake
        "Jerry": 44109, # Jeremy
        "James": 37532,
        "Robert": 22710,
        "Jordan": 28504,
        "Jackson": 38384,
        "Paul": 26534,
        "David": 28951,
        "Andrew": 26342,
        "Gary": 17911, # Garre Garret
    }
    return name_dict


def alternative_names():
    alt_dict = {
        "Jason": ["Jason", "Jase"],
        "Mike": ["Michael", "Mike", "Mikey"],
        "Jacob": ["Jacob", "Jake", "Jakob"],
        "Jerry": ["Jerry", "Jeremy", "Jermey", "Jerome"],
        "James": ["James", "Jim", "Jimmy", "Jamie"],
        "Robert": ["Robert", "Rob", "Robbie"],
        "Jordan": ["Jordan", "Jordy"],
        "Jackson": ["Jackson", "Jack", "Jax"],
        "Paul": ["Paul", "Pauly"],
        "David": ["David", "Dave", "Davey"],
        "Andrew": ["Andrew", "Andy"],
        "Gary": ["Gary", "Garre", "Garret", "Garrett", "Garry"]
    }
    return alt_dict


def name_in_generated(first_name, gen):
    names = alternative_names()[first_name]
    for name in names:
        assert isinstance(name, str)
        assert isinstance(gen, str)
        if name in gen:
            return 1
    return 0


def evaluate_generated_texts(texts, dataset_dict, drop_name, include_name):
    count = 0
    drop_names = 0
    include_names = 0
    for jerry_doc in dataset_dict.keys():
        jerry_int = int(jerry_doc)
        text = texts[jerry_int]
        generated_parts = text.split("\n\nSolution:\n")
        if len(generated_parts) == 0:
            continue
        generated = generated_parts[1]
        ground_truth = dataset_dict[jerry_doc]["ground_truth"]
        pred, gt = extract_floats(conf=None, response = generated, solution = None, ground_truth=ground_truth)
        drop_names += name_in_generated(drop_name, generated)
        include_names += name_in_generated(include_name, generated)
        if pred == gt:
            count += 1
    return count, drop_names, include_names