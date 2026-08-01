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
    num_spans = conf["interpretability"]["num_spans_per_feature"]
    mathematical_system_prompt = conf["interpretability"]["mathematical_system_prompt"]
    model_name = conf["model"]["name"]
  
    if mathematical_system_prompt:
        template = f"""
        You are labeling latent features of a language model trained on grade-school math word problems.

        You will see {num_spans} numbered spans from these problems. In each span, exactly one token is wrapped in double square brackets, like [[this]]. The feature you are labeling fires on precisely the bracketed token; the surrounding text is context only.

        Procedure:
        1. Resolve each bracketed token to the full word or entity it belongs to. A bracketed fragment
        of a longer word stands for the whole word: [[umbr]] inside "umbrella" means umbrella.
        2. Find the single most specific concept shared by these resolved words across (nearly) all {num_spans} spans.
        3. Name that concept as specifically as the evidence allows: a concrete entity (a particular
        first name, animal, food or drink, place, object); else a specific mathematical element
        (a particular unit, operation word, number format, quantity type); else a structural role.

        Constraints:
        - Your answer must name something you can point to in the bracketed tokens themselves. Never
        answer with a word that does not appear in, or directly behind, the bracketed tokens.
        - The illustration word in these instructions ("umbrella") is not evidence; no word from these
        instructions may appear in your answer unless it genuinely occurs in the spans.
        - State the most specific level the spans support: the particular name, animal, or unit itself,
        not its category ("first names", "animals", "units").
        - Do NOT describe features as "mathematical", "math-related", "word problems", "numbers", or "reasoning".
        - Do not mention the brackets, tokens, models, or activations in your answer.
        - Output exactly one sentence (max {max_words} words) in the exact form: "The shared latent concept is ...".
        - Your sentence must contain the specific shared word or name itself, copied verbatim from the
        spans and wrapped in quotation marks — category plus instance, e.g.: The shared latent concept
        is the first name "X". / the unit "X". / the animal "X".
        - If the bracketed tokens resolve to different words that share a category, name the category and
        quote two or three of the actual words, e.g.: first names such as "X" and "Y".
        - A category alone ("a first name", "an animal", "a unit") is never an acceptable answer. If you
        cannot quote the specific word(s), reply exactly: "No coherent concept."        
        """

    else:
        assert model_name == "meta-llama/Llama-3.2-1B"
        template = f"""
        You are a technical analyst identifying latent semantic features in mathematical text.
        
        All examples come from mathematics. Do NOT describe features as “mathematical” or “math-related”.

        In each span, exactly one token is wrapped in double square brackets, like [[this]]. Identify the single most specific concept that the bracketed tokens share across (nearly) all {num_spans} spans, and state that concept directly.

        Constraints:
        - Describe the latent concept, not surface tokens or model behavior.
        - Your description must distinguish these spans from other mathematical text.
        - Name the concept directly; do not mention the brackets, tokens, spans, models, or this task in your answer.
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
    center_offset = token_idx - start
    return span_ids, center_offset


def gemma_decode(gemma_tokenizer, span_ids, skip_special_tokens = False):

    spans = gemma_tokenizer.batch_decode(
            span_ids,
            skip_special_tokens=skip_special_tokens
        )
    return spans


def decode_marked_span(gemma_tokenizer, span_ids, center_offset, marker_left="[[", marker_right="]]", skip_special_tokens=False):
    prefix = gemma_tokenizer.decode(span_ids[:center_offset], skip_special_tokens=skip_special_tokens)
    activating = gemma_tokenizer.decode(span_ids[center_offset:center_offset + 1], skip_special_tokens=skip_special_tokens)
    suffix = gemma_tokenizer.decode(span_ids[center_offset + 1:], skip_special_tokens=skip_special_tokens)

    lead_ws = activating[:len(activating) - len(activating.lstrip())]
    core = activating.strip()
    marked = f"{lead_ws}{marker_left}{core}{marker_right}"
    return f"{prefix}{marked}{suffix}"


def retrieve_feature_examples(conf, heap):
    num_examples = conf["interpretability"]["num_spans_per_feature"]
    if len(heap) < num_examples: 
        return None
    
    assert heap[0][0] <= heap[-1][0]
    
    test_set_list = []
    test_set_idx_set = set()
    for ftr in reversed(heap):
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
    conf,
    gemma_tokenizer,
    tokenized_spans,
):
    mark_activation = True
    
    if mark_activation:
        spans = [
            decode_marked_span(gemma_tokenizer, span_ids, center_offset)
            for (span_ids, center_offset) in tokenized_spans
        ]
    else:
        spans = gemma_decode(gemma_tokenizer, [span_ids for (span_ids, _) in tokenized_spans])

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
        span_ids, center_offset = select_context_window(conf, examples[loop_idx], token_idx)
        tokenized_spans.append((span_ids, center_offset))

    return tokenized_spans


def spans_for_feature(conf, ftr_index):
    dataset, gemma_tokenizer = retrieve_testset_tokenizer(conf)

    feature_dict = retrieve_feature_dict(conf)
    heap = feature_dict[ftr_index]
    test_set_list = retrieve_feature_examples(conf, heap)
    if test_set_list == None:
        return f"Not enough examples exist for this feature!\nWe require contexts from at least {conf['interpretability']['num_spans_per_feature']} different observation points."
    
    tokenized_spans = tokenize_spans(conf, dataset, test_set_list)
    _, spans_block = build_feature_prompt(conf, gemma_tokenizer, tokenized_spans)
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

        tokenized_spans = tokenize_spans(conf, dataset, test_set_list)
        prompt, _ = build_feature_prompt(conf, gemma_tokenizer, tokenized_spans)
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
    mathematical_system_prompt = conf["interpretability"]["mathematical_system_prompt"]
    json_dir = write_feature_storage_dir(conf)
    if mathematical_system_prompt:
        json_path = json_dir + "/feature_explanations.jsonl"
    else: 
        json_path = json_dir + "/real_world_feature_explanations.jsonl"

    
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
            max_new_tokens=30,
            eos_token_id=terminators,
            do_sample=False,
        )
    
    assert len(final_feature_indices) == input_ids_tensor.shape[0]

    for loop_idx, ftr_index in enumerate(final_feature_indices):
        prompt_len = attention_mask_tensor[loop_idx].sum().item()
        response = outputs[loop_idx][prompt_len:]

        decoded = llama_tokenizer.decode(response, skip_special_tokens=False)
        decoded = decoded.split("<|eot_id|><|eot_id|>")[0]
        marker = "assistant<|end_header_id|>"
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
        "The shared latent concept is",
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
    total_snippets = conf["interpretability"]["total_snippets"]
    snippet_labels = ", ".join(str(i + 1) for i in range(total_snippets))
    template = f"""
    You are evaluating the interpretability of an explanation with respect to a set of text snippets.

    You will be given:
    - A list of {total_snippets} labeled text snippets ({snippet_labels})
    - A single explanation describing a specific concept, feature, or pattern

    Your task is to identify which snippet the explanation applies to MOST STRONGLY.
    Exactly {correct_snippets} of the provided snippets is correct.

    Guidelines:
    - Select ONLY the single snippet for which the explanation is clearly and directly applicable.
    - You must output exactly one label.
    - Do not explain your reasoning.

    Output format requirements (MANDATORY):
    - Output must be a Python-style list of snippet labels.
    - Example valid outputs: [1], [{total_snippets}]
    - Do NOT include any additional text, punctuation, or explanation.
    - Do NOT include quotes around labels.
    - Do NOT include reasoning or commentary.

    If the explanation best fits snippet 2, your entire output must be:
    [2]
    """
    return template


def dataset_score(conf):
    feature_dict = retrieve_feature_dict(conf)
    feature_explanations = retrieve_feature_explanations(conf)

    num_total_snippets = conf["interpretability"]["total_snippets"]
    num_correct_snippets = conf["interpretability"]["correct_snippets"]
    assert num_correct_snippets == 1
    num_incorrect_snippets = num_total_snippets - num_correct_snippets

    random.seed(23012026)
    shuffled_features = sorted(feature_explanations.keys()).copy()
    random.shuffle(shuffled_features)

    dataset, gemma_tokenizer = retrieve_testset_tokenizer(conf)

    decoy_indices = list(range(len(dataset)))

    random.shuffle(decoy_indices)

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
    
    num_needed = total_evaluations * num_incorrect_snippets
    assert len(dataset) > num_needed, (
        f"decoy pool ({len(dataset)}) cannot cover {total_evaluations} evaluations x {num_incorrect_snippets} decoys")

    idx_in_shuffled_features = 0
    correct_ftr_idx = feature_indices_with_prefix[idx_in_shuffled_features]
    explanation_decoy_dict[correct_ftr_idx] = []
    heap_examples = {t for (_, t, _) in feature_dict[correct_ftr_idx]}
    for test_set_idx in decoy_indices:      
        if test_set_idx not in heap_examples:
            explanation_decoy_dict[correct_ftr_idx] += [test_set_idx]
        if len(explanation_decoy_dict[correct_ftr_idx]) == num_incorrect_snippets:
            idx_in_shuffled_features += 1
            if idx_in_shuffled_features < total_evaluations:
                correct_ftr_idx = feature_indices_with_prefix[idx_in_shuffled_features]
                heap_examples = {t for (_, t, _) in feature_dict[correct_ftr_idx]}
                explanation_decoy_dict[correct_ftr_idx] = []
        if idx_in_shuffled_features == total_evaluations:
            break

    assert len(explanation_decoy_dict) == total_evaluations
    correct_indices_dict = {}
    all_examples_dict = {}
    for ftr_index in explanation_decoy_dict.keys():
        heap = feature_dict[ftr_index]
        assert heap[0][0] <= heap[-1][0]
        entries = heap[-num_correct_snippets:]
        correct_examples = [test_set_idx for (_, test_set_idx, _) in entries]
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
        
        marker = "assistant<|end_header_id|>"
        letter_idx = decoded.find(marker)
        if letter_idx != -1:
            decoded = decoded[letter_idx + len(marker):]
        
        decoded = decoded.split("<|eot_id|><|eot_id|>")[0]
        decoded = decoded.strip()
        
        while decoded[-len("<|eot_id|>"):] == "<|eot_id|>":
            decoded = decoded[:-len("<|eot_id|>")]

        pattern = r"\[\s*\d+\s*\]"

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
        "0": {"sentence": [f"{first_name} buys twice as many red ties as blue ties. The red ties cost 50% more than blue ties. He spent $200 on blue ties that cost $40 each. How much did he spend on ties?"], "ground_truth": 800},
        "1": {"sentence": [f"{first_name} sold clips to 48 of his friends in April, and then he sold half as many clips in May. How many clips did {first_name} sell altogether in April and May?"], "ground_truth": 72,},
        "2": {"sentence": [f"{first_name} writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"], "ground_truth": 624,},
        "3": {"sentence": [f"{first_name} is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"], "ground_truth": 48,},
        "4": {"sentence": [f"{first_name} created a care package to send to his brother, who was away at boarding school. {first_name} placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?"], "ground_truth": 16,},
        "5": {"sentence": [f"{first_name} can read 8 pages of a book in 20 minutes. How many hours will it take him to read 120 pages?"], "ground_truth": 5,},
        "6": {"sentence": [f"{first_name} is fundraising for his charity by selling brownies for $3 a slice and cheesecakes for $4 a slice. If {first_name} sells 43 brownies and 23 slices of cheesecake, how much money does {first_name} raise?"], "ground_truth": 221,},
        "7": {"sentence": [f"{first_name} is putting together 4 tables. Each table has 4 legs and each leg needs 2 screws. He has 40 screws. How many screws will he have left over?"], "ground_truth": 8,},
        "8": {"sentence": [f"{first_name} buys 10 packs of magic cards. Each pack has 20 cards and 1/4 of those cards are uncommon. How many uncommon cards did he get?"], "ground_truth": 50,},
        "9": {"sentence": [f"{first_name} creates a media empire. He creates a movie for $2000. Each DVD cost $6 to make. He sells it for 2.5 times that much. He sells 500 movies a day for 5 days a week. How much profit does he make in 20 weeks?"], "ground_truth": 448000,},
        "10": {"sentence": [f"{first_name} went to a shop to buy some groceries. He bought some bread for $2, butter for $3, and juice for two times the price of the bread. He had $15 for his shopping. How much money did {first_name} have left?"], "ground_truth": 6,},
        "11": {"sentence": [f"{first_name} has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does {first_name} have in total?"], "ground_truth": 15,},
        "12": {"sentence": [f"{first_name} is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?"], "ground_truth": 5,},
        "13": {"sentence": [f"{first_name} is three times older than Ben. Ben is two times older than Chris. If Chris is 4, how old is Caroline?"], "ground_truth": 24,},
        "14": {"sentence": [f"{first_name} eats 1 apple a day for two weeks. Over the next three weeks, he eats the same number of apples as the total of the first two weeks. Over the next two weeks, he eats 3 apples a day. Over these 7 weeks, how many apples does he average a week?"], "ground_truth": 10,},
        "15": {"sentence": [f"{first_name} bought 2 soft drinks for$ 4 each and 5 candy bars. He spent a total of 28 dollars. How much did each candy bar cost?"], "ground_truth": 4,},
        "16": {"sentence": [f"{first_name} has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?"], "ground_truth": 160,},
        "17": {"sentence": [f"{first_name} is throwing a huge Christmas party. He invites 30 people. Everyone attends the party, and half of the guests bring a plus one (one other person). He plans to serve a 3-course meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?"], "ground_truth": 135,},
        "18": {"sentence": [f"{first_name} volunteers at a shelter twice a month for 3 hours at a time. How many hours does he volunteer per year?"], "ground_truth": 72,},
        "19": {"sentence": [f"{first_name} puts $25 in his piggy bank every month for 2 years to save up for a vacation. He had to spend $400 from his piggy bank savings last week to repair his car. How many dollars are left in his piggy bank?"], "ground_truth": 200,},
        "20": {"sentence": [f"{first_name} has 16 toy cars, and the number of cars he has increases by 50% every year. How many toy cars will {first_name} have in three years?"], "ground_truth": 54,},
        "21": {"sentence": [f"{first_name} just turned 12 and started playing the piano. His friend Sheila told him about the 10,000-hour rule which says, after 10,000 hours of practice, you become an expert or master in your field. If {first_name} wants to become a piano expert before he is 20, how many hours a day will he need to practice if he practices every day, Monday – Friday, and takes two weeks off for vacation each year?"], "ground_truth": 5,},
        "22": {"sentence": [f"In 6 months Bella and {first_name} will be celebrating their 4th anniversary. How many months ago did they celebrate their 2nd anniversary?"], "ground_truth": 18,},
        "23": {"sentence": [f"{first_name} was preparing for a party to be held in four days. So, he made 24 gallons of root beer on the first day and put them in the refrigerator cooler. But later that evening, his children discovered the delicious nectar and robbed the cooler, drinking 4 of those gallons of root beer. On the second day, his wife Barbie also discovered the root beer and accidentally spilled 7 gallons. On the third day, {first_name}'s friend Ronnie visited {first_name}'s house and helped himself to the root beer, further reducing the amount remaining by 5 gallons. On the fourth day, 3 people showed up for the party. If {first_name} and the others shared the remaining root beer equally, how much was available for each to drink during the party?"], "ground_truth": 350,},
        "24": {"sentence": [f"{first_name} owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away?"], "ground_truth": 10,},
        "25": {"sentence": [f"{first_name} likes to collect model trains. He asks for one for every birthday of his, and asks for two each Christmas. {first_name} always gets the gifts he asks for, and asks for these same gifts every year for 5 years. At the end of the 5 years, his parents give him double the number of trains he already has. How many trains does {first_name} have now?"], "ground_truth": 45,},
        "26": {"sentence": [f"{first_name} has $5000. He spends $2800 on a new motorcycle, and then spends half of what's left on a concert ticket. {first_name} then loses a fourth of what he has left. How much money does he have left?"], "ground_truth": 825,},
        "27": {"sentence": [f"{first_name} has some coins. He has 2 more quarters than nickels and 4 more dimes than quarters. If he has 6 nickels, how much money does he have?"], "ground_truth": 350,},
        "28": {"sentence": [f"{first_name} is at a train station and is waiting for his train. He isn't sure how long he needs to wait, but he knows that the fourth train scheduled to arrive at the station is the one he needs to get on. The first train is scheduled to arrive in 10 minutes, and this train will stay in the station for 20 minutes. The second train is to arrive half an hour after the first train leaves the station, and this second train will stay in the station for a quarter of the amount of time that the first train stayed in the station. The third train is to arrive an hour after the second train leaves the station, and this third train is to leave the station immediately after it arrives. The fourth train will arrive 20 minutes after the third train leaves, and this is the train {first_name} will board. In total, how long, in minutes, will {first_name} wait for his train?"], "ground_truth": 27,},
        "29": {"sentence": [f"{first_name} decides to take up juggling to perform at the school talent show a month in the future. He starts off practicing juggling 3 balls, and slowly gets better adding 1 ball to his juggling act each week. After the end of the fourth week the talent show begins, but when {first_name} walks on stage he slips and drops three of his balls. 2 of them are caught by people in the crowd as they roll off the stage, but one gets lost completely since the auditorium is dark. With a sigh, {first_name} starts to juggle on stage with how many balls?"], "ground_truth": 4,},
    }
    return dataset_dict


def location_dataset(location):
    dataset_dict = {
        "0": {"sentence": [f"If the total cost of all the {location} supplies that Pauline wants to purchase is $150 before sales tax, and the sales tax is 8% of the total amount purchased, what will be the total amount that Pauline will spend on all the items, including sales tax?"], "ground_truth": 162},
        "1": {"sentence": [f"If adding seven more rabbits to the thirteen already in the cage would result in a total number of rabbits that is 1/3 of the number of rabbits Jasper saw in the {location} today, what is the total number of rabbits that Jasper saw in the {location} today?"], "ground_truth": 60},
        "2": {"sentence": [f"If there were initially 250 books in the {location} and 120 books were taken out on Tuesday to be read by children, followed by 35 books being returned on Wednesday and another 15 books being withdrawn on Thursday, what is the current total number of books in the {location}?"], "ground_truth": 150},
        "3": {"sentence": [f"Jason goes to the {location} 4 times more often than William goes. If William goes 2 times per week to the {location}, how many times does Jason go to the {location} in 4 weeks?"], "ground_truth": 32},
        "4": {"sentence": [f"If there were initially 235 books in the {location} and 227 books were borrowed on Tuesday, then 56 books were returned on Thursday, and 35 books were borrowed again on Friday, how many books are currently in the {location}?"], "ground_truth": 29},
        "5": {"sentence": [f"A {location} has a number of books. 35% of them are intended for children and 104 of them are for adults. How many books are in the {location}?"], "ground_truth": 160},
        "6": {"sentence": [f"If Benjamin spent a total of $15 at {location}, buying a salad, a burger, and two packs of fries, and one pack of fries cost $2, with the salad being three times that price, what is the cost of the burger?"], "ground_truth": 5},
    }
    return dataset_dict


def animals_dataset(animal):
    dataset_dict = {
        "0": {"sentence": [f"Out of the 60 {animal}s in a kennel, 9 {animal}s enjoy watermelon, 48 {animal}s enjoy salmon, and 5 {animal}s enjoy both watermelon and salmon. How many {animal}s in the kennel do not eat either watermelon or salmon?"], "ground_truth": 8},
        "1": {"sentence": [f"James needs to get more toys for his {animal} shelter. Each {animal} needs one toy. James currently has 4 toys on hand for 4 {animal}s, but there are 8 more {animal}s in the shelter now. After buying the toys, he went back to see that there are twice as many more {animal}s than when he left so he had to buy some more toys. When James came back yet again, 3 {animal}s were gone so he no longer needed those toys. How many toys in total does James need?"], "ground_truth": 33},
        "2": {"sentence": [f"If Elise purchased a 15kg bag of {animal} food and then another 10kg bag, resulting in a total of 40kg of {animal} food, how many kilograms of {animal} food did Elise already have before her recent purchases?"], "ground_truth": 15},
        "3": {"sentence": [f"John takes care of 10 {animal}s. Each {animal} takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of {animal}s?"], "ground_truth": 35},
        "4": {"sentence": [f"Cecilia just bought a new {animal}. According to her veterinarian, she has to feed the {animal} 1 cup of {animal} food every day for the first 180 days. Then she has to feed the {animal} 2 cups of {animal} food every day for the rest of its life. If one bag of {animal} food contains 110 cups, how many bags of {animal} food will Cecilia use in the first year?"], "ground_truth": 5},
        "5": {"sentence": [f"What is the total weight, in ounces, of the pet food that Mrs. Anderson bought if she purchased 2 bags of 3-pound cat food and 2 bags of {animal} food that weigh 2 pounds more than each bag of cat food?"], "ground_truth": 256},
        "6": {"sentence": [f"If Belle consumes 4 {animal} biscuits and 2 rawhide bones every evening, and each rawhide bone costs $1 and each {animal} biscuit costs $0.25, what is the total cost, in dollars, to feed Belle these treats for a week?"], "ground_truth": 21},
        "7": {"sentence": [f"If Ben's potato gun can launch a potato 6 football fields, and each football field is 200 yards long, and Ben's {animal} can run at a speed of 400 feet per minute, how many minutes will it take for his {animal} to fetch a potato that he launches?"], "ground_truth": 9},
        "8": {"sentence": [f"John adopts a {animal}. He takes the {animal} to the groomer, which costs $100. The groomer offers him a 30% discount for being a new customer. How much does the grooming cost?"], "ground_truth": 70},
        "9": {"sentence": [f"It takes 2.5 hours to groom a {animal} and 0.5 hours to groom a cat. What is the number of minutes it takes to groom 5 {animal}s and 3 cats?"], "ground_truth": 840},
        "10": {"sentence": [f"A cat has nine lives.  A {animal} has x less lives than a cat.  A mouse has 7 more lives than a {animal}. A mouse has 13 lives. What is the value of unknown variable x?"], "ground_truth": 3},
        "11": {"sentence": [f"If seven more {animal}s are added to the thirteen in the cage, the number of {animal}s in the cage will be 1/3 the number of {animal}s Jasper saw in the park today. How many {animal}s did Jasper see in the park today?"], "ground_truth": 60},
        "12": {"sentence": [f'Chris has a two-speed lawn mower. He can mow his entire lawn in "turtle" mode in 1 hour, or 40 minutes in "{animal}" mode. Today, he experimented by mowing half in turtle mode and half in {animal} mode. How many minutes did it take him to mow the lawn?'], "ground_truth": 50},
        "13": {"sentence": [f"At a James Bond movie party, each guest is either male (M) or female (F. 40% of the guests are women, 80% of the women are wearing {animal} ears, and 60% of the males are wearing {animal} ears. If the total number of guests at the party is 200, given all this information, what is the total number of people wearing {animal} ears?"], "ground_truth": 136},
        "14": {"sentence": [f"Every month, Madeline has to buy food, treats, and medicine for her {animal}. Food costs $25 per week. Treats cost $20 per month. Medicine costs $100 per month. How much money does Madeline spend on her {animal} per year if there are 4 weeks in a month?"], "ground_truth": 2640},
        "15": {"sentence": [f"Russell works at a pet store and is distributing straw among the rodents. The rats are kept in 3 cages in equal groups and each rat is given 6 pieces of straw. There are 10 cages of hamsters that are kept alone and each hamster is given 5 pieces of straw. There is also a pen of {animal} where 20 pieces of straw are distributed among the {animal}. No straw is used anywhere else in the store. If 160 pieces of straw have been distributed among the small rodents, how many rats are in each cage?"], "ground_truth": 5},
    }
    return dataset_dict


def group_dataset_map(conf):
    group_dataset_dict = {
        "names": jerrys_dataset,
        "animals": animals_dataset,
        "locations": location_dataset,
    }
    return group_dataset_dict[conf["intervenability"]["group"]]


def name_map(conf=None):
    gemma_name_dict = {
        "names": {
            "Mike": 11681,
            "Jason": 59158,
            "Jacob": 42983,
            "Jerry": 44109,
            "James": 37532,
            "Robert": 22710,
            "Jordan": 28504,
            "Jackson": 38384,
            "Paul": 26534,
            "David": 28951,
            "Andrew": 26342,
            "Gary": 17911,
        },
        "animals": {
            "dog": 36714,
            "rabbit": 33175,
            "bear": 9703,
            "goat": 31478,
            "horse": 37073,
        },
        "locations": {
            "room": 46124,
            "school": 17371,
            "park": 52199,
            "library": 20342,
            "fast-food restaurant": 36202,
        },
    }

    llama_name_dict = {
        "names": {
            "Bob": 7925,
            "John": 161,
            "James": 12102,
        },
    }

    if conf is not None and "llama" in conf["model"]["name"].lower():
        return llama_name_dict
    return gemma_name_dict


def alternative_names():
    alt_dict = {
        # gemma-2-2b
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
        "Gary": ["Gary", "Garre", "Garret", "Garrett", "Garry"],

        # llama-3.2-1b
        "Bob": ["Bob", "Bobby", "Tobias", "Toby"],
        "John": ["John", "Johnny"]
        # James similar to above
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


def object_in_generated(object, gen):
    if object in gen:
        return 1
    return 0


def evaluate_generated_texts(conf, texts, dataset_dict, drop_name, include_name):
    count = 0
    drop_names = 0
    include_names = 0
    interv_group = conf["intervenability"]["group"]
    replacement_group = conf["intervenability"]["replacement_group"]

    for jerry_doc in dataset_dict.keys():
        jerry_int = int(jerry_doc)
        text = texts[jerry_int]
        
        generated_parts = text.split("\n\nSolution:\n")
        if len(generated_parts) == 0:
            continue
        generated = generated_parts[1]
        ground_truth = dataset_dict[jerry_doc]["ground_truth"]

        pred, gt = extract_floats(conf=None, response = generated, solution = None, ground_truth=ground_truth)
        if interv_group == "names" and replacement_group == "names":
            drop_names += name_in_generated(drop_name, generated)
            include_names += name_in_generated(include_name, generated)
        elif interv_group == "names" and replacement_group != "names":
            drop_names += name_in_generated(drop_name, generated)
            include_names += object_in_generated(include_name, generated)
        elif interv_group != "names" and replacement_group == "names":
            drop_names += object_in_generated(drop_name, generated)
            include_names += name_in_generated(include_name, generated)
        elif interv_group != "names" and replacement_group != "names":
            drop_names += object_in_generated(drop_name, generated)
            include_names += object_in_generated(include_name, generated)

        if pred == gt:
            count += 1

    return count, drop_names, include_names