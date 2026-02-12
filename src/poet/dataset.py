from datasets import load_dataset, DatasetDict

def retrieve_dataset(conf, tokenizer, split: str = "train"):
    """Retrieve a dataset from Hugging Face."""

    is_mathematical = conf["dataset"]["is_mathematical"]
    for_generation = conf["dataset"]["for_generation"]
    dataset_length = conf["dataset"]["eval_length"]

    write_gsm8k_heap = conf["interpretability"]["write_gsm8k_heap"]

    do_intervene = conf["intervenability"]["do_intervene"]
    
    dataset_name = conf["dataset"]["name"]
    if for_generation and conf["model"]["run_eval"]:
        dataset_name = conf["eval"]["dataset"]
    if write_gsm8k_heap:
        dataset_name = conf["eval"]["dataset"]
        print(dataset_name)
    print(dataset_name)

    dataset_configuration_map = {
         "wikitext-2-raw": ("wikitext", "wikitext-2-raw-v1"),
         "wikitext-2": ("wikitext", "wikitext-2-v1"),
         "wikitext-103-raw": ("wikitext", "wikitext-103-raw-v1"),
         "wikitext-103": ("wikitext", "wikitext-103-v1"),
         "meta-math": ('meta-math/MetaMathQA', None),
         "gsm8k": ('gsm8k', 'main'),
    }
    
    assert split in ['test', 'train', 'validation']

    hub_name, dataset_configuration = dataset_configuration_map[dataset_name]
    
    if "wikitext" in dataset_name: 
        assert hub_name == "wikitext"  
        if split == "train": dataset = load_dataset(hub_name, dataset_configuration)
        else: dataset = load_dataset(hub_name, dataset_configuration, split=split)
        # else: dataset = load_dataset(dataset_name, dataset_configuration_map[dataset_name], split=f'{split}[:100]')

        dataset = dataset.filter(
            lambda x: x["text"] is not None and x["text"].strip() != ""
        )

        dataset = dataset.filter(
            lambda x: not x["text"].strip().startswith("=")
        )

    elif dataset_name == "meta-math":
        dataset = load_dataset(hub_name)

        # dataset["train"] = dataset["train"].select(range(100))

        split1 = dataset["train"].train_test_split(
            test_size=0.05,
            seed=42,
        )

        split2 = split1["test"].train_test_split(
            test_size=0.8,
            seed=42,
        )

        if split == "test":
            math_dataset = DatasetDict({
                "test": split2["test"],
            })
        
        else:
            math_dataset = DatasetDict({
                "train": split1["train"],
                "validation": split2["train"],
                "test": split2["test"],
            })

        for key in math_dataset.keys():
            math_dataset[key] = math_dataset[key].map(
                lambda ex: {
                    "problem": ex["query"],
                    "solution": ex["response"],
                },
                remove_columns=["query", "response", "type", "original_question"],
            )

    elif dataset_name == "gsm8k":
        dataset = load_dataset(hub_name, dataset_configuration)

        gsm8k_split = dataset["train"].train_test_split(
            test_size=0.10,
            seed=42,
        )

        if split == "test":
            math_dataset = DatasetDict({
                "test": dataset["test"],
            })

        else:
            math_dataset = DatasetDict({
                "train": gsm8k_split["train"],
                "validation": gsm8k_split["test"],
                "test": dataset["test"],
            })
        
        for key in math_dataset.keys():
            math_dataset[key] = math_dataset[key].map(
                lambda ex: {
                    "problem": ex["question"],
                    "solution": ex["answer"],
                },
                remove_columns=["question", "answer"],
            )

    else: raise NotImplementedError

        
    if is_mathematical:
        if for_generation and not do_intervene:
            assert dataset_length != None
            math_dataset["test"] = math_dataset["test"].shuffle(seed=42).select(range(dataset_length))
        else:
            assert dataset_length == None

        for key in math_dataset.keys():
            math_dataset[key] = math_dataset[key].filter(
                lambda x: x["problem"] is not None and x["problem"].strip() != ""
            )

            math_dataset[key] = math_dataset[key].filter(
                lambda x: x["solution"] is not None and x["solution"].strip() != ""
            )


    def tokenize_ntp(examples):        
        texts = [t.strip() for t in examples["text"]]
        tokens = tokenizer(texts)
        return tokens
    

    def group_texts(examples, block_size = 512):       
        input_ids = sum(examples["input_ids"], [])
        attention_mask = sum(examples["attention_mask"], [])

        total_length = len(input_ids)
        total_length = (total_length // block_size) * block_size

        input_ids = input_ids[:total_length]
        attention_mask = attention_mask[:total_length]

        result = {
            "input_ids": [
                input_ids[i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
            "attention_mask": [
                attention_mask[i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
        }

        assert all(isinstance(x, list) for x in examples["input_ids"])

        result["labels"] = result["input_ids"].copy()
        return result
        

    def tokenize_math(examples, max_length=512):
        prompts = []
        full_texts = []

        for p, s in zip(examples["problem"], examples["solution"]):
            if p is None or s is None:
                p = ""
                s = ""
            prompt = f"Problem:\n{p}\n\nSolution:\n"
            full_texts.append(prompt + s)
            prompts.append(prompt)

        enc = tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            padding_side="right",
            max_length=max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = []
        for ids, mask, prompt in zip(input_ids, attention_mask, prompts):
            lbl = ids.copy()
            prompt_len = len(
                tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )["input_ids"]
            )

            lbl[:prompt_len] = [-100] * prompt_len
            lbl = [
                -100 if m == 0 else l
                for l, m in zip(lbl, mask)
            ]
            labels.append(lbl)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    def tokenize_math_generation(examples, max_length=256):
        prompts = []

        for p, s in zip(examples["problem"], examples["solution"]):
            if p is None or s is None:
                p = ""
                s = ""
            prompt = f"Problem:\n{p}\n\nSolution:\n"
            prompts.append(prompt)

        enc = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            padding_side="left",
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    if is_mathematical:
        if split == "test":
            lm_dataset = DatasetDict({
                "test": [],
            })
            
        else:
            lm_dataset = DatasetDict({
                "train": [],
                "validation": [],
                "test": [],
            })

        if for_generation:
            for key in math_dataset.keys():
                lm_dataset[key] = math_dataset[key].map(
                    tokenize_math_generation,
                    batched=True,
                    # remove_columns=["problem", "solution"]
                )

        else:
            for key in math_dataset.keys():
                lm_dataset[key] = math_dataset[key].map(
                    tokenize_math,
                    batched=True,
                    remove_columns=["problem", "solution"]
                )

    elif hub_name == "wikitext":
        if for_generation: raise NotImplementedError
        tokenized = dataset.map(
            tokenize_ntp,
            batched=True,
            remove_columns=["text"]
        )

        lm_dataset = tokenized.map(
            group_texts,
            batched=True
        )

    else: raise NotImplementedError

    return lm_dataset

