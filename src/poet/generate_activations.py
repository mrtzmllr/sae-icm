from src.poet.pretrained import retrieve_pretrained
from src.poet.insert_sae import insert_sae
from src.poet.dataset import retrieve_dataset


def preprocessing(conf):
    model_name = conf["model"]["name"]
    dataset_name = conf["training"]["dataset"]
    print("set model and dataset names")

    model, tokenizer = retrieve_pretrained(model_name)

    model = insert_sae(model, conf)

    model.config.use_cache = False
    if conf["training"]["gradient_checkpointing"]: model.gradient_checkpointing_enable()

    print("loaded model")

    tokenized_datasets = retrieve_dataset(dataset_name, tokenizer)
    print("loaded dataset")

    return model, tokenized_datasets