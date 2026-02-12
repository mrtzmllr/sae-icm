from src.poet.pretrained import retrieve_pretrained
from src.poet.insert_sae import SAEonGemma2
from src.poet.dataset import retrieve_dataset


def preprocessing(conf):
    print("set model and dataset names")

    language_model, tokenizer = retrieve_pretrained(conf)
    language_model.config.use_cache = False
    
    print("loaded language_model")

    tokenized_datasets = retrieve_dataset(conf, tokenizer)
    print("loaded dataset")

    model = _retrieve_sae_model(conf)

    model.language_model = language_model

    for p in model.language_model.parameters():
        p.requires_grad_(False)

    return model, tokenized_datasets


def _retrieve_sae_model(conf):
    model = SAEonGemma2(conf=conf)
    return model