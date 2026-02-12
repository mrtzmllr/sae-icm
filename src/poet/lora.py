from peft import LoraConfig, get_peft_model


def apply_lora(model, conf):
    """Apply LoRA to the given model."""
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        # "gate_proj", "up_proj", "down_proj",
    ]

    include_sae = conf["training"]["include_sae"]
    
    train_W_enc = conf['sae']['train_W_enc']
    train_b_enc = conf['sae']['train_b_enc']
    train_b_dec = conf['sae']['train_b_dec']

    if train_W_enc: target_modules += ["W_enc.weight"]
    if train_b_enc: target_modules += ["W_enc.bias"]
    if train_b_dec: target_modules += ["W_dec.bias"]

    any_trainable = conf["sae"]["any_trainable"]
    assert any_trainable == train_W_enc or train_b_enc or train_b_dec

    rank = conf["lora"]["rank"]
    alpha = conf["lora"]["alpha"]
    dropout = conf["lora"]["dropout"]

    lora_config = LoraConfig(
        r=rank,  
        lora_alpha=alpha,
        lora_dropout=dropout,
        # bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="all" if any_trainable else "none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if include_sae and any_trainable:
        model.model.model.sae.W_enc.weight.requires_grad = train_W_enc
        model.model.model.sae.W_enc.bias.requires_grad = train_b_enc
        model.model.model.sae.W_dec.bias.requires_grad = train_b_dec

    return model