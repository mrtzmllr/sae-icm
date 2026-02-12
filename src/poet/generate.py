import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from src.poet.directories import write_output_dir, finetuned_sae_states_dir
from src.poet.config import load_config
from src.poet.insert_sae import Gemma2SAEForCausalLM, insert_sae
from src.poet.argparse import parse_args
from src.poet.dataset import retrieve_dataset
from src.poet.model_config import Gemma2SAEConfig
from src.poet.model_config import retrieve_config

all_args = parse_args()

conf = load_config(all_args, run_eval=True)
model_name = conf["model"]["name"]
dataset_name = conf["eval"]["dataset"]

model_path = conf["eval"]["model_path"]

model_config = retrieve_config(conf)

model = Gemma2SAEForCausalLM.from_pretrained(model_path, config=model_config)
# model = Gemma2SAEForCausalLM.from_pretrained(model_name) #, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = insert_sae(model=model, conf=conf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model_path)

# for name, p in model.named_parameters():
#     if "enc" in name and "sae" in name: 
#         print(name, p.requires_grad)
#         print(p.shape)
#     if "dec" in name and "sae" in name: 
#         print(name, p.requires_grad)
#         print(p.shape)

# print("transpose", (model.model.sae.W_dec @ model.model.sae.W_enc).abs().mean().item())
# print("transpose", ((model.model.sae.W_enc @ model.model.sae.W_dec).trace() / model.model.saHoweve.d_model).item())
# print(torch.round(model.model.sae.W_enc @ model.model.sae.W_dec, decimals = 5))
# diff = model.model.sae.W_enc @ model.model.sae.W_dec - torch.eye(model.model.sae.d_model, device=model.model.sae.W_enc.device)
# print(diff.shape)
# print(torch.round(diff, decimals = 5)[:10,:10])
# print(diff.abs().sum().item())
# print("dec", model.model.sae.b_dec.mean().item())
# print("enc", model.model.sae.b_enc.abs().sum().item())


# names = ["Joseph Ratzinger", "Friedrich Schiller", "Dirk Nowitzki",
#            "Marlene Dietrich", "Max Planck", "Albert Einstein", "Friedrich Hölderlin", "Philipp Lahm", "Manuel Neuer", "Hans Zimmer",
#         #    "Ludwig van Beethoven", 
#            "Ferdinand von Zeppelin",
#            "Franz Beckenbauer",
#            "Angela Merkel", "Johannes Brahms"]


# prompts = [name + " was born in the country of" for name in names]

p = "The town of Belize has 400 homes. One fourth of the town's homes are white. One fifth of the non-white homes have a fireplace. How many of the non-white homes do not have a fireplace?"
s = "One fourth of the town's homes are white, so there are 400/4 = 100 white homes. The remaining non-white homes are 400 - 100 = 300 homes. One fifth of the non-white homes have a fireplace, so there are 300/5 = 60 non-white homes with a fireplace. Therefore, the number of non-white homes without a fireplace is 300 - 60 = 240. #### 240 The answer is: 240"
prompts = [f"Problem:\n{p}\n\nSolution:\n"]
full = [f"Problem:\n{p}\n\nSolution:\n" + s]

# prompts = ["We know that every 30 minutes, a machine produces 30 cans of soda. Since there are 60 minutes in an hour, and 8 hours in total, the total number of minutes is 60 * 8 = 480 minutes. If a machine produces"]
# ["Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?"]
# ["Convert $10101_3$ to a base 10 integer"]
# ["Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"]
# ["A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have total over the three months?"]

for prompt in prompts:
# "The arsenal was constructed at the request of Governor"
    pp = f"Problem:\n{p}\n\nSolution:\n"
    full_text = pp + s


    enc = tokenizer(
        pp,
        truncation=True,
        padding="max_length",
        padding_side = "right",
        max_length=512,
        return_tensors="pt",
    )

    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=200,
            do_sample=False,
        )

    
    text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )
    
    print(text)
    
    # print(model.model.reconstruction / 100, "average reconstruction error over eval set")