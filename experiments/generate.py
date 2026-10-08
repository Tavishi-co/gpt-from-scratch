import torch

from src.gpt import GPT
from src.tokenizer import CharacterTokenizer


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Load corpus
# --------------------------------------------------

with open(
    "experiments/corpus.txt",
    "r",
    encoding="utf-8",
) as f:

    text = f.read()


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = CharacterTokenizer(text)


# --------------------------------------------------
# Model
# --------------------------------------------------

checkpoint = torch.load(
    "results/gpt_model.pt",
    map_location=DEVICE,
)

config = checkpoint["config"]

model = GPT(
    vocab_size=len(checkpoint["vocab"]),
    max_position_embeddings=config["block_size"],
    hidden_size=config["hidden_size"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    intermediate_size=config["intermediate_size"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = "The transformer"

input_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=DEVICE,
)


# --------------------------------------------------
# Generate
# --------------------------------------------------

generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
)


# --------------------------------------------------
# Decode
# --------------------------------------------------

generated_text = tokenizer.decode(
    generated[0].tolist()
)

print("\nGenerated text:\n")
print(generated_text)