import torch
from torch.utils.data import DataLoader, Dataset

from src.gpt import GPT
from src.tokenizer import CharacterTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 8
BLOCK_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 3e-4

HIDDEN_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 4
INTERMEDIATE_SIZE = 512


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load corpus
# --------------------------------------------------

with open(
    "experiments/corpus.txt",
    "r",
    encoding="utf-8",
) as f:

    text = f.read()

print("Corpus characters:", len(text))


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = CharacterTokenizer(text)

encoded = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long,
)

print("Vocabulary size:", tokenizer.vocab_size)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class LanguageModelDataset(Dataset):

    def __init__(
        self,
        data,
        block_size,
    ):
        self.data = data
        self.block_size = block_size

    def __len__(self):

        return len(self.data) - self.block_size

    def __getitem__(self, index):

        x = self.data[
            index:index + self.block_size
        ]

        y = self.data[
            index + 1:index + self.block_size + 1
        ]

        return x, y


dataset = LanguageModelDataset(
    encoded,
    BLOCK_SIZE,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = GPT(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=BLOCK_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE,
    dropout=0.0,
)

model = model.to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# Training
# --------------------------------------------------

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=x,
            targets=y,
        )

        loss = outputs["loss"]

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(dataloader)
    )

    if (
        epoch == 0
        or (epoch + 1) % 50 == 0
    ):

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {average_loss:.4f}"
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.stoi,
        "inverse_vocab": tokenizer.itos,
        "config": {
            "block_size": BLOCK_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "intermediate_size": INTERMEDIATE_SIZE,
        },
    },
    "results/gpt_model.pt",
)

print()
print("Training complete.")
print("Saved: results/gpt_model.pt")