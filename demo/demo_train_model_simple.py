import torch
import tiktoken
from torch.cuda import temperature
from src.GPT_dataset import create_dataloader
from src.gpt_model import GPTModel, generate, text_to_token_ids, token_ids_to_text
from src.train_model_simple import train_model_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
with open("./test/the-verdict.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters: ", total_characters)
print("Tokens: ", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=False,
    num_workers=0
)
val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
num_epochs = 10
train_losses, val_losses, token_seens = train_model_simple(
    model= model, train_loader=train_loader, val_loader=val_loader,
    optimizer= optimizer, device= device, num_epochs= num_epochs, 
    eval_freq=5, eval_iter=1,
    start_context="Every effor moves you", tokenizer=tokenizer
)

torch.save(model.state_dict(), "model.pth")
