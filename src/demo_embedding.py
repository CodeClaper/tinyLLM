## This module is simple embedding example.

import torch
from GPT_dataset import create_dataloader

input_ids = torch.tensor([2, 3, 5, 1])

vacob_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vacob_size, output_dim)

with open ("./test/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
input, target = next(data_iter)

print("Token IDS:\n", input)
print("\nToken Size:\n", input.shape)

token_embeddings = token_embedding_layer(input)
print(token_embeddings.shape)

pos_embedding_layer = torch.nn.Embedding(4, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(4))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
print(input_embeddings)


