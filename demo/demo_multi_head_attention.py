import torch
from src.multi_head_attention_wrapper import MultiHeadAttentionWrapper

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your 
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts 
     [0.22, 0.58, 0.33], # with 
     [0.77, 0.25, 0.10], # one 
     [0.05, 0.80, 0.55]] # step
)

batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
context_length = batch.shape[1]

d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shap:", context_vecs.shape)


