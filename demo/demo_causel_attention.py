import torch
from src.causal_attention import CausalAttention
from src.self_attention_v2 import SelfAttentionV2

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your 
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts 
     [0.22, 0.58, 0.33], # with 
     [0.77, 0.25, 0.10], # one 
     [0.05, 0.80, 0.55]] # step
)

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_socres = queries @ keys.T
attn_weights = torch.softmax(attn_socres / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)

context_length = attn_socres.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

masked_simple = attn_weights * mask_simple
print(masked_simple)

row_sum = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sum
print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_socres.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights2 = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights2)


torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

torch.manual_seed(123)
print(dropout(attn_weights2))

batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shap:", context_vecs.shape)
