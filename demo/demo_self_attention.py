import torch

from src.self_attention_v1 import SelfAttentionV1
from src.self_attention_v2 import SelfAttentionV2

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your 
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts 
     [0.22, 0.58, 0.33], # with 
     [0.77, 0.25, 0.10], # one 
     [0.05, 0.80, 0.55]] # step
)
query = inputs[1]

print(inputs.shape)
print(inputs.shape[0])

attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weight 2:", attn_scores_2_tmp)
print("Sum of row2", attn_scores_2_tmp.sum())


def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_scores_2_native = softmax_native(attn_scores_2)
print("Attention weight 2:", attn_scores_2_native)
print("Sum 2", attn_scores_2_native.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weight 2:", attn_weights_2)
print("Sum 2", attn_weights_2.sum())


attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i][j] = torch.dot(x_i, x_j)
print(attn_scores)

attn_scores_tmp = inputs @ inputs.T
print(attn_scores_tmp)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
print("All rows sum:", attn_weights.sum(dim=-1))


all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
print(W_query)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

keys = inputs @ W_key
values = inputs @ W_value

print("keys.shap:", keys.shape)
print("values.shape", values.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

print(keys)
print(keys.T)

attn_scores2 = query_2 @ keys.T
print(attn_scores2)

d_k = keys.shape[-1]
print(d_k)
attn_weights2 = torch.softmax(attn_scores2 / d_k ** 0.5, dim=-1)
print(attn_weights2)
context_vec2= attn_weights2 @ values
print(context_vec2)

torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)
print(sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
print(sa_v2(inputs))
