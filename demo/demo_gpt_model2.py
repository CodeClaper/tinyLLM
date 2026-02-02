import torch
import tiktoken

from src.gpt_model import GPTModel, text_to_token_ids

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
input = [] 
target = []
input.append(text_to_token_ids("every effort moves", tokenizer))
input.append(text_to_token_ids("I really like", tokenizer))
input = torch.cat(input, dim=0)
print(input)

target.append(text_to_token_ids(" effort moves you", tokenizer))
target.append(text_to_token_ids(" really like chocolate", tokenizer))
target = torch.cat(target, dim=0)
print(target)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
with torch.no_grad():
    logits = model(input)
probas = torch.softmax(logits, dim=-1)
print(probas)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs: \n", token_ids)

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], target[text_idx]]
print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], target[text_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
neg_avg_log_propbas = avg_log_probas * -1
print(neg_avg_log_propbas)

logits_flat = logits.flatten(0, 1)
targets_flat = target.flatten()
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("losss: ", loss)
perplexity = torch.exp(loss)
print("perplexity: ", perplexity)
