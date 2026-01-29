from ast import mod
import torch
import tiktoken

from src.gpt_model import GPTModel, generate_text_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.n_vocab)
batch = []
txt = "Every effort moves you"

batch.append(torch.tensor(tokenizer.encode(txt)))
batch.append(torch.tensor(tokenizer.encode(txt)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

print("Input batch:\n", batch)
print("\nOut shap:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameter: {total_params:,}")

total_ouput_params = sum(p.numel() for p in model.out_head.parameters())
print(f"Total number of output parameter: {total_ouput_params:,}")

total_params_gpt2 = total_params - total_ouput_params
print(f"Total number of parameter considering weight tying: {total_params:,}")

total_size_bytes = total_params * 4 ## Assuming float32, 4bytes per parameter
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape", encoded_tensor.shape)
model.eval() ## Disable dropout since we are not training the model.
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
