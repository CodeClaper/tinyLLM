# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Run all demos**: `bash run_demo.sh`
- **Run tests**: `bash run_test.sh` (or `cd test && python3 -m pytest -vv`)
- **Run a single test file**: `python3 -m pytest test/test_tokenizer.py -vv`
- **Run a single test function**: `python3 -m pytest test/test_vocab.py::test_token_text -vv`
- **Run a specific demo**: `python3 -m demo.demo_gpt_model`
- **Train a model**: `python3 -m demo.demo_train_model_simple` (saves `model.pth`)
- **Load & generate from saved model**: `python3 -m demo.demo_load_model`

## Architecture

Educational from-scratch GPT-style language model built in PyTorch. Not a pip-installable package — code is run directly from the repo root.

### Directory structure

- `src/` — library modules, each implementing one transformer component
- `demo/` — self-contained runnable scripts that import from `src/`
- `test/` — pytest tests using `test/the-verdict.txt` as test data

### Module dependency chain

```
tokenizer.py → vocab.py → simple_tokenizer.py  (tokenization pipeline)
my_tiktoken.py                                   (tiktoken GPT-2 BPE wrapper, standalone)

gelu.py → feed_forward.py                        (FFN for transformer blocks)
self_attention_v1.py / v2.py → causal_attention.py → multi_head_attention_wrapper.py
                                                     multi_head_attention.py (efficient)
layer_norm.py (standalone)

multi_head_attention.py + feed_forward.py + layer_norm.py → transformer_block.py
                                                              ↓
                                              gpt_model.py (GPTModel + generation + loss)
                                                  ↓
                                         train_model_simple.py (training loop)
GPT_dataset.py (Dataset/DataLoader, standalone)
```

### Model configuration (GPT_CONFIG_124M)

Based on GPT-2 124M parameters:
- `vocab_size`: 50257, `context_length`: 1024 (256 in training demos), `emb_dim`: 768
- `n_heads`: 12, `n_layers`: 12, `drop_rate`: 0.1, `qkv_bias`: False

The training demos use a reduced `context_length: 256` and train on the short story text `test/the-verdict.txt`.

### Key details

- Uses **Pre-LayerNorm** transformer blocks (layer norm inside the residual path)
- Architecture is a **decoder-only transformer** (GPT): causal (masked) self-attention, no encoder-decoder cross-attention
- Requires **Python 3.10** and **PyTorch 2.0.0** (see `requirements.txt`)
- GPU support is automatic: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- All demos import via `from src.<module> import <Class>` (no package exports, `src/__init__.py` is empty)
- Notable filename quirks: `dummy_gtp_model.py` (not "gpt"), `demo_causel_attention.py` (not "causal")
