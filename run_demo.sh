#!/usr/bin/bash
python3 -m demo.demo_causel_attention
echo "=========================================="
python3 -m demo.demo_embedding
echo "=========================================="
python3 -m demo.demo_self_attention
echo "=========================================="
python3 -m demo.demo_multi_head_attention_wrapper
echo "=========================================="
python3 -m demo.demo_multi_head_attention
echo "=========================================="
python3 -m demo.demo_dummy_gpt_model
echo "=========================================="
python3 -m demo.demo_transformer_block
