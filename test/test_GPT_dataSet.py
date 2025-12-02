# This is test for GPTDataSet module.

import torch
from src.GPT_dataset import create_dataloader

def test_create_dataloader():
    with open("./the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
