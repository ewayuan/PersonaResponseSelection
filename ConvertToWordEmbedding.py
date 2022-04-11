import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    with open('readme.txt') as f:
        lines = f.readlines()