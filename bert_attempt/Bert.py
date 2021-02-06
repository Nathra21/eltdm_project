#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
from main import load_imdb_model


tokenizer, model = load_imdb_model()
df = pd.read_csv("imdb/imdb.csv")
sample = df.sample(n=64, random_state=2021)
embedded = tokenizer.batch_encode_plus(sample["review"].tolist(), padding=True, truncation=True, max_length=512)
input = torch.tensor(embedded["input_ids"]).cuda()
output = model(input)