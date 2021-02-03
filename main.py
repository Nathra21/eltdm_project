from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.autograd.profiler as profiler

def load_imdb_model():
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/bert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/bert-imdb").eval().cuda().requires_grad_(False)
    return tokenizer, model

