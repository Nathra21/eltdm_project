import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch

def read_text_file(path, encoding="utf-8", lines=False):
    with open(path, "r", encoding=encoding) as file:
        text = file.read()
        if lines:
            text = text.split("\n")[:-1]
    return text

def load_imdb_model():
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/bert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/bert-imdb").eval().cuda().requires_grad_(False)
    return tokenizer, model

def analyze_stack(stack):
    """Analyze a stack string obtained from the pytorch profiler to extract code lines and scopes
    """
    out = []
    for call in stack:
        path, line, func = re.findall("(.*?)\((.*?)\): (.*)", call)[0]
        line = int(line) - 1
        if not os.path.exists(path):
            continue
        
        text = read_text_file(path, lines=True)
        content = text[line].strip()
        call_line = line
        while not (text[line].startswith("class") or text[line].startswith("def") or line==0):
            line -= 1
        top = text[line]
        out.append({
            "path": path,
            "line": call_line + 1,
            "top": top,
            "func": func,
            "content": content,
        })
    return pd.DataFrame(out)

def profile_command(func):
    """Profile a function without arguments and return a parsed dataframe.
    """
    with torch.profile.profiler.profile(use_cuda=True, with_stack=True) as prof:
        func()
    
    pdf = pd.DataFrame([
        {attr: evt.__getattribute__(attr) for attr in ["name", "self_cpu_time_total", "self_cuda_time_total", "stack"]} for evt in prof.function_events
    ]).rename(columns=lambda x: x.split("_")[1] if "self_" in x else x)

    return pdf