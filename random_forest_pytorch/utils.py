import numpy as np
import pandas as pd

import torch
import re
import os

def compute_path(tree, k):
    """Compute path from nodes (leaf or internal) to root.
    Args:
        tree
        k (int) : index from start
    Return:
        List of [id, 1] if LeftSubTree, [id, -1] if RightSubTree
    """
    l = [[k,0]]
    left = tree.children_left
    right = tree.children_right

    while l[-1][0] != 0:
        parent_left = np.flatnonzero((left == l[-1][0]),)
        parent_right = np.flatnonzero((right == l[-1][0]),)

        if len(parent_left) > 0:
            l.append([parent_left[0], 1])
        else:
            l.append([parent_right[0], -1])

    return l[1:]


def read_text_file(path, encoding="utf-8", lines=False):
    with open(path, "r", encoding=encoding) as file:
        text = file.read()
        if lines:
            text = text.split("\n")[:-1]
    return text

def analyze_stack(stack):
    """Analyze a stack string obtained from the pytorch profiler to extract code lines and scopes
    Args:
        stack (list of string) format : `/path/to/file(<number>): function`
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
    with torch.autograd.profiler.profile(use_cuda=True, with_stack=True) as prof:
        func()

    pdf = pd.DataFrame([
        {attr: evt.__getattribute__(attr) for attr in ["name", "self_cpu_time_total", "self_cuda_time_total", "stack"]} for evt in prof.function_events
    ]).rename(columns=lambda x: x.split("_")[1] if "self_" in x else x)

    return pdf
