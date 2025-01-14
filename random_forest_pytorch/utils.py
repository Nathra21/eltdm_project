import re
import os
import timeit
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.tree import plot_tree
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from hummingbird.ml import convert

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

def plot_decision_tree(tree):
    plt.subplots(figsize=(10, 10))
    plot_tree(tree, fontsize=12, filled=True, impurity=False, node_ids=True)

def make_complex_dataset(n_samples):
    X_1, y_1 = make_circles(n_samples=n_samples//3, factor=0.6)
    X_2, y_2 = make_circles(n_samples=n_samples//3, factor=0.8)
    X_3, y_3 = make_circles(n_samples=n_samples//3, factor=0.9)
    X_2 *= 1.5
    X_3 *= 2
    X = np.concatenate([X_1, X_2, X_3])
    y = np.concatenate([y_1, y_2, y_3])
    return X, y


def format_time(timespan, precision=3):
    """Formats the timespan in a human readable form.
    Copied from IPython source.
    """

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60*60*24),("h", 60*60),("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    
    # Unfortunately the unicode 'micro' symbol can cause problems in
    # certain terminals.  
    # See bug: https://bugs.launchpad.net/ipython/+bug/348466
    # Try to prevent crashes by being more secure than it needs to
    # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
    units = [u"s", u"ms",u'us',"ns"] # the save value   
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb5'.encode(sys.stdout.encoding)
            units = [u"s", u"ms",u'\xb5s',"ns"]
        except:
            pass
    scaling = [1, 1e3, 1e6, 1e9]
        
    if timespan > 0.0:
        order = min(-int(np.floor(np.log10(timespan)) // 3), 3)
    else:
        order = 3
    # return u"%.*g %s" % (precision, timespan * scaling[order], units[order])
    return f"{timespan*scaling[order]:>{precision+1}.{precision}g} {units[order]}"

def get_all_backends(constructor, clf, include_sklearn=True, include_hummingbird=False, **kwargs):
    out = {}
    
    if include_sklearn:
        out["sklearn"] = clf

    out.update({
        "numpy": constructor(clf, backend="numpy", **kwargs),
        "torch_cpu": constructor(clf, backend="torch", device="cpu", **kwargs),
        "torch_cuda": constructor(clf, backend="torch", device="cuda", **kwargs),
    })

    if include_hummingbird:
        out["hummingbird_cpu"] = convert(clf, "pytorch")
        out["hummingbird_cuda"] = convert(clf, "pytorch").to("cuda")
    
    return out

def time_all_backends(models, X, number=1000, repeat=7, return_std=True):
    # Pour bien comparer les trois backends, il faut avoir préparé à l'avance leurs inputs.
    Xs = {
        "sklearn": X,
        "numpy": X,
        "torch_cpu": torch.tensor(X).float(),
        "torch_cuda": torch.tensor(X).float().cuda(),
    }
    Xs["hummingbird_cpu"] = Xs["torch_cpu"]
    Xs["hummingbird_cuda"] = Xs["torch_cuda"]

    backends = models.keys()
    assert set(backends) <= set(Xs), "Some model keys are not in the standard backends."

    out = {}
    for backend in backends:
        func = lambda: models[backend].predict(Xs[backend])
        times = np.array(timeit.repeat(func, number=number, repeat=repeat)) / number
        if return_std:
            out[backend] = (times.mean(), times.std())
        else:
            out[backend] = times.mean()
    
    return out

def pretty_print_time_all_backends(models, X, number=1000, repeat=7):
    times = time_all_backends(models, X, number, repeat)
    for backend, (mu, std) in times.items():
        print(backend.ljust(17), "->", format_time(mu), "±", format_time(std))