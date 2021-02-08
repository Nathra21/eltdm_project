from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
import numpy as np
import torch

from random_forest_gemm import DecisionTreeGemm, RandomForestGEMM

from utils import profile_command, analyze_stack

# Create RF
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)


rf_pt_cuda = RandomForestGEMM(clf, "torch", "cuda")
X_pt = torch.Tensor(X)
X_pt_cuda = X_pt.cuda()

def wrapper():
    return rf_pt_cuda.predict(X_pt_cuda)

pdf = profile_command(wrapper)
pdf.to_pickle("pdf.pickle")
