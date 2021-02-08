from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import numpy as np
import torch

from random_forest_pytorch.random_forest_gemm import RandomForestGEMM

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

X_pt = torch.Tensor(X)
X_pt_cuda = X_pt.cuda()

def test_rf_numpy():
    rf = RandomForestGEMM(clf, "numpy")
    test = np.all(rf.predict(X) == clf.predict(X))

    assert test, "Predictions are not equal between RandomForestGEMM on NumPy and sklearn RandomForestClassifier"

def test_rf_torch_cpu():
    rf = RandomForestGEMM(clf, "torch")
    test = np.all(rf.predict(X_pt).numpy() == clf.predict(X))

    assert test, "Predictions are not equal between RandomForestGEMM on PyTorch (CPU) and sklearn RandomForestClassifier"

def test_rf_torch_gpu():
    rf = RandomForestGEMM(clf, "torch", "cuda")
    test = np.all(rf.predict(X_pt_cuda).cpu().numpy() == clf.predict(X))

    assert test, "Predictions are not equal between RandomForestGEMM on PyTorch (GPU) and sklearn RandomForestClassifier"
