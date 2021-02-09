import numpy as np
import torch
from random_forest_pytorch.utils import compute_path

class DecisionTreeGEMM:
    """Implementation proposed from `Taming Model Serving Complexity, Performance
    and Cost: A Compilation to Tensor Computations Approach`
    Args:
        decision_tree (sklearn DecisionTree)
    """
    
    def __init__(self, decision_tree, backend="numpy", device="cpu"):
        """Create matrix A,B,C,D,E from decision_tree"""
        assert backend in ["numpy", "torch"]
        self.backend = backend
        self.back = np if backend == "numpy" else torch

        assert device in ["cpu", "cuda"]
        self.device = device

        tree = decision_tree.tree_

        is_internal_nodes = tree.children_left != tree.children_right
        internal_nodes = np.flatnonzero(is_internal_nodes)
        leaf_nodes = np.flatnonzero(~is_internal_nodes)
        split_features_internal_nodes = tree.feature[internal_nodes]
        
        # Matrix A
        self.A = np.zeros(shape=(tree.n_features, len(internal_nodes)))
        for i,j in enumerate(split_features_internal_nodes):
            self.A[j,i] = 1
        
        # Matrix B
        self.B = tree.threshold[internal_nodes]
        
        # Matrix C
        sub_to_global_internal_nodes = { v:k for k,v in enumerate(internal_nodes) }
        s_to_g = lambda x : sub_to_global_internal_nodes[x]
        self.C = np.zeros(shape=(len(internal_nodes), len(leaf_nodes)))
        
        for j, leaf_idx in enumerate(leaf_nodes):
            path = compute_path(tree, leaf_idx)

            for i in range(len(path)):
                # Apply transformation on index
                # subset (internal nodes)
                #      -> global (internal and leaf nodes)
                path[i][0] = s_to_g(path[i][0])

            for node_idx, value in path:
                self.C[node_idx,j] = value
        
        # Matrix D
        self.D = np.sum(self.C == 1, axis=0)
        
        # Matrix E
        self.E = np.zeros(shape=(len(leaf_nodes), decision_tree.n_classes_))
        self.probas = tree.value[leaf_nodes].reshape(-1,decision_tree.n_classes_)/np.sum(tree.value[leaf_nodes], axis=-1)
        leaf_to_class = self.probas.argmax(axis=-1).flatten()

        for i in range(len(leaf_nodes)):
            self.E[i,leaf_to_class[i]] = 1
        
        if backend == "torch":
            self.A, self.B, self.C, self.D, self.E, self.probas = map(
                lambda x: torch.tensor(x, device=self.device, dtype=torch.float32),
                (self.A, self.B, self.C, self.D, self.E, self.probas)
            )
    
    def convert_to_float(self, tensor):
        if self.backend == "numpy":
            return tensor.astype(np.float32)
        else:
            return tensor.float()

    def GEMM(self, X):
        """Implement GEMM Strategy"""
        T = X @ self.A
        T = self.convert_to_float(T < self.B)
        T = T @ self.C
        T = self.convert_to_float(T == self.D)
        T = T @ self.E
        return T

    def predict(self, X):
        """Return class (integer) for each data point"""
        T = self.GEMM(X)
        return self.back.argmax(T, axis=1)
    
    def predict_onehot(self, X):
        """One Hot Encoding version of self.predict"""
        return self.GEMM(X)

class NaiveRandomForestGEMM:
    def __init__(self, random_forest, backend="numpy", device="cpu"):
        """Create estimators from random_forest"""
        assert backend in ["numpy", "torch"]
        self.backend = backend
        self.back = np if backend == "numpy" else torch
        self.device = device

        self.trees = [DecisionTreeGEMM(estimator, backend, device) for estimator in random_forest.estimators_]
        self.n_classes_ = random_forest.n_classes_
    
    def vote(self, X):
        """Count the vote from each tree for each data point"""
        return self.back.stack([e.predict_onehot(X) for e in self.trees]).sum(axis=0)

    def predict(self, X):
        predictions = self.vote(X)
        return self.back.argmax(predictions, axis=1)

class RandomForestGEMM:
    def __init__(self, random_forest, backend="numpy", device="cpu"):
        """Create estimators from random_forest"""
        assert backend in ["numpy", "torch"]
        self.backend = backend
        self.back = np if backend == "numpy" else torch
        self.device = device

        self.trees = [DecisionTreeGEMM(estimator, backend, device) for estimator in random_forest.estimators_]
        self.n_classes_ = random_forest.n_classes_
        self.n_features_ = random_forest.n_features_

        self.max_internal_nodes = 0
        self.max_leaves_nodes = 0

        for tree in self.trees:
            self.max_internal_nodes = max(self.max_internal_nodes, tree.A.shape[1])
            self.max_leaves_nodes = max(self.max_leaves_nodes, tree.C.shape[1])

        self.n_trees = len(self.trees)
        A_stacked = self.back.zeros((self.n_trees, self.n_features_, self.max_internal_nodes))
        B_stacked = self.back.zeros((self.n_trees, self.max_internal_nodes))
        C_stacked = self.back.zeros((self.n_trees, self.max_internal_nodes, self.max_leaves_nodes))
        D_stacked = self.back.zeros((self.n_trees, self.max_leaves_nodes))
        E_stacked = self.back.zeros((self.n_trees, self.max_leaves_nodes, self.n_classes_))
        probas_stacked = self.back.zeros((self.n_trees, self.max_leaves_nodes, self.n_classes_))

        for i, tree in enumerate(self.trees):
            A_stacked[i, 0 : tree.A.shape[0], 0 : tree.A.shape[1]] = tree.A
            B_stacked[i, 0 : tree.B.shape[0]] = tree.B
            C_stacked[i, 0 : tree.C.shape[0], 0 : tree.C.shape[1]] = tree.C
            D_stacked[i, 0 : tree.D.shape[0]] = tree.D
            E_stacked[i, 0 : tree.E.shape[0], 0 : tree.E.shape[1]] = tree.E
            probas_stacked[i, 0 : tree.probas.shape[0], 0 : tree.probas.shape[1]] = tree.probas


        self.A_stacked = self.back.hstack([A for A in A_stacked])
        self.B_stacked = B_stacked.reshape(-1)
        self.C_stacked = C_stacked
        self.D_stacked = D_stacked.reshape(-1)
        self.E_stacked = E_stacked
        self.probas_stacked = probas_stacked

        if backend == "torch":
            self.A_stacked, self.B_stacked, self.C_stacked, self.D_stacked, self.E_stacked, self.probas_stacked = map(
                lambda x: x.to(self.device).float(),
                (self.A_stacked, self.B_stacked, self.C_stacked, self.D_stacked, self.E_stacked, self.probas_stacked)
            )

    def convert_to_float(self, tensor):
        if self.backend == "numpy":
            return tensor.astype(np.float32)
        else:
            return tensor.float()

    def permute(self, x, *dims):
        if self.backend == "numpy":
            return x.transpose(*dims)
        else:
            return x.permute(*dims)

    def vote(self, X):
        """Count the vote from each tree for each data point"""
        T = X@self.A_stacked
        T = self.convert_to_float(T < self.B_stacked)
        T = T.reshape(-1, self.n_trees, self.max_internal_nodes)
        T = self.permute(T,1,0,2)

        T = T@self.C_stacked
        T = T.reshape(-1, self.n_trees * self.max_leaves_nodes)
        T = self.convert_to_float(T == self.D_stacked)
        T = T.reshape(self.n_trees, -1, self.max_leaves_nodes)

        T = T@self.probas_stacked
        T = T.sum(axis=0)
        return T

    def predict(self, X):
        predictions = self.vote(X)
        return self.back.argmax(predictions, axis=1)
