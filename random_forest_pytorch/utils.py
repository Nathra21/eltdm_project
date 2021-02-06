import numpy as np

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
