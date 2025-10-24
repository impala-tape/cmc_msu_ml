import numpy as np

def get_part_of_array(X: np.ndarray) -> np.ndarray:
    return X[::4, 120:500:5]


def sum_non_neg_diag(X: np.ndarray) -> int:
    diag = np.diag(X)
    diagNotNeg = diag[diag >= 0]
    if diagNotNeg.size < 1: 
        return -1
    return np.sum(diagNotNeg)


def replace_values(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.copy()
    means = X.mean(axis=0)
    low = 0.25 * means
    high = 1.5 * means
    Y = X.copy()
    mask = (Y < low) | (Y > high)
    Y[mask] = -1
    return Y
