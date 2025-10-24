from typing import List
from copy import deepcopy

def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    res = []
    for i in range(0, len(X), 4):
        r = []
        for j in range(120, 500, 5):
            r.append(X[i][j])
        res.append(r)
    return res


def sum_non_neg_diag(X: List[List[int]]) -> int:
    n = len(X)
    m = len(X[0]) if n > 0 else 0
    s = 0
    cnt = 0
    for k in range(min(n, m)):
        v = X[k][k]
        if v >= 0:
            s += v
            cnt += 1
    return s if cnt > 0 else -1

def replace_values(X: List[List[float]]) -> List[List[float]]:
    n = len(X)
    m = len(X[0]) if n > 0 else 0
    Y = deepcopy(X)
    if n == 0 or m == 0:
        return Y
    means = [sum(X[i][j] for i in range(n)) / n for j in range(m)]
    for j in range(m):
        low = 0.25 * means[j]
        high = 1.5 * means[j]
        for i in range(n):
            if Y[i][j] < low or Y[i][j] > high:
                Y[i][j] = -1
    return Y
