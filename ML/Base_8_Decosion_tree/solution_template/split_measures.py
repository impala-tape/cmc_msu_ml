from collections import Counter
from math import log


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    labels = list(sample)
    if len(labels) == 0:
        return {'gini': 0.0, 'entropy': 0.0, 'error': 0.0}

    counts = Counter(labels).values()
    total = float(len(labels))
    probs = [count / total for count in counts]

    gini = 1.0 - sum(p * p for p in probs)
    entropy = -sum(p * log(p) for p in probs)
    error = 1.0 - max(probs)

    measures = {'gini': float(gini), 'entropy': float(entropy), 'error': float(error)}
    return measures
