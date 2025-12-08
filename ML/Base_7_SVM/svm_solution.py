import numpy as np
from sklearn.svm import SVC


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features)
    train_target: np.array, (num_elements_train)
    test_features: np.array, (num_elements_test x num_features)

    return: np.array, (num_elements_test)
    """

    clf = SVC(kernel='rbf', C=10)
    clf.fit(train_features, train_target)
    y_pred = clf.predict(test_features)

    return y_pred
