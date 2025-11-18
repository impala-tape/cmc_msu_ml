import numpy as np


class Preprocessor:
    """
    Base preprocessor interface compatible with scikit-like API.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, X, Y=None):
        """
        Base fit does nothing and returns self.
        """
        self._fitted = True
        return self

    def transform(self, X):
        """
        Identity transform for the base class.
        """
        return X

    def fit_transform(self, X, Y=None):
        """
        Convenience method: fit and then transform.
        """
        self.fit(X, Y)
        return self.transform(X)


class MyOneHotEncoder(Preprocessor):
    """
    One-hot encoder for categorical features.

    For each column i and its sorted list of unique values
    [a_1, ..., a_{|f_i|}] we create |f_i| binary features.
    For i < j the features of column i go before the features
    of column j.
    """

    def __init__(self, dtype=np.float64):
        super().__init__()
        self.dtype = dtype
        self.columns_ = None
        self.categories_ = None
        self._value_to_index = None
        self._feature_offsets = None
        self.n_output_features_ = None

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        # Remember column order
        self.columns_ = list(X.columns)
        self.categories_ = []
        self._value_to_index = []

        # Collect sorted unique values for each feature
        for col in self.columns_:
            values = X[col].to_numpy()
            # Use Python's sorted to support both numbers and strings
            uniques = sorted(set(values.tolist()))
            self.categories_.append(uniques)
            mapping = {val: idx for idx, val in enumerate(uniques)}
            self._value_to_index.append(mapping)

        # Pre-compute global offsets for each feature
        self._feature_offsets = []
        offset = 0
        for uniques in self.categories_:
            self._feature_offsets.append(offset)
            offset += len(uniques)
        self.n_output_features_ = offset

        self._fitted = True
        return self

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        if not self._fitted:
            raise RuntimeError("MyOneHotEncoder is not fitted yet")

        n_objects = X.shape[0]
        result = np.zeros((n_objects, self.n_output_features_), dtype=self.dtype)

        # We rely on the same column order as in fit
        for row_idx in range(n_objects):
            for j, col in enumerate(self.columns_):
                val = X.iloc[row_idx, j]
                mapping = self._value_to_index[j]
                # According to the task statement we do not
                # need to handle unseen values separately.
                col_offset = self._feature_offsets[j]
                local_idx = mapping[val]
                result[row_idx, col_offset + local_idx] = 1

        return result

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:
    """
    Simple counter encoding.

    For every feature and every category value a we store:
      mean  = average target for this value
      freq  = fraction of training objects with this value
      rel   = (mean + a) / (freq + b)

    Each original feature is transformed into three features
    [mean, freq, rel], so the output shape is
    [n_objects, 3 * n_features].
    """

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.columns_ = None
        self.n_objects_ = None
        # stats_[j] is a dict: value -> (mean, count)
        self.stats_ = None

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.columns_ = list(X.columns)
        y = np.asarray(Y)
        n_objects, n_features = X.shape
        self.n_objects_ = n_objects
        self.stats_ = []

        for col in self.columns_:
            col_values = X[col].to_numpy()
            stats_for_col = {}
            # Use set to iterate over unique categories
            for val in set(col_values.tolist()):
                mask = (col_values == val)
                count = int(mask.sum())
                if count == 0:
                    mean = 0.0
                else:
                    mean = float(y[mask].mean())
                stats_for_col[val] = (mean, count)
            self.stats_.append(stats_for_col)

        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        if self.stats_ is None:
            raise RuntimeError("SimpleCounterEncoder is not fitted yet")

        n_objects, n_features = X.shape
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for row_idx in range(n_objects):
            for j, col in enumerate(self.columns_):
                val = X.iloc[row_idx, j]
                mean, count = self.stats_[j][val]
                freq = count / float(self.n_objects_) if self.n_objects_ > 0 else 0.0
                rel = (mean + a) / (freq + b) if (freq + b) != 0 else 0.0

                base = 3 * j
                result[row_idx, base] = mean
                result[row_idx, base + 1] = freq
                result[row_idx, base + 2] = rel

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:
    """
    K-fold version of counter encoding.

    For every fold we compute statistics on the remaining
    folds only, and then use these statistics to encode
    objects from the held-out fold.
    """

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.columns_ = None
        # For each fold we keep per-column stats like in SimpleCounterEncoder
        self.fold_stats_ = None
        self.fold_sizes_ = None
        # Mapping from object index to fold id
        self.sample_fold_ = None

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.columns_ = list(X.columns)
        y = np.asarray(Y)
        n_objects, n_features = X.shape

        # Prepare folds
        splits = list(group_k_fold(n_objects, n_splits=self.n_folds, seed=seed))
        self.fold_stats_ = []
        self.fold_sizes_ = []
        self.sample_fold_ = np.empty(n_objects, dtype=int)

        for fold_id, (valid_idx, train_idx) in enumerate(splits):
            # Remember which fold each object belongs to
            self.sample_fold_[valid_idx] = fold_id

            self.fold_sizes_.append(len(train_idx))

            # Compute statistics on the training part of this fold
            stats_for_fold = []
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]

            for col in self.columns_:
                col_values = X_train[col].to_numpy()
                stats_for_col = {}
                for val in set(col_values.tolist()):
                    mask = (col_values == val)
                    count = int(mask.sum())
                    if count == 0:
                        mean = 0.0
                    else:
                        mean = float(y_train[mask].mean())
                    stats_for_col[val] = (mean, count)
                stats_for_fold.append(stats_for_col)

            self.fold_stats_.append(stats_for_fold)

        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        if self.fold_stats_ is None:
            raise RuntimeError("FoldCounters is not fitted yet")

        n_objects, n_features = X.shape
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for row_idx in range(n_objects):
            fold_id = int(self.sample_fold_[row_idx])
            stats_for_fold = self.fold_stats_[fold_id]
            fold_size = float(self.fold_sizes_[fold_id])

            for j, col in enumerate(self.columns_):
                val = X.iloc[row_idx, j]
                mean, count = stats_for_fold[j][val]
                freq = count / fold_size if fold_size > 0 else 0.0
                rel = (mean + a) / (freq + b) if (freq + b) != 0 else 0.0

                base = 3 * j
                result[row_idx, base] = mean
                result[row_idx, base + 1] = freq
                result[row_idx, base + 2] = rel

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y, seed=1)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Sorted unique feature values determine the order of weights
    unique_vals = np.unique(x)
    w = np.zeros(unique_vals.shape[0], dtype=np.float64)

    for idx, val in enumerate(unique_vals):
        mask = (x == val)
        count = mask.sum()
        if count == 0:
            w[idx] = 0.0
        else:
            # For logistic loss with one-hot features the optimum
            # is the empirical mean of the target for this value.
            w[idx] = y[mask].mean()

    return w
