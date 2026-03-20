import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        if x.ndim != 3:
            raise ValueError("Expected input with shape (n_samples, height, width)")
        height, width = x.shape[1], x.shape[2]
        self._rows = np.arange(height, dtype=np.float64)[:, None]
        self._cols = np.arange(width, dtype=np.float64)[None, :]
        self._norm_h = max(height - 1, 1)
        self._norm_w = max(width - 1, 1)
        self._quantiles = np.linspace(0.02, 0.98, 25)
        self._thresholds = np.linspace(0.0, 20.0, 41)
        max_radius = float(np.sqrt((height - 1) ** 2 + (width - 1) ** 2))
        self._radial_bins = np.linspace(0.0, max_radius, 37)
        self._fitted = True
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        if not getattr(self, "_fitted", False):
            raise ValueError("Transformer has to be fitted before calling transform")
        features = [self._extract_features(sample) for sample in x]
        return np.vstack(features)

    def _extract_features(self, sample):
        flat = sample.ravel()
        feature_vector = [
            float(flat.mean()),
            float(flat.std()),
            float(flat.min()),
            float(flat.max()),
            float(np.sqrt(np.mean(flat * flat))),
        ]

        feature_vector.extend(np.quantile(flat, self._quantiles).tolist())

        for threshold in self._thresholds:
            feature_vector.append(float(np.mean(flat < threshold)))

        depth_weights = np.clip(20.0 - sample, 0.0, None)
        weight_sum = float(depth_weights.sum()) + 1e-12

        row_center = float((depth_weights * self._rows).sum() / weight_sum)
        col_center = float((depth_weights * self._cols).sum() / weight_sum)
        row_var = float(
            (depth_weights * (self._rows - row_center) ** 2).sum() / weight_sum
        )
        col_var = float(
            (depth_weights * (self._cols - col_center) ** 2).sum() / weight_sum
        )
        row_col_cov = float(
            (
                depth_weights
                * (self._rows - row_center)
                * (self._cols - col_center)
            ).sum()
            / weight_sum
        )

        min_row, min_col = np.unravel_index(np.argmin(sample), sample.shape)
        max_row, max_col = np.unravel_index(np.argmax(sample), sample.shape)

        feature_vector.extend(
            [
                weight_sum / sample.size,
                row_center / self._norm_h,
                col_center / self._norm_w,
                np.sqrt(row_var) / self._norm_h,
                np.sqrt(col_var) / self._norm_w,
                row_col_cov / (self._norm_h * self._norm_w),
                min_row / self._norm_h,
                min_col / self._norm_w,
                float(sample[min_row, min_col]),
                max_row / self._norm_h,
                max_col / self._norm_w,
                float(sample[max_row, max_col]),
            ]
        )

        radius = np.sqrt(
            (self._rows - row_center) ** 2 + (self._cols - col_center) ** 2
        )
        for i in range(len(self._radial_bins) - 1):
            in_ring = (radius >= self._radial_bins[i]) & (
                radius < self._radial_bins[i + 1]
            )
            if np.any(in_ring):
                ring_values = sample[in_ring]
                feature_vector.append(float(ring_values.mean()))
                feature_vector.append(float(ring_values.std()))
            else:
                feature_vector.extend([0.0, 0.0])

        grad_x, grad_y = np.gradient(sample)
        grad_norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        feature_vector.extend(
            [
                float(grad_norm.mean()),
                float(grad_norm.std()),
                float(np.quantile(grad_norm, 0.9)),
                float(np.quantile(grad_norm, 0.99)),
            ]
        )
        return np.array(feature_vector, dtype=np.float64)


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    regressor = Pipeline(
        [
            ("vectorizer", PotentialTransformer()),
            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=700,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
