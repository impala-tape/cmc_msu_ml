import os

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline


class PotentialTransformer:
    """Converts potential matrices into engineered tabular features."""

    def fit(self, x, y=None):
        if x.ndim != 3:
            raise ValueError("Expected shape (n_samples, height, width)")

        h, w = x.shape[1], x.shape[2]
        self._h_norm = max(h - 1, 1)
        self._w_norm = max(w - 1, 1)

        self._rows = np.arange(h, dtype=np.float64)[:, None]
        self._cols = np.arange(w, dtype=np.float64)[None, :]
        self._quantiles = np.linspace(0.01, 0.99, 33)
        self._thresholds = np.linspace(0.0, 20.0, 51)
        self._hist_bins = np.linspace(0.0, 20.0, 41)

        max_radius = float(np.hypot(h - 1, w - 1))
        self._ring_edges = np.linspace(0.0, max_radius, 31)

        self._fitted = True
        return self

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)

    def transform(self, x):
        if not getattr(self, "_fitted", False):
            raise ValueError("Call fit before transform")
        return np.vstack([self._extract_one(sample) for sample in x])

    def _extract_one(self, sample):
        flat = sample.reshape(-1)

        q_vals = np.quantile(flat, self._quantiles)
        iqr = float(q_vals[24] - q_vals[8])
        hist = np.histogram(flat, bins=self._hist_bins, density=True)[0]

        features = [
            float(flat.mean()),
            float(flat.std()),
            float(flat.min()),
            float(flat.max()),
            float(np.median(flat)),
            iqr,
            float(np.sqrt(np.mean(flat * flat))),
        ]
        features.extend(q_vals.tolist())
        features.extend(hist.tolist())

        for threshold in self._thresholds:
            features.append(float(np.mean(flat <= threshold)))

        depth = np.clip(20.0 - sample, 0.0, None)
        mass = float(depth.sum()) + 1e-12

        center_y = float((depth * self._rows).sum() / mass)
        center_x = float((depth * self._cols).sum() / mass)

        delta_y = self._rows - center_y
        delta_x = self._cols - center_x
        var_y = float((depth * (delta_y**2)).sum() / mass)
        var_x = float((depth * (delta_x**2)).sum() / mass)
        cov_xy = float((depth * delta_y * delta_x).sum() / mass)

        min_y, min_x = np.unravel_index(np.argmin(sample), sample.shape)
        max_y, max_x = np.unravel_index(np.argmax(sample), sample.shape)

        features.extend(
            [
                mass / sample.size,
                center_y / self._h_norm,
                center_x / self._w_norm,
                np.sqrt(var_y) / self._h_norm,
                np.sqrt(var_x) / self._w_norm,
                cov_xy / (self._h_norm * self._w_norm),
                min_y / self._h_norm,
                min_x / self._w_norm,
                float(sample[min_y, min_x]),
                max_y / self._h_norm,
                max_x / self._w_norm,
                float(sample[max_y, max_x]),
            ]
        )

        radius = np.sqrt(delta_y**2 + delta_x**2)
        for low, high in zip(self._ring_edges[:-1], self._ring_edges[1:]):
            mask = (radius >= low) & (radius < high)
            if np.any(mask):
                ring_values = sample[mask]
                features.append(float(ring_values.mean()))
                features.append(float(ring_values.std()))
            else:
                features.extend([0.0, 0.0])

        row_profile = sample.mean(axis=1)
        col_profile = sample.mean(axis=0)
        features.extend(np.quantile(row_profile, [0.1, 0.3, 0.5, 0.7, 0.9]).tolist())
        features.extend(np.quantile(col_profile, [0.1, 0.3, 0.5, 0.7, 0.9]).tolist())

        grad_y, grad_x = np.gradient(sample)
        grad_norm = np.hypot(grad_y, grad_x)
        features.extend(
            [
                float(grad_norm.mean()),
                float(grad_norm.std()),
                float(np.quantile(grad_norm, 0.85)),
                float(np.quantile(grad_norm, 0.95)),
                float(np.quantile(grad_norm, 0.99)),
            ]
        )

        return np.asarray(features, dtype=np.float64)


def load_dataset(data_dir):
    files, x, y = [], [], []
    for file_name in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file_name))
        files.append(file_name)
        x.append(potential["data"])
        y.append(float(potential["target"]))
    return files, np.asarray(x), np.asarray(y)


def train_model_and_predict(train_dir, test_dir):
    _, x_train, y_train = load_dataset(train_dir)
    test_files, x_test, _ = load_dataset(test_dir)

    regressor = Pipeline(
        [
            ("features", PotentialTransformer()),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=900,
                    random_state=42,
                    n_jobs=-1,
                    max_features=0.75,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )

    regressor.fit(x_train, y_train)
    predictions = regressor.predict(x_test)
    return {file_name: float(value) for file_name, value in zip(test_files, predictions)}
