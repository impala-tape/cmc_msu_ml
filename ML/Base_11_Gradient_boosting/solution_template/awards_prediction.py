import pandas as pd
from lightgbm import LGBMRegressor
from numpy import ndarray


TEXT_COLUMNS = [
    "genres",
    "directors",
    "filming_locations",
    "keywords",
]

CATEGORICAL_COLUMNS = TEXT_COLUMNS + [
    "actor_0_gender",
    "actor_1_gender",
    "actor_2_gender",
]

MODEL_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "max_depth": 8,
    "random_state": 0,
    "verbosity": -1,
    "n_jobs": -1,
}


def _join_list_feature(values: object) -> str:
    if isinstance(values, list):
        return " | ".join(map(str, values))
    return str(values)


def _list_count(values: object) -> int:
    if isinstance(values, list):
        return len(values)
    return 0


def _list_char_len(values: object) -> int:
    if isinstance(values, list):
        return sum(len(str(value)) for value in values)
    return len(str(values))


def _prepare_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df_train.copy()
    test = df_test.copy()

    for df in (train, test):
        for column in TEXT_COLUMNS:
            values = df[column]
            df[f"{column}_count"] = values.apply(_list_count)
            df[f"{column}_char_len"] = values.apply(_list_char_len)
            df[column] = values.apply(_join_list_feature)

    for column in CATEGORICAL_COLUMNS:
        train_values = train[column].astype(str)
        test_values = test[column].astype(str)
        categories = pd.Index(pd.concat([train_values, test_values]).unique())

        train[column] = pd.Categorical(train_values, categories=categories)
        test[column] = pd.Categorical(test_values, categories=categories)

    numeric_columns = [
        column
        for column in train.columns
        if column not in CATEGORICAL_COLUMNS
    ]
    for column in numeric_columns:
        median = train[column].median()
        train[column] = train[column].fillna(median)
        test[column] = test[column].fillna(median)

    return train, test


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    y_train = df_train.pop("awards")
    x_train, x_test = _prepare_features(df_train, df_test)

    regressor = LGBMRegressor(**MODEL_PARAMS)
    regressor.fit(
        x_train,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    return regressor.predict(x_test)
