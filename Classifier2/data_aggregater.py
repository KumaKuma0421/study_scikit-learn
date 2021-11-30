import os
import numpy as np
import pandas as pd
import common

base_path = r"Z:\DATA_2_予戦"
point_pattern = [1000]
std_pattern = [0]
position_pattern = ["flat", "grid", "offset"]
param_pattern = [0, 1, 2, 3]
extension_pattern = [1, 2, 3]
aggregate_file_name_1 = os.path.join(base_path, "aggregate_1.csv")
aggregate_file_name_2 = os.path.join(base_path, "aggregate_2.csv")
aggregate_file_name_3 = os.path.join(base_path, "aggregate_3.csv")

columns1 = [
    "pattern",
    "point",
    "position",
    "standard",
    "param",
    "estimator",
    "validator",
    "mean_fit_time",
    "std_fit_time",
    "mean_score_time",
    "std_score_time",
    "params",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
    "best_train_score",
    "best_test_score",
]

columns2 = [
    "pattern",
    "point",
    "position",
    "standard",
    "param",
    "estimator",
    "validator",
    "0-0",
    "0-1",
    "0-2",
    "1-0",
    "1-1",
    "1-2",
    "2-0",
    "2-1",
    "2-2",
    "best_params",
    "best_train_score",
    "best_test_score",
]

columns3 = [
    "pattern",
    "point",
    "position",
    "standard",
    "param",
    "estimator",
    "validator",
    "precision-0",
    "precision-1",
    "precision-2",
    "recall-0",
    "recall-1",
    "recall-2",
    "f1-score-0",
    "f1-score-1",
    "f1-score-2",
    "support-0",
    "support-1",
    "support-2",
    "accuracy",
    "macro avg",
    "weighted avg",
    "best_params",
    "best_train_score",
    "best_test_score",
]


def aggregate_file_1(
    point, position, std, param, pattern, estimator, validator, extension, target_file
):
    """ファイルを集約していきます。"""
    print(target_file)

    data = pd.read_csv(target_file, index_col=0)
    data["pattern"] = pattern
    data["point"] = point
    data["position"] = position
    data["standard"] = std
    data["param"] = param
    if not os.path.exists(aggregate_file_name_1):
        data[columns1].to_csv(aggregate_file_name_1)
    else:
        data[columns1].to_csv(aggregate_file_name_1, mode="a", header=False)


def aggregate_file_2(
    point, position, std, param, pattern, estimator, validator, extension, target_file
):
    """ファイルを集約していきます。"""
    print(target_file)

    data = pd.read_csv(target_file, index_col=0)
    data["pattern"] = pattern
    data["point"] = point
    data["position"] = position
    data["standard"] = std
    data["param"] = param
    data["estimator"] = estimator
    data["validator"] = validator
    data["0-0"] = data.iloc[0, 0]
    data["0-1"] = data.iloc[0, 1]
    data["0-2"] = data.iloc[0, 2]
    data["1-0"] = data.iloc[1, 0]
    data["1-1"] = data.iloc[1, 1]
    data["1-2"] = data.iloc[1, 2]
    data["2-0"] = data.iloc[2, 0]
    data["2-1"] = data.iloc[2, 1]
    data["2-2"] = data.iloc[2, 2]
    if not os.path.exists(aggregate_file_name_2):
        data.head(1)[columns2].to_csv(aggregate_file_name_2)
    else:
        data.head(1)[columns2].to_csv(aggregate_file_name_2, mode="a", header=False)


def aggregate_file_3(
    point, position, std, param, pattern, estimator, validator, extension, target_file
):
    """ファイルを集約していきます。"""
    print(target_file)

    data = pd.read_csv(target_file, index_col=0)
    data["pattern"] = pattern
    data["point"] = point
    data["position"] = position
    data["standard"] = std
    data["param"] = param
    data["estimator"] = estimator
    data["validator"] = validator
    data["precision-0"] = data.iloc[0, 0]
    data["precision-1"] = data.iloc[0, 1]
    data["precision-2"] = data.iloc[0, 2]
    data["recall-0"] = data.iloc[1, 0]
    data["recall-1"] = data.iloc[1, 1]
    data["recall-2"] = data.iloc[1, 2]
    data["f1-score-0"] = data.iloc[2, 0]
    data["f1-score-1"] = data.iloc[2, 1]
    data["f1-score-2"] = data.iloc[2, 2]
    data["support-0"] = data.iloc[3, 0]
    data["support-1"] = data.iloc[3, 1]
    data["support-2"] = data.iloc[3, 2]
    if not os.path.exists(aggregate_file_name_3):
        data.head(1)[columns3].to_csv(aggregate_file_name_3)
    else:
        data.head(1)[columns3].to_csv(aggregate_file_name_3, mode="a", header=False)


def search_file(point, position, std, param, pattern, estimator, validator, extension):
    """ファイル名を構築します。"""
    folder_name = f"DATA_2_{point}_{position}_s{std}_p{param}"
    sub_foler_name = "analyze"
    file_name = f"analyze_#{pattern:02}_{estimator}_{validator}_{extension}.csv"
    target_file = os.path.join(base_path, folder_name, sub_foler_name, file_name)
    if os.path.exists(target_file):
        if extension == 1:
            aggregate_file_1(
                point, position, std, param, pattern, estimator, validator, extension, target_file
            )
        elif extension == 2:
            aggregate_file_2(
                point, position, std, param, pattern, estimator, validator, extension, target_file
            )
        elif extension == 3:
            aggregate_file_3(
                point, position, std, param, pattern, estimator, validator, extension, target_file
            )
        else:
            raise ValueError(f"Invalid file name {target_file}.")


def reset_index_files(target_file):
    data = pd.read_csv(target_file, index_col=0)
    data.reset_index(drop=True, inplace=True)
    data.index = np.arange(1, data.index.size + 1)
    data.index.name = "index"
    data.to_csv(target_file)


def start_action():
    """処理を開始します。"""
    if os.path.exists(aggregate_file_name_1):
        os.remove(aggregate_file_name_1)
    if os.path.exists(aggregate_file_name_2):
        os.remove(aggregate_file_name_2)
    if os.path.exists(aggregate_file_name_3):
        os.remove(aggregate_file_name_3)

    for point in point_pattern:
        for position in position_pattern:
            for std in std_pattern:
                for param in param_pattern:
                    for pattern in common.PATTERN:
                        for estimator in common.ESTIMATORS:
                            for validator in common.VALIDATORS:
                                for extension in extension_pattern:
                                    search_file(
                                        point,
                                        position,
                                        std,
                                        param,
                                        pattern,
                                        estimator,
                                        validator,
                                        extension,
                                    )

    reset_index_files(aggregate_file_name_1)
    reset_index_files(aggregate_file_name_2)
    reset_index_files(aggregate_file_name_3)


if __name__ == "__main__":
    start_action()
    print("Done.")
