import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import common


SPLIT_SIZE = 10


def get_validator(name):
    """
    交差検証のオブジェクトを取得します。
    """
    response = None

    if name == "KFold":
        response = KFold(
            n_splits=SPLIT_SIZE, random_state=common.RANDOM_SEED, shuffle=True
        )
    elif name == "StratifiedKFold":
        response = StratifiedKFold(
            n_splits=SPLIT_SIZE, random_state=common.RANDOM_SEED, shuffle=True
        )
    elif name == "ShuffleSplit":
        response = ShuffleSplit(n_splits=SPLIT_SIZE, random_state=common.RANDOM_SEED)
    elif name == "TimeSeriesSplit":
        response = TimeSeriesSplit(n_splits=SPLIT_SIZE)
    else:
        pass

    return response


def color_value(value):
    if value == 0:
        return "k"
    elif value == 1:
        return "g"
    else:
        return "r"


def marker_value(value):
    if value == 0:
        return "X"
    elif value == 1:
        return "o"
    else:
        return "*"


def plot_data(idx, validator_name, split_x, split_y):
    fig, axes = plt.subplots(5, 2, figsize=[common.PIX_1600, common.PIX_1600])
    count = 0

    for row in range(5):
        for col in range(2):
            feature0 = split_x[count][:, 0]
            feature1 = split_x[count][:, 1]

            width = 0.2
            color_values = [color_value(n) for n in split_y[count]]
            # marker_values = [marker_value(n) for n in split_y[count]]
            # marker_values = ["X", "o", "*"]
            marker_values = "o"  # なぜか上記はエラーになる。
            ALPHA = 0.5

            axes[row, col].scatter(
                feature0,
                feature1,
                c=color_values,
                marker=marker_values,
                linewidth=width,
                alpha=ALPHA,
            )

            axes[row, col].grid(axis="both")
            axes[row, col].set_xlabel("color")
            axes[row, col].set_ylabel("shape")
            axes[row, col].set_title("color/shape")

            count += 1

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH,
        common.DATA_SUB_PATH,
        f"validate_#{idx:02}_{validator_name}.png",
    )
    fig.savefig(save_file)
    plt.close()


def analyze_core(idx, x_train, x_test, y_train, y_test):
    for validator_name in common.VALIDATORS:
        print(f"=={validator_name}==")
        validator = get_validator(validator_name)
        split_x = []
        split_y = []
        for idx_train, idx_test in validator.split(X=x_train, y=y_train):
            split_x.append(x_train[idx_train])
            split_y.append(y_train[idx_train])
        plot_data(idx, validator_name, split_x, split_y)


def analyze(idx):
    """
    分析を開始します。
    """
    target_name = common.make_path(
        common.DATA_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}.csv"
    )
    print("=" * 80)
    print(target_name)

    data = pd.read_csv(target_name, index_col=0)
    data_x = data.iloc[:, 0:2].values
    if common.STANDARD_SW == 1:
        data_x = StandardScaler().fit_transform(data_x)
    data_y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

    analyze_core(idx, x_train, x_test, y_train, y_test)


def start_action():
    """処理を開始します。"""
    for i in common.PATTERN:
        analyze(i)


if __name__ == "__main__":
    start_action()
    print("Done.")
