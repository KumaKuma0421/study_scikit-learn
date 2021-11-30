"""
scikit-learnのmake_classification()等によるデータの生成
"""
import numpy as np
import pandas as pd
from sklearn.datasets import \
    make_moons, make_circles, \
    make_classification, make_blobs, \
    make_gaussian_quantiles
import common


BALANCE = [0.7, 0.3]


def make_data_1():
    """make_classfication()でデータを作成します。"""
    return make_classification(
        n_samples=common.DATA_COUNT,
        n_informative=common.FEATURE_SIZE,
        n_features=common.FEATURE_SIZE,
        n_classes=common.TARGET_SIZE,
        random_state=common.RANDOM_SEED,
        n_redundant=0,
        n_clusters_per_class=2,
        weights=BALANCE)


def make_data_2():
    """make_blobs()でデータを作成します。"""
    return make_blobs(
        random_state=common.RANDOM_SEED,
        n_samples=common.DATA_COUNT,
        n_features=common.FEATURE_SIZE,
        centers=common.TARGET_SIZE,
        cluster_std=1.5)


def make_data_3():
    """make_gaussian_quantiles()でデータを作成します。"""
    return make_gaussian_quantiles(
        random_state=common.RANDOM_SEED,
        n_samples=common.DATA_COUNT,
        n_features=common.FEATURE_SIZE,
        n_classes=common.TARGET_SIZE)


def make_data_4():
    """make_moons()でデータを作成します。"""
    return make_moons(
        n_samples=common.DATA_COUNT,
        random_state=common.RANDOM_SEED, noise=0.0)


def make_data_5():
    """make_circles()でデータを作成します。"""
    return make_circles(
        n_samples=common.DATA_COUNT,
        random_state=common.RANDOM_SEED,
        noise=0.0, factor=0.5)


def make_data_6():
    """y = -ax + bの式でデータを作成します。"""
    row = int(common.DATA_COUNT / 2)
    f1t0 = np.linspace(0, 1, row).reshape(-1, 1)
    f1t1 = np.linspace(0, 1, row).reshape(-1, 1)
    f2t0 = 1.5 - f1t0 * 1.2
    f2t1 = 1 - f1t1 * 1.2
    t_0 = np.repeat(0, row).reshape(-1, 1)
    t_1 = np.repeat(1, row).reshape(-1, 1)
    f_1 = np.vstack((f1t0, f1t1))
    f_2 = np.vstack((f2t0, f2t1))
    value_x = np.hstack((f_1, f_2))
    value_y = np.vstack((t_0, t_1))
    return (value_x, value_y)


strategy = [
    make_data_1,
    make_data_2,
    make_data_3,
    make_data_4,
    make_data_5,
    make_data_6
]


def start_action():
    """処理を開始します。"""
    common.make_path(common.DATA_PATH, None, None)

    for i in common.PATTERN:
        data = (strategy[i - 1])()

        data_frame = pd.DataFrame(
            data[0],
            columns=[f"feature{n + 1}" for n in range(common.FEATURE_SIZE)],
            index=np.arange(1, common.DATA_COUNT + 1))
        data_frame["target"] = data[1]
        data_frame.index.name = "index"

        file_name = common.make_path(
            common.DATA_PATH, common.DATA_SUB_PATH, f"data_#{(i):02}.csv")
        print(file_name)
        data_frame.to_csv(file_name)


if __name__ == '__main__':
    start_action()
    print("Done.")
