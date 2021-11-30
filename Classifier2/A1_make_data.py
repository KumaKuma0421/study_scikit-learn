"""
説明変数の生成
"""
import math
import numpy as np
from numpy.random import default_rng
import pandas as pd
import common


def make_explanatory_variable_0(count, param1, param2):
    """
    説明変数を生成します。
    """
    base_count = int(math.sqrt(count))
    color = np.linspace(start=param1, stop=param2, num=base_count)
    color = np.tile(color, base_count)#.round(2) Pythonのround()は･･･
    shape = np.linspace(start=param1, stop=param2, num=base_count)
    shape = np.repeat(shape, base_count)

    response = pd.DataFrame(
        {"color": color, "shape": shape}, index=np.arange(1, color.size + 1))
    response.index.name = "index"

    return response


def make_explanatory_variable_bias(count, param1, param2, seed):
    """
    説明変数を生成します。
    """
    rng = default_rng(seed=seed)

    count0 = int(count * (param1 / 100))
    feature0 = rng.random(count0)
    count1 = int(count * (param2 / 100))
    feature1 = rng.random(count1) + 1.0
    count2 = count - count0 - count1
    feature2 = rng.random(count2) + 2.0
    stack = np.hstack((feature0, feature1, feature2))
    return rng.choice(stack, size=stack.size, replace=False)


def make_explanatory_variable_1(count, param1, param2):
    """
    説明変数を生成します。
    """
    color = make_explanatory_variable_bias(
        count, param1, param2, common.RANDOM_SEED - 1)
    shape = make_explanatory_variable_bias(
        count, param1, param2, common.RANDOM_SEED + 1)

    response = pd.DataFrame(
        {"color": color, "shape": shape}, index=np.arange(1, count + 1))
    response.index.name = "index"

    return response


ACTION = {
    "grid": {
        "param1": 0.0,
        "param2": 3.0,
        "function": make_explanatory_variable_0,
    },
    "flat": {
        "param1": 33,
        "param2": 33,
        "function": make_explanatory_variable_1,
    },
    "offset": {
        "param1": 5,
        "param2": 55,
        "function": make_explanatory_variable_1,
    },
}


def start_action():
    """処理を開始します。"""
    common.make_path(common.DATA_PATH, None, None)
    common.make_path(common.GRAPH_PATH, None, None)

    action = ACTION[common.STYLE]
    func = action["function"]
    param1 = action["param1"]
    param2 = action["param2"]
    data = func(common.DATA_COUNT, param1, param2)

    file_name = common.make_path(common.DATA_PATH, None, common.EXPLANATORY)
    print(file_name)
    data.to_csv(file_name)


if __name__ == '__main__':
    start_action()
    print("Done.")
