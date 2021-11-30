"""
分類結果をプロット
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


def estimator_response_plot(
        idx, estimator_name, validator_name, x_value, y_value,
        z1_value, z2_value, z3_value):
    """エンジンが提供する解析情報をプロットします。"""
    color_map = plt.cm.RdBu
    fig, axes = plt.subplots(1, 1, figsize=(common.PIX_1600, common.PIX_1600))

    axes.contourf(x_value, y_value, z1_value, cmap=color_map, alpha=0.8)
    axes.contourf(x_value, y_value, z2_value, cmap=color_map, alpha=0.8)
    axes.contourf(x_value, y_value, z3_value, cmap=color_map, alpha=0.8)
    axes.set_xlim(x_value.min(), x_value.max())
    axes.set_ylim(y_value.min(), y_value.max())
    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_title(f"analyze_#{idx:02}_{estimator_name}_{validator_name}")
    plt.tight_layout()
    save_file = common.make_path(
        common.GRAPH_PATH, common.ANALYZE_SUB_PATH,
        f"analyze_#{idx:02}_{estimator_name}_{validator_name}.png")
    plt.savefig(save_file)
    # plt.show()
    plt.close()


def analyze(idx, estimator_name, validator_name):
    """
    分析を開始します。
    """
    target_name = common.make_path(
        common.DATA_PATH, common.ANALYZE_SUB_PATH,
        f"analyze_#{idx:02}_{estimator_name}_{validator_name}.csv")
    print(target_name)

    data = pd.read_csv(target_name, index_col=0)
    x_flat = data["x_value"].values
    y_flat = data["y_value"].values
    z1_flat = data["z1_value"].values
    z2_flat = data["z2_value"].values
    z3_flat = data["z3_value"].values
    row_max = data["row"].max() + 1
    col_max = data["col"].max() + 1
    x_value = np.reshape(x_flat, (row_max, col_max))
    y_value = np.reshape(y_flat, (row_max, col_max))
    z1_value = np.reshape(z1_flat, (row_max, col_max))
    z2_value = np.reshape(z2_flat, (row_max, col_max))
    z3_value = np.reshape(z3_flat, (row_max, col_max))

    estimator_response_plot(idx, estimator_name,
                            validator_name, x_value, y_value,
                            z1_value, z2_value, z3_value)


def start_action():
    """処理を開始します。"""
    for i in common.PATTERN:
        for estimator in common.ESTIMATORS:
            for validator in common.VALIDATORS:
                analyze(i, estimator, validator)


if __name__ == '__main__':
    start_action()
    print("Done.")
