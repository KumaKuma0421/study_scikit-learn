"""
分類結果をプロット
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


def yield_target():
    """処理を開始します。"""
    for i in common.PATTERN:
        row = 0
        for estimator in common.ESTIMATORS:
            yield i, row, estimator
            row += 1


def read_plot_data(idx, estimator_name, validator_name):
    """
    ファイルから等高線情報を取得します。
    """
    target_name = common.make_path(
        common.DATA_PATH,
        common.ANALYZE_SUB_PATH,
        f"analyze_#{idx:02}_{estimator_name}_{validator_name}.csv",
    )
    print(target_name)

    data = pd.read_csv(target_name, index_col=0)
    x_flat = data["x_value"].values
    y_flat = data["y_value"].values
    z_flat = data["z_value"].values
    row_max = data["row"].max() + 1
    col_max = data["col"].max() + 1
    x_value = np.reshape(x_flat, (row_max, col_max))
    y_value = np.reshape(y_flat, (row_max, col_max))
    z_value = np.reshape(z_flat, (row_max, col_max))

    return x_value, y_value, z_value


def start_action():
    """エンジンが提供する解析情報をプロットします。"""
    color_map = plt.cm.RdBu
    columns = len(common.PATTERN)
    rows = len(common.ESTIMATORS)

    for validator in common.VALIDATORS:
        fig, axes = plt.subplots(rows, columns, figsize=(common.PIX_1600, common.PIX_1600))

        for pattern, row, estimator in yield_target():
            x_value, y_value, z_value = read_plot_data(pattern, estimator, validator)
            axes[row, pattern - 1].contour(
                x_value, y_value, z_value, cmap=color_map, alpha=0.8
            )
            axes[row, pattern - 1].set_xlim(x_value.min(), x_value.max())
            axes[row, pattern - 1].set_ylim(y_value.min(), y_value.max())
            axes[row, pattern - 1].set_xticks(())
            axes[row, pattern - 1].set_yticks(())
            axes[row, pattern - 1].set_title(f"#{pattern:02}_{estimator}_{validator}", fontsize=8)

        plt.tight_layout()
        save_file = common.make_path(
            common.GRAPH_PATH, common.ANALYZE_SUB_PATH, f"analyze_{validator}.png"
        )
        plt.savefig(save_file)
        #plt.show()
        plt.close()


if __name__ == "__main__":
    start_action()
    print("Done.")
