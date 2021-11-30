import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


aggregate_file_name_1 = os.path.join(common.BASE_PATH, "aggregate_1.csv")
target_params = [0, 1, 2] # 3は計測していない

def plot_data(data):
    """データをプロットします。"""
    fig, axes = plt.subplots(1, 1, figsize=[common.PIX_1600, common.PIX_800])

    x_value = np.arange(data.index.size)
    x_label = data.index.values
    param_0 = np.zeros(data.iloc[:, 0].shape)
    param_1 = np.zeros(data.iloc[:, 0].shape)
    param_2 = np.zeros(data.iloc[:, 0].shape)
    param_3 = np.zeros(data.iloc[:, 0].shape)
    width = 0.3
    for validator in common.VALIDATORS:
        for param in target_params:
            col = f"mean_fit_time({validator}-p{param})"
            value = data.loc[:, col]
            if param == 0:
                bar_obj = axes.bar(x_value - width, value, width=width, bottom=param_0, label=f"{validator} {param}")
                axes.bar_label(bar_obj, label_type="center", fmt="%.3f", fontsize=8)
                param_0 += value
            elif param == 1:
                bar_obj = axes.bar(x_value, value, width=width, bottom=param_1, label=f"{validator} {param}")
                axes.bar_label(bar_obj, label_type="center", fmt="%.3f", fontsize=8)
                param_1 += value
            else:
                bar_obj = axes.bar(x_value + width, value, width=width, bottom=param_3, label=f"{validator} {param}")
                axes.bar_label(bar_obj, label_type="center", fmt="%.3f", fontsize=8)
                param_3 += value
    axes.set_xticks(x_value)
    axes.set_xticklabels(x_label)
    axes.set_xlabel("estimator-validator")
    axes.set_ylabel("mean_fit_time")
    axes.set_title("estimator-validator/mean_fit_time score")
    axes.legend()
    plt.tight_layout()
    plt.show()

    save_file = common.make_path(common.BASE_PATH, None, f"plot_data_bar_1.png")
    fig.savefig(save_file)
    plt.close()


def abstract_data(target_file):
    """Validatorによって結果は変わってくるのか？"""
    data = pd.read_csv(target_file, index_col=0)
    group_data_1 = data.groupby(by=["estimator", "validator", "param"])
    group_data_2 = pd.DataFrame()

    for estimator in common.ESTIMATORS:
        col_sum = 0
        for validator in common.VALIDATORS:
            for param in target_params:
                row = f"{estimator}"
                col = f"mean_fit_time({validator}-p{param})"
                mean = group_data_1.get_group((estimator, validator, param))[
                    "mean_fit_time"].mean()
                group_data_2.loc[row, col] = mean
                col_sum += mean
        group_data_2.loc[row, "sum"] = col_sum

    group_data_2.index.name = "index"
    plot_data(group_data_2.sort_values(by="sum", ascending=False))


if __name__ == "__main__":
    abstract_data(aggregate_file_name_1)

    print("Done.")
