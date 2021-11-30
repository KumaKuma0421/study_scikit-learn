import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


aggregate_file_name_3 = os.path.join(common.BASE_PATH, "aggregate_3.csv")
target_params = [0, 3]
target_std = [0, 1]

def plot_data_bar(data):
    """データをプロットします。"""
    fig, axes = plt.subplots(1, 1, figsize=[common.PIX_1600, common.PIX_800])

    x_value = np.arange(data.index.size)
    x_label = data.index.values
    left_base_100 = np.zeros(data.iloc[:, 0].shape)
    left_base_200 = np.zeros(data.iloc[:, 0].shape)
    right_base_100 = np.zeros(data.iloc[:, 0].shape)
    right_base_200 = np.zeros(data.iloc[:, 0].shape)
    for validator in common.VALIDATORS:
        for param in target_params:
            for std in target_std:
                col = f"accuracy({validator}-s{std}-p{param})"
                value = data.loc[:, col]
                if param == 0:
                    if std == 0:
                        bar_obj = axes.bar(x_value - 0.3, value, width=0.2, bottom=left_base_100,
                        label=f"{validator} {param} {std}")
                        axes.bar_label(bar_obj, label_type="center", fmt="%.3f")
                        left_base_100 += value
                    else:
                        bar_obj = axes.bar(x_value - 0.1, value, width=0.2, bottom=left_base_200,
                        label=f"{validator} {param} {std}")
                        axes.bar_label(bar_obj, label_type="center", fmt="%.3f")
                        left_base_200 += value
                else:
                    if std == 0:
                        bar_obj = axes.bar(x_value + 0.1, value, width=0.2, bottom=right_base_100,
                        label=f"{validator} {param} {std}")
                        axes.bar_label(bar_obj, label_type="center", fmt="%.3f")
                        right_base_100 += value
                    else:
                        bar_obj = axes.bar(x_value + 0.3, value, width=0.2, bottom=right_base_200,
                        label=f"{validator} {param} {std}")
                        axes.bar_label(bar_obj, label_type="center", fmt="%.3f")
                        right_base_200 += value
    axes.set_xticks(x_value)
    axes.set_xticklabels(x_label)
    axes.set_xlabel("estimator-validator")
    axes.set_ylabel("accuracy")
    axes.set_title("estimator-validator/accuracy score")
    axes.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    save_file = common.make_path(common.BASE_PATH, None, f"plot_data_bar_2.png")
    fig.savefig(save_file)
    plt.close()


def abstract_data(target_file):
    """Validatorによって結果は変わってくるのか？"""
    data = pd.read_csv(target_file, index_col=0)
    group_data_1 = data.groupby(by=["estimator", "validator", "param", "standard"])
    group_data_2 = pd.DataFrame()

    for estimator in common.ESTIMATORS:
        col_sum = 0
        for validator in common.VALIDATORS:
            for param in target_params:
                for std in target_std:
                    row = f"{estimator}"
                    col = f"accuracy({validator}-s{std}-p{param})"
                    mean = group_data_1.get_group((estimator, validator, param, std))[
                        "accuracy"].mean()
                    group_data_2.loc[row, col] = mean
                    col_sum += mean
        group_data_2.loc[row, "sum"] = col_sum

    group_data_2.index.name = "index"
    # plot_data_plot(group_data_2)
    plot_data_bar(group_data_2.sort_values(by="sum", ascending=False))


if __name__ == "__main__":
    abstract_data(aggregate_file_name_3)

    print("Done.")
