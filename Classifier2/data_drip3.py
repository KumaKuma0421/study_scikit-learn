import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


aggregate_file_name_3 = os.path.join(common.BASE_PATH, "aggregate_3.csv")


def plot_data(data):
    """データをプロットします。"""
    fig, axes = plt.subplots(1, 1, figsize=[common.PIX_1600, common.PIX_800])

    im = axes.imshow(data.T.values, cmap='hot', aspect="auto")
    axes.set_xticks(np.arange(data.index.values.size))
    axes.set_xticklabels(data.index.values, rotation=90)
    axes.set_yticks(np.arange(data.columns.values.size))
    axes.set_yticklabels(data.columns.values)
    axes.grid(axis="both")

    color_bar = fig.colorbar(im, ax=axes, aspect=20, pad=0.01)
    color_bar.set_label("accuracy")

    plt.tight_layout()
    plt.show()

    save_file = common.make_path(common.BASE_PATH, None, f"plot_data_bar_3.png")
    fig.savefig(save_file)
    plt.close()


def abstract_data(target_file):
    """Validatorによって結果は変わってくるのか？"""
    data = pd.read_csv(target_file, index_col=0)
    group_data_out = pd.DataFrame()

    for ptn in common.PATTERN:
        group_data_in = data.groupby(by=["estimator", "validator", "pattern"])
        for estimator in common.ESTIMATORS:
            for validator in common.VALIDATORS:
                for pattern in common.PATTERN:
                    row = f"{estimator}-{validator}"
                    col = f"pattern({pattern})"
                    accuracy = group_data_in.get_group((estimator, validator, pattern))[
                        "accuracy"].mean()
                    group_data_out.loc[row, col] = accuracy
        group_data_out.index.name = "index"
    plot_data(group_data_out)


if __name__ == "__main__":
    abstract_data(aggregate_file_name_3)

    print("Done.")
