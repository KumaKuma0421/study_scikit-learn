"""
生成データの分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common

ALPHA = 0.5


def make_data_spec_1(idx, data_frame):
    """
    データ分析その１
    """
    fig, axes = plt.subplots(2, 1, figsize=[common.PIX_1600, common.PIX_800])

    data_label = ["regular", "superior", "invalid", "all"]
    color_pattern = ['#00AA00', '#AA0000', '#AAAAAA', '#0000AA']

    superior = data_frame[data_frame["target"] % 10 == 2]
    regular = data_frame[data_frame["target"] % 10 == 1]
    invalid = data_frame[data_frame["target"] == 0]

    color_values = [
        regular["color"].values,
        superior["color"].values,
        invalid["color"].values,
        data_frame["color"].values]
    shape_values = [
        regular["shape"].values,
        superior["shape"].values,
        invalid["shape"].values,
        data_frame["shape"].values]

    axes[0].hist(color_values, label=data_label, color=color_pattern)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="both")
    axes[0].set_title("color")

    axes[1].hist(shape_values, label=data_label, color=color_pattern)
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="both")
    axes[1].set_title("shape")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}_1.png")
    fig.savefig(save_file)
    plt.close()


def make_data_spec_2(idx, data_frame):
    """
    色相と形状の散布図を作成します。
    """
    fig, axes = plt.subplots(1, 1, figsize=[common.PIX_1600, common.PIX_800])

    superior = data_frame[data_frame["target"] % 10 == 2]
    regular = data_frame[data_frame["target"] % 10 == 1]
    invalid = data_frame[data_frame["target"] == 0]

    width = 4
    axes.scatter(regular["color"].values,
                 regular["shape"].values,
                 label="regular",
                 c='g', marker="o", linewidth=width, alpha=ALPHA)
    axes.scatter(superior["color"].values,
                 superior["shape"].values,
                 label="superior",
                 c='r', marker="*", linewidth=width, alpha=ALPHA)
    axes.scatter(invalid["color"].values,
                 invalid["shape"].values,
                 label="invalid",
                 c='k', marker="X", linewidth=width, alpha=ALPHA)

    axes.grid(axis="both")
    axes.legend()
    axes.set_xlabel("color")
    axes.set_ylabel("shape")
    axes.set_title("color/shape")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}_2.png")
    fig.savefig(save_file)
    plt.close()


def make_data_spec_3(idx, data_frame):
    """
    説明変数と目的変数の散布図を作成します。
    """
    fig, axes = plt.subplots(2, 1, figsize=[common.PIX_1600, common.PIX_800])

    data0 = data_frame[data_frame["target"] == 0]
    data1 = data_frame[data_frame["target"] == 1]
    data2 = data_frame[data_frame["target"] == 2]

    width = 4
    axes[0].scatter(data0["color"].values,
                    data0["target"].values,
                    label="invalid",
                    marker="X", c='k', linewidth=width, alpha=ALPHA)
    axes[0].scatter(data1["color"].values,
                    data1["target"].values,
                    label="regular",
                    marker="o", c='g', linewidth=width, alpha=ALPHA)
    axes[0].scatter(data2["color"].values,
                    data2["target"].values,
                    label="superior",
                    marker="*", c='r', linewidth=width, alpha=ALPHA)
    axes[0].set_ylabel("target")
    axes[0].set_xticks(np.arange(start=0, stop=3.1, step=0.1))
    axes[0].grid(axis="both")
    axes[0].legend()
    axes[0].set_title("color")

    axes[1].scatter(data0["shape"].values,
                    data0["target"].values,
                    label="invalid",
                    marker="X", c='k', linewidth=width, alpha=ALPHA)
    axes[1].scatter(data1["shape"].values,
                    data1["target"].values,
                    label="regular",
                    marker="o", c='g', linewidth=width, alpha=ALPHA)
    axes[1].scatter(data2["shape"].values,
                    data2["target"].values,
                    label="superior",
                    marker="*", c='r', linewidth=width, alpha=ALPHA)
    axes[1].set_ylabel("target")
    axes[1].set_xticks(np.arange(start=0, stop=3.1, step=0.1))
    axes[1].grid(axis="both")
    axes[1].legend()
    axes[1].set_title("shape")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}_3.png")
    fig.savefig(save_file)
    plt.close()


def start_action():
    """処理を開始します。"""
    for i in common.PATTERN:
        file_name = common.make_path(
            common.DATA_PATH, common.DATA_SUB_PATH, f"data_#{i:02}.csv")
        print(file_name)

        data = pd.read_csv(file_name, index_col=0)
        make_data_spec_1(i, data)
        make_data_spec_2(i, data)
        make_data_spec_3(i, data)


if __name__ == '__main__':
    start_action()
    print("Done.")
