"""
分類データの結果を検証 PIVOT
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


def analyze_data(file_path):
    """データを解析します。"""
    pivot = pd.DataFrame(columns=["label"])
    data = pd.read_csv(file_path, index_col=0)
    data_by_pattern = data.groupby(by="pattern")
    label = data_by_pattern.get_group(common.PATTERN[0])["label"].values
    pivot["label"] = label

    for idx, group in data_by_pattern:
        values = group.reset_index(drop=True)
        pivot[f"accuracy_{idx}"] = values["accuracy"]

    sum_values = pivot.iloc[:, 2:].sum(axis=1)
    avg_values = pivot.iloc[:, 2:].mean(axis=1)
    std_values = pivot.iloc[:, 2:].std(axis=1)
    pivot["sum"] = sum_values
    pivot["avg"] = avg_values
    pivot["std"] = std_values
    pivot.sort_values(by=["sum", "std"], ascending=[False, True], inplace=True)
    pivot.index = np.arange(1, pivot.index.size + 1)
    pivot.index.name = "index"
    save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_3_accuracy.csv")
    pivot.to_csv(save_file)

    return pivot


def plot_data(data):
    """データをプロットします。"""
    fig, axes = plt.subplots(2, 1, figsize=[common.PIX_1600, common.PIX_800])

    label = data["label"].values
    accuracy_label = {i: f"accuracy_{i}" for i in common.PATTERN}

    ax0 = axes[0].twinx()
    for i in common.PATTERN:
        axes[0].plot(label, data.iloc[:, i].values,
                     label=accuracy_label[i])
    ax0.plot(label, data["std"].values, label="std", ls="--")

    axes[0].set_xticks(label)
    axes[0].set_xticklabels(label, rotation=90)
    axes[0].set_title("accuracy/pattern score")
    axes[0].legend(loc="center left")
    ax0.legend(loc="center right")
    axes[0].grid(axis="both")

    for i in np.arange(len(label)):
        axes[1].plot(accuracy_label.values(),
                     data.iloc[i, 1:-3].values, label=label[i])
    axes[1].grid(axis="both")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.REPORT_SUB_PATH, "report_3_accuracy.png")
    fig.savefig(save_file)
    plt.close()


def start_action():
    """処理を開始します。"""
    file_name = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_3.csv")
    data_frame = analyze_data(file_name)
    plot_data(data_frame)


if __name__ == '__main__':
    start_action()
    print("Done.")
