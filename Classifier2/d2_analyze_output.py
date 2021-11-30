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

    data_precision = pivot.copy(deep=True)
    data_recall = pivot.copy(deep=True)

    for idx, group in data_by_pattern:
        values = group.reset_index(drop=True)
        data_precision[f"precision-0_{idx}"] = values["precision-0"]
        data_precision[f"precision-1_{idx}"] = values["precision-1"]
        data_precision[f"precision-2_{idx}"] = values["precision-2"]
        data_recall[f"recall-0_{idx}"] = values["recall-0"]
        data_recall[f"recall-1_{idx}"] = values["recall-1"]
        data_recall[f"recall-2_{idx}"] = values["recall-2"]

    precision_columns = [
        f"precision-{n}_{m}" for n in range(3) for m in common.PATTERN]
    precision_sum = data_precision.loc[:, precision_columns].sum(axis=1)
    precision_avg = data_precision.loc[:, precision_columns].mean(axis=1)
    precision_std = data_precision.loc[:, precision_columns].std(axis=1)
    data_precision["precision_sum"] = precision_sum
    data_precision["precision_avg"] = precision_avg
    data_precision["precision_std"] = precision_std

    recall_columns = [
        f"recall-{n}_{m}" for n in range(3) for m in common.PATTERN]
    recall_sum = data_recall.loc[:, recall_columns].sum(axis=1)
    recall_avg = data_recall.loc[:, recall_columns].mean(axis=1)
    recall_std = data_recall.loc[:, recall_columns].std(axis=1)
    data_recall["recall_sum"] = recall_sum
    data_recall["recall_avg"] = recall_avg
    data_recall["recall_std"] = recall_std

    data_precision.sort_values(
        by=["precision_sum", "precision_std"],
        ascending=[False, True], inplace=True)
    data_precision.index = np.arange(1, data_precision.index.size + 1)
    data_precision.index.name = "index"

    data_recall.sort_values(
        by=["recall_sum", "recall_std"],
        ascending=[False, True], inplace=True)
    data_recall.index = np.arange(1, data_recall.index.size + 1)
    data_recall.index.name = "index"

    precision_save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_2_precision.csv")
    data_precision.to_csv(precision_save_file)

    recall_save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_2_recall.csv")
    data_recall.to_csv(recall_save_file)

    return data_precision, data_recall


def plot_data_precision(data):
    """データをプロットします。"""
    fig, axes = plt.subplots(2, 1, figsize=[common.PIX_1600, common.PIX_800])

    label = data["label"].values
    label2 = [f"#{n}" for n in common.PATTERN]
    precision_0_label = {i: f"precision-0_{i}" for i in common.PATTERN}
    precision_1_label = {i: f"precision-1_{i}" for i in common.PATTERN}
    precision_2_label = {i: f"precision-2_{i}" for i in common.PATTERN}

    ax0 = axes[0].twinx()
    for i in common.PATTERN:
        axes[0].plot(label, data.loc[:, f"precision-0_{i}"].values,
                     label=precision_0_label[i])
        axes[0].plot(label, data.loc[:, f"precision-1_{i}"].values,
                     label=precision_1_label[i])
        axes[0].plot(label, data.loc[:, f"precision-2_{i}"].values,
                     label=precision_2_label[i])

    ax0.plot(label, data["precision_std"].values, label="std", ls="--")

    axes[0].set_xticks(label)
    axes[0].set_xticklabels(label, rotation=90)
    axes[0].set_title("precision score")
    #axes[0].legend(loc="center left")
    #ax0.legend(loc="center right")
    axes[0].grid(axis="both")

    for i in np.arange(len(label)) + 1:
        target = [f"precision-0_{n}" for n in common.PATTERN]
        axes[1].plot(label2, data.loc[i, target].values, label="precision-0")

        target = [f"precision-1_{n}" for n in common.PATTERN]
        axes[1].plot(label2, data.loc[i, target].values, label="precision-1")

        target = [f"precision-2_{n}" for n in common.PATTERN]
        axes[1].plot(label2, data.loc[i, target].values, label="precision-2")

    axes[1].grid(axis="both")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.REPORT_SUB_PATH, "report_2_precision.png")
    fig.savefig(save_file)
    plt.close()


def plot_data_recall(data):
    """データをプロットします。"""
    fig, axes = plt.subplots(2, 1, figsize=[common.PIX_1600, common.PIX_800])

    label = data["label"].values
    label2 = [f"#{n}" for n in common.PATTERN]
    recall_0_label = {i: f"recall-0_{i}" for i in common.PATTERN}
    recall_1_label = {i: f"recall-1_{i}" for i in common.PATTERN}
    recall_2_label = {i: f"recall-2_{i}" for i in common.PATTERN}

    ax0 = axes[0].twinx()
    for i in common.PATTERN:
        axes[0].plot(label, data.loc[:, f"recall-0_{i}"].values,
                     label=recall_0_label[i])
        axes[0].plot(label, data.loc[:, f"recall-1_{i}"].values,
                     label=recall_1_label[i])
        axes[0].plot(label, data.loc[:, f"recall-2_{i}"].values,
                     label=recall_2_label[i])

    ax0.plot(label, data["recall_std"].values, label="std", ls="--")

    axes[0].set_xticks(label)
    axes[0].set_xticklabels(label, rotation=90)
    axes[0].set_title("recall score")
    #axes[0].legend(loc="center left")
    #ax0.legend(loc="center right")
    axes[0].grid(axis="both")

    for i in np.arange(len(label)) + 1:
        target = [f"recall-0_{n}" for n in common.PATTERN]
        axes[1].plot(label2, data.loc[i, target].values, label="recall-0")

        target = [f"recall-1_{n}" for n in common.PATTERN]
        axes[1].plot(label2, data.loc[i, target].values, label="recall-1")

        target = [f"recall-2_{n}" for n in common.PATTERN]
        axes[1].plot(label2, data.loc[i, target].values, label="recall-2")

    axes[1].grid(axis="both")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.REPORT_SUB_PATH, "report_2_recall.png")
    fig.savefig(save_file)
    plt.close()


def start_action():
    """処理を開始します。"""
    file_name = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_2.csv")

    data_precision, data_recall = analyze_data(file_name)
    plot_data_precision(data_precision)
    plot_data_recall(data_recall)


if __name__ == '__main__':
    start_action()
    print("Done.")
