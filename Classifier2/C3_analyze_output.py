"""
分類データの結果を検証３
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


columns = ["pattern", "label", "accuracy",
           "best_train_score", "best_test_score"]

WIDTH = 0.4


def agguregate_files(idx):
    """ファイルを集約します。"""
    data_summary = pd.DataFrame(columns=columns)
    for estimator in common.ESTIMATORS:
        for validator in common.VALIDATORS:
            file_name = f"analyze_#{(idx):02}_{estimator}_{validator}_3.csv"
            file_path = common.make_path(
                common.DATA_PATH, common.ANALYZE_SUB_PATH, file_name)

            if os.path.isfile(file_path):
                print(file_name)
                data = pd.read_csv(file_path, index_col=0)
                data["pattern"] = idx
                data["label"] = data["estimator"].head(
                    1) + ":" + data["validator"].head(1)
                data_summary = data_summary.append(data[columns].head(1))

    data_summary.index = np.arange(1, data_summary.index.size + 1)
    data_summary.index.name = "index"
    save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, f"report_#{idx:02}_3.csv")
    data_summary.to_csv(save_file)

    return data_summary


def plot(idx, data_frame):
    """データをプロットします。"""
    fig, axes = plt.subplots(2, 1, figsize=[common.PIX_1600, common.PIX_800])

    df0 = data_frame.sort_values(
        by=["accuracy", "best_test_score", "best_train_score"],
        ascending=[False, False, False]).reset_index(drop=True)
    min_accuracy = df0["accuracy"].min()
    max_accuracy = df0["accuracy"].max()
    color_map = [common.color_func(n, min_accuracy, max_accuracy)
                 for n in df0["accuracy"].values]

    bar_obj = axes[0].bar(
        df0.index.values,
        df0["accuracy"].values, label="accuracy", color=color_map)
    axes[0].bar_label(bar_obj, label_type="edge", fmt="%.3f")
    axes[0].set_xticks(df0.index.values)
    axes[0].set_xticklabels(df0["label"].values, rotation=90)
    if 1.0 - min_accuracy < 0.05:
        min_accuracy = 1.0 - 0.05
    axes[0].set_ylim(min_accuracy, 1.0)
    axes[0].set_title(f"pattern #{(idx):02} accuracy report")
    axes[0].grid(axis="both")

    df1 = data_frame.sort_values(
        by=["best_test_score", "best_train_score", "accuracy"],
        ascending=[False, False, False]).reset_index(drop=True)
    min_train_score = df0["best_train_score"].min() - 0.05
    min_test_score = df0["best_test_score"].min() - 0.05
    min_score = min(min_train_score, min_test_score)

    axes[1].bar(df1.index.values - WIDTH / 2,
                df1["best_train_score"].values, label="best_train_score", width=WIDTH)
    axes[1].bar(df1.index.values + WIDTH / 2,
                df1["best_test_score"].values, label="best_test_score", width=WIDTH)
    axes[1].set_xticks(df1.index.values)
    axes[1].set_xticklabels(df1["label"].values, rotation=90)
    if 1.0 - min_score < 0.05:
        min_score = 1.0 - 0.05
    axes[1].set_ylim(min_score, 1.0)
    axes[1].set_title(
        f"pattern #{(idx):02} best_train_score/best_test_score report")
    axes[1].grid(axis="both")
    axes[1].legend()

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.REPORT_SUB_PATH, f"report_#{idx:02}_3.png")
    fig.savefig(save_file)
    plt.close()


def start_action():
    """処理を開始します。"""
    data_total = pd.DataFrame(columns=columns)
    for i in common.PATTERN:
        agguregate_data = agguregate_files(i)
        data_total = data_total.append(agguregate_data)
        plot(i, agguregate_data)

    data_total.index = np.arange(1, data_total.index.size + 1)
    data_total.index.name = "index"
    save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_3.csv")
    data_total.to_csv(save_file)


if __name__ == '__main__':
    start_action()
    print("Done.")
