"""
分類データの結果を検証２
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


columns = ["pattern", "label",
           "precision-0", "precision-1", "precision-2", "precision_sum",
           "recall-0", "recall-1", "recall-2", "recall_sum"]


def agguregate_files(idx):
    """ファイルを集約します。"""
    data_summary = pd.DataFrame(columns=columns)
    for estimator in common.ESTIMATORS:
        for validator in common.VALIDATORS:
            file_name = f"analyze_#{(idx):02}_{estimator}_{validator}_2.csv"
            file_path = common.make_path(
                common.DATA_PATH, common.ANALYZE_SUB_PATH, file_name)

            if os.path.isfile(file_path):
                print(file_name)
                data = pd.read_csv(file_path, index_col=0)
                data["pattern"] = idx
                data["label"] = data["estimator"].head(
                    1) + ":" + data["validator"].head(1)
                data00 = data.loc[1, "0"].astype(int)
                data01 = data.loc[2, "0"].astype(int)
                data02 = data.loc[3, "0"].astype(int)
                data10 = data.loc[1, "1"].astype(int)
                data11 = data.loc[2, "1"].astype(int)
                data12 = data.loc[3, "1"].astype(int)
                data20 = data.loc[1, "2"].astype(int)
                data21 = data.loc[2, "2"].astype(int)
                data22 = data.loc[3, "2"].astype(int)
                precision0 = data00 + data01 + data02
                precision1 = data10 + data11 + data12
                precision2 = data20 + data21 + data22
                recall0 = data00 + data10 + data20
                recall1 = data01 + data11 + data21
                recall2 = data02 + data12 + data22
                data["precision-0"] = \
                    data00 / precision0 if precision0 > 0 else 0
                data["precision-1"] = \
                    data11 / precision1 if precision1 > 0 else 0
                data["precision-2"] = \
                    data22 / precision2 if precision2 > 0 else 0
                data["precision_sum"] = data["precision-0"] + \
                    data["precision-1"] + data["precision-2"]
                data["recall-0"] = \
                    data00 / recall0 if recall0 > 0 else 0
                data["recall-1"] = \
                    data11 / recall1 if recall1 > 0 else 0
                data["recall-2"] = \
                    data22 / recall2 if recall2 > 0 else 0
                data["recall_sum"] = data["recall-0"] + \
                    data["recall-1"] + data["recall-2"]
                data_summary = data_summary.append(data[columns].head(1))

    data_summary.index = np.arange(1, data_summary.index.size + 1)
    data_summary.index.name = "index"
    save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, f"report_#{idx:02}_2.csv")
    data_summary.to_csv(save_file)

    return data_summary


def plot(idx, data_frame):
    """データをプロットします。"""
    fig, axes = plt.subplots(2, 1, figsize=[common.PIX_1600, common.PIX_800])

    df0 = data_frame.sort_values(
        by="precision_sum", ascending=False).reset_index(drop=True)

    barP0 = axes[0].bar(df0.index.values, df0["precision-0"].values,
                        label="precision-0", lw="3", color='k', alpha=0.6)
    axes[0].bar_label(barP0, label_type="center", fmt="%.3f")
    value0 = df0["precision-0"].values

    barP1 = axes[0].bar(df0.index.values, df0["precision-1"].values,
                        bottom=value0,
                        label="precision-1", lw="3", color='g')
    axes[0].bar_label(barP1, label_type="center", fmt="%.3f")
    value0 += df0["precision-1"].values

    barP2 = axes[0].bar(df0.index.values, df0["precision-2"].values,
                        bottom=value0,
                        label="precision-2", lw="3", color='r')
    axes[0].bar_label(barP2, label_type="center", fmt="%.3f")

    axes[0].set_xticks(df0.index.values)
    axes[0].set_xticklabels(df0["label"].values, rotation=90)
    axes[0].set_title(f"pattern #{(idx):02} confusion matrix precision")
    axes[0].grid(axis="both")
    axes[0].legend()

    df1 = data_frame.sort_values(
        by="recall_sum", ascending=False).reset_index(drop=True)

    barC0 = axes[1].bar(df1.index.values, df1["recall-0"].values,
                        label="recall-0", lw="3", color='k', alpha=0.6)
    axes[1].bar_label(barC0, label_type="center", fmt="%.3f")
    value1 = df1["recall-0"].values

    barC1 = axes[1].bar(df1.index.values, df1["recall-1"].values,
                        bottom=value1,
                        label="recall-1", lw="3", color='g')
    axes[1].bar_label(barC1, label_type="center", fmt="%.3f")
    value1 += df1["recall-1"].values

    barC2 = axes[1].bar(df1.index.values, df1["recall-2"].values,
                        bottom=value1,
                        label="recall-2", lw="3", color='r')
    axes[1].bar_label(barC2, label_type="center", fmt="%.3f")

    axes[1].set_xticks(df1.index.values)
    axes[1].set_xticklabels(df1["label"].values, rotation=90)
    axes[1].set_title(
        f"pattern #{(idx):02} confusion matrix recall")
    axes[1].grid(axis="both")
    axes[1].legend()

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.REPORT_SUB_PATH, f"report_#{idx:02}_2.png")
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
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_2.csv")
    data_total.to_csv(save_file)


if __name__ == '__main__':
    start_action()
    print("Done.")
