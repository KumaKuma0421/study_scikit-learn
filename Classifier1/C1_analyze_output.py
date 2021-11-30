"""
分類データの結果を検証１
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common


columns = ["pattern", "label", "mean_fit_time", "std_fit_time",
           "mean_test_score", "std_test_score", "rank_test_score",
           "best_train_score", "best_test_score"]


def agguregate_files(idx):
    """ファイルを集約します。"""
    data_summary = pd.DataFrame(columns=columns)
    for estimator in common.ESTIMATORS:
        for validator in common.VALIDATORS:
            file_name = f"analyze_#{(idx):02}_{estimator}_{validator}_1.csv"
            file_path = common.make_path(
                common.DATA_PATH, common.ANALYZE_SUB_PATH, file_name)

            if os.path.isfile(file_path):
                print(file_name)
                data = pd.read_csv(file_path, index_col=0)
                data["pattern"] = idx
                data["label"] = data["estimator"] + ":" + data["validator"]
                data_one = data.sort_values(
                    by="mean_fit_time", ascending=False)
                data_summary = data_summary.append(data_one[columns].head(1))

    data_summary.index = np.arange(1, data_summary.index.size + 1)
    data_summary.index.name = "index"
    save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, f"report_#{(idx):02}_1.csv")
    data_summary.to_csv(save_file)

    return data_summary, f"report_#{(idx):02}_1"


def abstract_data(data_frame, order_by, ascending):
    """データを抽出します。"""
    data_one = data_frame.sort_values(by=order_by, ascending=ascending) \
        .reset_index(drop=True).head(20)
    x_values = np.arange(data_one.index.size)

    return data_one, data_one["label"].values, x_values


def plot_1(idx, data_frame, title, picks, labels, x_val, suffix):
    """データをプロットします。"""
    fig, axes = plt.subplots(1, 1, figsize=[common.PIX_1600, common.PIX_800])

    values = []
    min_value = 1.0
    max_value = 0.0
    for pick in picks:
        values.append(data_frame[pick].values)
        if "std_" in pick:
            pass
        else:
            min_value = min(min_value, data_frame[pick].min())
            max_value = max(max_value, data_frame[pick].max())

    color_map = [common.color_func(n, min_value, max_value) for n in values[0]]

    axes_right = axes.twinx()
    bar_obj = axes.bar(x_val, values[0], label=picks[0], color=color_map)
    axes.bar_label(bar_obj, label_type="edge", fmt="%.3f")
    axes_right.plot(x_val, values[1], label=picks[1], color='b')

    axes.set_xticks(x_val)
    axes.set_xticklabels(labels, rotation=90)
    if max_value - min_value < 0.05:
        min_value = max_value - 0.05
    axes.set_ylim(min_value, max_value)
    axes.set_title(f"{title} time report {suffix}")
    axes.legend(loc="upper left")
    axes_right.legend(loc="upper right")
    axes.grid(axis="both")
    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.REPORT_SUB_PATH,
        f"report_#{(idx):02}_1_" + suffix)
    fig.savefig(save_file)
    plt.close()


def plot_2(idx, data_frame, title, picks, labels, x_val, suffix):
    """データをプロットします。"""
    fig, axes = plt.subplots(1, 1, figsize=[common.PIX_1600, common.PIX_800])

    values = []
    min_value = 1.0
    for pick in picks:
        values.append(data_frame[pick].values)
        if "std_" in pick:
            pass
        else:
            min_value = min(min_value, data_frame[pick].min())

    width = 0.3
    offset = 0.3
    axes_right = axes.twinx()
    axes.bar(x_val - offset, values[0], label=picks[0],
             width=width, color='#00BB00')
    axes.bar(x_val, values[2], label=picks[2],
             width=width, color="#0000BB")
    axes.bar(x_val + offset, values[3], label=picks[3],
             width=width, color="#BB0000")
    axes_right.plot(x_val, values[1], label=picks[1], color='b')

    axes.set_xticks(x_val)
    axes.set_xticklabels(labels, rotation=90)
    if 1.0 - min_value < 0.05:
        min_value = 1.0 - 0.05
    axes.set_ylim(min_value, 1.0)
    axes.set_title(f"{title} score report")
    axes.legend(loc="upper left")
    axes_right.legend(loc="upper right")
    axes.grid(axis="both")
    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.REPORT_SUB_PATH,
        f"report_#{(idx):02}_1_" + suffix)
    fig.savefig(save_file)
    plt.close()


def plot(idx):
    """分析処理を開始します。"""
    data_frame, title = agguregate_files(idx)

    data1, labels1, x_value_1 = abstract_data(
        data_frame, ["mean_fit_time", "std_fit_time"], ascending=[False, False])
    plot_1(idx, data1, title, ["mean_fit_time",
           "std_fit_time"], labels1, x_value_1, "fit.png")

    data2, labels2, x_value_2 = abstract_data(
        data_frame, ["mean_test_score", "std_test_score"], ascending=[False, False])
    plot_1(idx, data2, title, ["mean_test_score",
           "std_test_score"], labels2, x_value_2, "score.png")

    data3, labels3, x_value_3 = abstract_data(
        data_frame, [
            "mean_test_score",
            "std_test_score",
            "best_train_score",
            "best_test_score"],
        ascending=[
            False,
            True,
            False,
            False])
    plot_2(idx, data3, title, [
        "mean_test_score",
        "std_test_score",
        "best_train_score",
        "best_test_score"], labels3, x_value_3, "summary.png")

    return data_frame


def start_action():
    """処理を開始します。"""
    data_total = pd.DataFrame(columns=columns)
    for i in common.PATTERN:
        plot_data = plot(i)
        data_total = data_total.append(plot_data)

    data_total.index = np.arange(1, data_total.index.size + 1)
    data_total.index.name = "index"
    save_file = common.make_path(
        common.DATA_PATH, common.REPORT_SUB_PATH, "report_1.csv")
    data_total.to_csv(save_file)


if __name__ == '__main__':
    start_action()
    print("Done.")
