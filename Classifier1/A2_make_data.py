"""
生成データの分析
"""
import pandas as pd
import matplotlib.pyplot as plt
import common

ALPHA = 0.5


def make_data_spec_1(idx, data_frame):
    """
    データ分析その１
    """
    fig, axes = plt.subplots(common.FEATURE_SIZE, 1, figsize=[
                             common.PIX_1600, common.PIX_800])

    feature_label = [f"feature{i + 1}" for i in range(common.FEATURE_SIZE)]

    target0 = data_frame[data_frame["target"] == 0]
    target1 = data_frame[data_frame["target"] == 1]

    features0 = [target0["feature1"].values, target0["feature2"].values]
    features1 = [target1["feature1"].values, target1["feature2"].values]

    axes[0].hist(features0, label=feature_label)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="both")
    axes[0].set_title("features (target=0)")

    axes[1].hist(features1, label=feature_label)
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="both")
    axes[1].set_title("features (target=1)")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}_1.png")
    fig.savefig(save_file)
    plt.close()


def make_data_spec_2(idx, data_frame):
    """
    データ分析その２
    """
    fig, axes = plt.subplots(1, 1, figsize=[common.PIX_1600, common.PIX_800])

    target0 = data_frame.query("target == 0")
    target1 = data_frame.query("target == 1")
    feature01 = target0["feature1"].values
    feature02 = target0["feature2"].values
    feature11 = target1["feature1"].values
    feature12 = target1["feature2"].values

    width = 4
    axes.scatter(feature01, feature02, label="target=0",
                 c='b', marker='o', linewidth=width, alpha=ALPHA)
    axes.scatter(feature11, feature12, label="target=1",
                 c='r', marker='*', linewidth=width, alpha=ALPHA)

    axes.grid(axis="both")
    axes.legend()
    axes.set_xlabel("feature1")
    axes.set_ylabel("feature2")
    axes.set_title("features1/features2")

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}_2.png")
    fig.savefig(save_file)
    plt.close()


def make_data_spec_3(idx, data_frame):
    """
    データ分析その３
    """
    fig, axes = plt.subplots(common.FEATURE_SIZE, 1, figsize=[
                             common.PIX_1600, common.PIX_800])

    tags = [f"feature{n + 1}" for n in range(common.FEATURE_SIZE)]
    width = 4
    for i, tag in enumerate(tags):
        axes[i].scatter(data_frame[tag].values, data_frame["target"].values,
                        marker="*", linewidth=width, alpha=ALPHA)
        axes[i].set_ylabel("target")
        axes[i].grid(axis="both")
        axes[i].set_title(tag)

    fig.tight_layout()
    # fig.show()
    save_file = common.make_path(
        common.GRAPH_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}_3.png")
    fig.savefig(save_file)
    plt.close()


def start_action():
    """処理を開始します。"""
    common.make_path(common.GRAPH_PATH, None, None)

    for i in common.PATTERN:
        file_name = common.make_path(
            common.DATA_PATH, common.DATA_SUB_PATH, f"data_#{(i):02}.csv")
        print(file_name)

        data = pd.read_csv(file_name, index_col=0)
        make_data_spec_1(i, data)
        make_data_spec_2(i, data)
        make_data_spec_3(i, data)


if __name__ == '__main__':
    start_action()
    print("Done.")
