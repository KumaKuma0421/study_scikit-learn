"""
02系スクリプトの共通ファイル
"""
import os

CATEGORY = 2
PATTERN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# PATTERN = [2, 3]
DATA_COUNT = 1 * 1000

# STYLE = "flat"
# STYLE = "grid"
STYLE = "offset"

STANDARD_SW = 0
PARAM_SW = 1

SUFFIX = f"{STYLE}_s{STANDARD_SW}_p{PARAM_SW}"
DATA_SUFFIX = "_予戦"
# DATA_SUFFIX = "_決戦"

# BASE_PATH = r"F:\TEST_DATA\DATA_2" + DATA_SUFFIX
# BASE_PATH = r"C:\Users\hayashida\Documents\Python\Data\DATA_1" + DATA_SUFFIX
BASE_PATH = r"C:\Work\python-3.9.8-embed-amd64\Data\DATA_2" + DATA_SUFFIX

DATA_PATH = BASE_PATH + f"\\DATA_{CATEGORY}_{DATA_COUNT}_{SUFFIX}"
GRAPH_PATH = BASE_PATH + f"\\GRAPH_{CATEGORY}_{DATA_COUNT}_{SUFFIX}"
PARAM_PATH = BASE_PATH + f"\\DATA_{CATEGORY}_{DATA_COUNT}_{STYLE}_s{STANDARD_SW}_p1"
DATA_SUB_PATH = "Data"
ANALYZE_SUB_PATH = "Analyze"
PARAM_SUB_PATH = "HyperParameter"
REPORT_SUB_PATH = "Report"

EXPLANATORY = "explanatory.csv"

# 予戦
ESTIMATORS = ["KNC", "DTC", "RFC", "ABC", "GBC", "SVM",
              "GPC", "GNB", "LDA", "QDA", "MLP", "LR", "SGDC"]
# 本戦
# ESTIMATORS = ["KNC", "DTC", "RFC", "GBC", "SVM", "MLP"]
# テスト用
# ESTIMATORS = ["GBC", "SVM"]

# 本番用
VALIDATORS = ["KFold", "StratifiedKFold", "ShuffleSplit", "TimeSeriesSplit"]
# テスト用
# VALIDATORS = ["KFold"]

PIX_1600 = 16
PIX_800 = 8

RANDOM_SEED = 123


def color_func(value, min_value, max_value):
    """matplotlibのカラー設定値を計算します。"""
    response = 'k'
    if max_value - min_value < 0.001:
        ratio = 1.0
    else:
        ratio = (value - min_value) / (max_value - min_value)

    if ratio >= 0.90:
        response = '#00EE00'
    elif ratio >= 0.80:
        response = '#00CC00'
    elif ratio >= 0.70:
        response = '#00AA00'
    elif ratio >= 0.60:
        response = '#008800'
    else:
        response = '#003300'

    return response


def make_path(directory, middle_path, file_name):
    """パスの作成とディレクトリの作成"""
    if middle_path is None:
        full_path = directory
    else:
        full_path = os.path.join(directory, middle_path)

    if not os.path.exists(full_path):
        print(f"== Now making directory {full_path}.")
        os.makedirs(full_path)

    if file_name is None:
        return full_path
    else:
        return os.path.join(full_path, file_name)


if __name__ == '__main__':
    print("This is setup script.")
