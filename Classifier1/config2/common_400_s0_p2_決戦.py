"""
01系スクリプトの共通ファイル
"""
import os

CATEGORY = 1
PATTERN = [1, 2, 3, 4, 5, 6]
# PATTERN = [2, 3]
DATA_COUNT = 4 * 100

FEATURE_SIZE = 2
TARGET_SIZE = 2

STANDARD_SW = 0
PARAM_SW = 2

SUFFIX = f"s{STANDARD_SW}_p{PARAM_SW}"
DATA_SUFFIX = "_決戦"

# BASE_PATH = r"F:\TEST_DATA\DATA_1" + DATA_SUFFIX
# BASE_PATH = r"C:\Users\hayashida\Documents\Python\Data\DATA_1" + DATA_SUFFIX
BASE_PATH = r"C:\Work\python-3.9.8-embed-amd64\Data\DATA_1" + DATA_SUFFIX

DATA_PATH  = BASE_PATH + f"\\DATA_{CATEGORY}_{DATA_COUNT}_{SUFFIX}"
ORIG_PATH  = BASE_PATH + f"\\DATA_{CATEGORY}_{DATA_COUNT}_s{STANDARD_SW}_p0"
PARAM_PATH = BASE_PATH + f"\\DATA_{CATEGORY}_{DATA_COUNT}_s{STANDARD_SW}_p1"
TEST_PATH  = BASE_PATH + f"\\DATA_{CATEGORY}_{DATA_COUNT}_s{STANDARD_SW}_p2"
GRAPH_PATH = BASE_PATH + f"\\GRAPH_{CATEGORY}_{DATA_COUNT}_{SUFFIX}"
DATA_SUB_PATH = "Data"
ANALYZE_SUB_PATH = "Analyze"
PARAM_SUB_PATH = "HyperParameter"
REPORT_SUB_PATH = "Report"

EXPLANATORY = "explanatory.csv"

# 予戦
# ESTIMATORS = ["KNC", "DTC", "RFC", "ABC", "GBC", "SVM",
#               "GPC", "GNB", "LDA", "QDA", "MLP", "LR", "SGDC"]
# 本戦
ESTIMATORS = ["KNC", "DTC", "RFC", "ABC", "GBC", "SVM", "GPC", "QDA", "MLP"]
# テスト用
# ESTIMATORS = ["GBC", "SVM"]

# 本番用
VALIDATORS = ["KFold", "StratifiedKFold", "ShuffleSplit", "TimeSeriesSplit"]
# テスト用
# VALIDATORS = ["KFold", "ShuffleSplit"]

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
