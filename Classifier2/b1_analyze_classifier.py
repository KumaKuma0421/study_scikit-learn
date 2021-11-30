"""
各分類機、パラメータ、交差検証エンジンを使って分類
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import \
    train_test_split, \
    GridSearchCV, KFold, StratifiedKFold, ShuffleSplit, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import \
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import \
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib # from sklearn.externals import joblib
import common


JOBS = 8
SPLIT_SIZE = 10


def get_hyper_parameter(idx, estimator_name, validator_name):
    """jsonファイルから、パラメータを取り出します。"""
    json_file = common.make_path(
        common.PARAM_PATH, common.PARAM_SUB_PATH,
        f"param_#{idx:02}_{estimator_name}_{validator_name}.json")
    with open(json_file, mode="r") as f:
        json_data = json.load(f)
    return json_data


def get_validator(name):
    """
    交差検証のオブジェクトを取得します。
    """
    response = None

    if name == "KFold":
        response = KFold(
            n_splits=SPLIT_SIZE,
            random_state=common.RANDOM_SEED, shuffle=True)
    elif name == "StratifiedKFold":
        response = StratifiedKFold(
            n_splits=SPLIT_SIZE,
            random_state=common.RANDOM_SEED, shuffle=True)
    elif name == "ShuffleSplit":
        response = ShuffleSplit(
            n_splits=SPLIT_SIZE,
            random_state=common.RANDOM_SEED)
    elif name == "TimeSeriesSplit":
        response = TimeSeriesSplit(n_splits=SPLIT_SIZE)
    else:
        pass

    return response


def get_estimator(name):
    """
    分類器のオブジェクトを取得します。
    """
    estimator = None
    params = None

    if name == "KNC":
        estimator = KNeighborsClassifier()
        params = {
            "n_neighbors": [num for num in range(1, 21)],
            "weights": ["uniform", "distance"],
            "n_jobs": [JOBS],
        }

    elif name == "DTC":
        estimator = DecisionTreeClassifier()
        params = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            # "max_depth": [num for num in range(1, 21)],
            "max_features": [None, "sqrt", "log2"],
        }

    elif name == "RFC":
        estimator = RandomForestClassifier()
        params = {
            "n_estimators": [10, 20, 50, 100, 120, 150, 180, 200, 240, 300],
            "criterion": ["gini", "entropy"],
            # "max_depth": [num for num in range(1, 21)],
            "max_features": [None, "sqrt", "log2"],
            "n_jobs": [JOBS],
        }

    elif name == "ABC":
        estimator = AdaBoostClassifier()
        params = {
            "n_estimators": [10, 20, 50, 80, 100, 120, 150, 200],
            "algorithm": ["SAMME", "SAMME.R"],
            "random_state": [common.RANDOM_SEED],
        }

    elif name == "GBC":
        estimator = GradientBoostingClassifier()
        params = {
            "loss": ["deviance", "exponential"],
            "n_estimators": [10, 20, 50, 80, 100, 120, 150, 200],
            "criterion": ["friedman_mse", "squared_error"],
            "random_state": [common.RANDOM_SEED],
            "max_features": [None, "sqrt", "log2"],
        }

    elif name == "SVM":
        estimator = SVC()
        params = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
            "random_state": [common.RANDOM_SEED],
        }

    elif name == "GPC":
        estimator = GaussianProcessClassifier()
        params = {
            "random_state": [common.RANDOM_SEED],
            "n_jobs": [JOBS],
        }

    elif name == "GNB":
        estimator = GaussianNB()
        params = {
        }

    elif name == "LDA":
        estimator = LinearDiscriminantAnalysis()
        params = {
            "solver": ["svd", "lsqr", "eigen"],
        }

    elif name == "QDA":
        estimator = QuadraticDiscriminantAnalysis()
        params = {
        }

    elif name == "MLP":
        estimator = MLPClassifier()
        params = {
            # "hidden_layer_sizes": [10, 100, 1000],
            "hidden_layer_sizes": [100],
            "solver": ["lbfgs", "sgd", "adam"],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            # "max_iter": [2000, 5000, 10000],
            "max_iter": [200],
            "early_stopping": [True],
            "random_state": [common.RANDOM_SEED],
        }

    elif name == "LR":
        estimator = LogisticRegression()
        params = {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "random_state": [common.RANDOM_SEED],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": [100, 200, 500, 1000],
            "n_jobs": [JOBS],
        }

    elif name == "SGDC":
        estimator = SGDClassifier()
        params = {
            "penalty": ["l2", "l1", "elasticnet"],
            "n_jobs": [JOBS],
            "random_state": [common.RANDOM_SEED],
        }

    else:
        estimator = None
        params = None

    if common.PARAM_SW == 1:
        return estimator, params
    else:
        return estimator, {}


def estimator_response_save(
        idx, estimator_name, validator_name, estimator, x_train):
    """エンジンが提供する解析情報を出力します。"""
    step = 0.02

    x_min = x_train[:, 0].min()
    x_max = x_train[:, 0].max()
    y_min = x_train[:, 1].min()
    y_max = x_train[:, 1].max()

    xx_val, yy_val = np.meshgrid(np.arange(x_min, x_max, step),
                                 np.arange(y_min, y_max, step))

    if hasattr(estimator, "decision_function"):
        z0_value = estimator.decision_function(
            np.c_[xx_val.ravel(), yy_val.ravel()])[:, 0].reshape(xx_val.shape)
        z1_value = estimator.decision_function(
            np.c_[xx_val.ravel(), yy_val.ravel()])[:, 1].reshape(xx_val.shape)
        z2_value = estimator.decision_function(
            np.c_[xx_val.ravel(), yy_val.ravel()])[:, 2].reshape(xx_val.shape)
    else:
        z0_value = estimator.predict_proba(
            np.c_[xx_val.ravel(), yy_val.ravel()])[:, 0].reshape(xx_val.shape)
        z1_value = estimator.predict_proba(
            np.c_[xx_val.ravel(), yy_val.ravel()])[:, 1].reshape(xx_val.shape)
        z2_value = estimator.predict_proba(
            np.c_[xx_val.ravel(), yy_val.ravel()])[:, 2].reshape(xx_val.shape)

    data = pd.DataFrame(
        columns=["row", "col", "x_value", "y_value",
                 "z0_value", "z1_value", "z2_value"])
    data["x_value"] = xx_val.ravel()
    data["y_value"] = yy_val.ravel()
    data["z1_value"] = z0_value.ravel()
    data["z2_value"] = z1_value.ravel()
    data["z3_value"] = z2_value.ravel()
    row = np.arange(xx_val.shape[0])
    rows = np.repeat(row, xx_val.shape[1])
    col = np.arange(xx_val.shape[1])
    cols = np.tile(col, xx_val.shape[0])
    data["row"] = rows
    data["col"] = cols
    save_file = common.make_path(
        common.DATA_PATH, common.ANALYZE_SUB_PATH,
        f"analyze_#{idx:02}_{estimator_name}_{validator_name}.csv")
    data.to_csv(save_file)


def analyze_core(
        idx,
        x_train, x_test, y_train, y_test,
        estimator_name, estimator, params, validator_name, validator):
    """
    データの学習、テストを行います。
    """
    print('-' * 80)
    print(
        f"====== {idx}/{estimator_name}/{str(params)}/{validator_name} ======")

    result1_summary = pd.DataFrame()
    result2_summary = pd.DataFrame()
    result3_summary = pd.DataFrame()

    if common.PARAM_SW == 3:
        load_file = common.make_path(
            common.PARAM_PATH, common.PARAM_SUB_PATH,
            f"param_#{idx:02}_{estimator_name}_{validator_name}.pkl")
        best_estimator = joblib.load(load_file)
        print(f"deserialized {load_file}.")
    else:
        search = GridSearchCV(estimator, params, cv=validator,
                            n_jobs=JOBS, verbose=0)
        #search = RandomizedSearchCV(estimator, params, cv=validator_obj, n_jobs=JOBS)
        print(f"set up {estimator_name}/{validator_name}.")
        search.fit(x_train, y_train)
        best_estimator = search.best_estimator_
    
    print(f"search={best_estimator}")

    y_pred = best_estimator.predict(x_test)
    best_train_score = best_estimator.score(x_train, y_train)
    best_test_score = best_estimator.score(x_test, y_test)

    estimator_response_save(idx, estimator_name,
                            validator_name, best_estimator, x_train)

    parameters = best_estimator.get_params()
    if common.PARAM_SW == 1:
        save_file = common.make_path(
            common.DATA_PATH, common.PARAM_SUB_PATH,
            f"param_#{idx:02}_{estimator_name}_{validator_name}.json")
        with open(save_file, mode="w") as f:
            json.dump(parameters, f, ensure_ascii=False, indent=2)
        save_file = common.make_path(
            common.DATA_PATH, common.PARAM_SUB_PATH,
            f"param_#{idx:02}_{estimator_name}_{validator_name}.pkl")
        joblib.dump(best_estimator, save_file, compress=True)

    if common.PARAM_SW != 3:
        result1 = pd.DataFrame.from_dict(search.cv_results_)
        result1["estimator"] = estimator_name
        result1["validator"] = validator_name
        result1["best_train_score"] = best_train_score
        result1["best_test_score"] = best_test_score

        result1_summary = pd.concat([result1_summary, result1])
        print_result = result1[['estimator', 'validator',
                                'params', 'mean_test_score', 'rank_test_score']]
        print(print_result)
        print("-" * 100)

    result2 = pd.DataFrame(confusion_matrix(y_test, y_pred))
    result2["estimator"] = estimator_name
    result2["validator"] = validator_name
    result2["best_params"] = str(parameters)
    result2["best_train_score"] = best_train_score
    result2["best_test_score"] = best_test_score
    result2_summary = pd.concat([result2_summary, result2])
    print(result2)
    print("-" * 100)

    result3 = pd.DataFrame.from_dict(
        classification_report(y_test, y_pred, output_dict=True))
    result3["estimator"] = estimator_name
    result3["validator"] = validator_name
    result3["best_params"] = str(parameters)
    result3["best_train_score"] = best_train_score
    result3["best_test_score"] = best_test_score
    result3_summary = pd.concat([result3_summary, result3])
    print(result3)
    print("-" * 100)

    if common.PARAM_SW != 3:
        result1_summary.set_index(
            np.arange(1, len(result1_summary.index) + 1), inplace=True)
        result1_summary.index.name = "index"
    
    result2_summary.set_index(
        np.arange(1, len(result2_summary.index) + 1), inplace=True)
    result2_summary.index.name = "index"
    
    result3_summary.set_index(
        np.arange(1, len(result3_summary.index) + 1), inplace=True)
    result3_summary.index.name = "index"

    if common.PARAM_SW != 3:
        save_file = common.make_path(
            common.DATA_PATH, common.ANALYZE_SUB_PATH,
            f"analyze_#{idx:02}_{estimator_name}_{validator_name}_1.csv")
        result1_summary.to_csv(save_file)

    save_file = common.make_path(
        common.DATA_PATH, common.ANALYZE_SUB_PATH,
        f"analyze_#{idx:02}_{estimator_name}_{validator_name}_2.csv")
    result2_summary.to_csv(save_file)

    save_file = common.make_path(
        common.DATA_PATH, common.ANALYZE_SUB_PATH,
        f"analyze_#{idx:02}_{estimator_name}_{validator_name}_3.csv")
    result3_summary.to_csv(save_file)


def analyze(idx, estimator_name, validator_name):
    """
    分析を開始します。
    """
    target_name = common.make_path(
        common.DATA_PATH, common.DATA_SUB_PATH, f"data_#{idx:02}.csv")
    print('=' * 80)
    print(target_name)

    data = pd.read_csv(target_name, index_col=0)
    data_x = data.iloc[:, 0:2].values
    if common.STANDARD_SW == 1:
        data_x = StandardScaler().fit_transform(data_x)
    data_y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.3)

    if common.PARAM_SW < 2:
        estimator, params = get_estimator(estimator_name)
    else:
        estimator, dummy = get_estimator(estimator_name)
        params = get_hyper_parameter(idx, estimator_name, validator_name)

    estimator, params = get_estimator(estimator_name)
    validator = get_validator(validator_name)
    analyze_core(idx, x_train, x_test, y_train, y_test,
                 estimator_name, estimator, params, validator_name, validator)


def start_action():
    """処理を開始します。"""
    for i in common.PATTERN:
        for estimator in common.ESTIMATORS:
            for validator in common.VALIDATORS:
                analyze(i, estimator, validator)


if __name__ == '__main__':
    start_action()
    print("Done.")
