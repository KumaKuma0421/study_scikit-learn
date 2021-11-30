"""
目的変数の生成
"""
import pandas as pd
import common


def make_object_variable_rule_1(param_color, param_shape):
    """
    直線で区切られたregularの中に直線で区切られたsuperiorが生成される
    """
    response = 0
    regular_slope = -0.8
    regular_intercept = 2.5
    superior_slope = -1.4
    superior_intercept = 4.7

    if param_shape > (param_color * superior_slope + superior_intercept):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    elif param_shape > (param_color * regular_slope + regular_intercept):
        # 一般的な品質の場合、１
        response = 1
    else:
        response = 0

    return response


def make_object_variable_rule_2(param_color, param_shape):
    """
    直線で区切られたregularの中に直線で区切られたsuperiorが生成される
    """
    response = 0
    invalid_slope = -0.6
    invalid_intercept = 3.9
    superior_slope = -1.4
    superior_intercept = 4.2
    regular_slope = -0.8
    regular_intercept = 2.1

    if param_shape > (param_color * invalid_slope + invalid_intercept):
        # 色相、形状ともに限界を超えた場合、０
        response = 0
    elif param_shape > (param_color * superior_slope + superior_intercept):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    elif param_shape > (param_color * regular_slope + regular_intercept):
        # 一般的な品質の場合、１
        response = 1
    else:
        response = 0

    return response


def make_object_variable_rule_3(param_color, param_shape):
    """
    右端で矩形となるregularの中に直線で区切られたsuperiorが生成される
    """
    response = 0
    slope = -1.2
    intercept = 4.7

    if param_color < 1 or param_shape < 1:
        # １項目でも品質が悪い場合、０
        return response

    if param_shape > (param_color * slope + intercept):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_4(param_color, param_shape):
    """
    中央で矩形となるregularの中に直線で区切られたsuperiorが生成される
    """
    response = 0
    slope = -1.2
    intercept = 4.5
    quarity_min = 0.8
    quarity_max = 2.8

    if param_color < quarity_min or param_shape < quarity_min:
        # １項目でも品質が悪い場合、０
        return response
    elif param_color > quarity_max or param_shape > quarity_max:
        # １項目でも品質が悪い場合、０
        return response

    if param_shape > (param_color * slope + intercept):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_5(param_color, param_shape):
    """
    矩形となるregularの中に小さな矩形でsuperiorが生成される
    """
    response = 0
    superior_color = 1.9
    superior_shape = 2.2

    if param_color < 1 or param_shape < 1:
        # １項目でも品質が悪い場合、０
        return response

    if param_color > superior_color and param_shape > superior_shape:
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_6(param_color, param_shape):
    """
    矩形となるregularの中に小さな矩形でsuperiorが生成される
    """
    response = 0
    superior_color = 1.7
    superior_shape = 1.9
    quarity_min = 0.8
    quarity_max = 2.8

    if param_color < quarity_min or param_shape < quarity_min:
        # １項目でも品質が悪い場合、０
        return response
    elif param_color > quarity_max or param_shape > quarity_max:
        # １項目でも品質が悪い場合、０
        return response

    if param_color > superior_color and param_shape > superior_shape:
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_7(param_color, param_shape):
    """
    双曲線となるregularの中に双曲線上にsuperiorが生成される
    """
    response = 0

    if param_color == 0:
        return response
    elif param_shape > (4 / param_color):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        if param_shape > (2 / param_color):
            # 一般的な品質の場合、１
            response = 1
        else:
            response = 0

    return response


def make_object_variable_rule_8(param_color, param_shape):
    """
    双曲線となるregularの中に双曲線上にsuperiorが生成される
    """
    response = 0

    if param_color == 0:
        return response
    elif param_shape > (6 / param_color):
        response = 0
    elif param_shape > (4 / param_color):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        if param_shape > (2 / param_color):
            # 一般的な品質の場合、１
            response = 1
        else:
            response = 0

    return response


def make_object_variable_rule_9(param_color, param_shape):
    """
    右端で矩形となるregularの中に放物線上のsuperiorが生成される
    """
    response = 0

    if param_color < 1 or param_shape < 1:
        # １項目でも品質が悪い場合、０
        return response

    if param_shape > ((param_color - 3) ** 2 + 1.5):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_10(param_color, param_shape):
    """
    中央で矩形となるregularの中に放物線上のsuperiorが生成される
    """
    response = 0
    quarity_min = 0.8
    quarity_max = 2.8

    if param_color < quarity_min or param_shape < quarity_min:
        # １項目でも品質が悪い場合、０
        return response
    elif param_color > quarity_max or param_shape > quarity_max:
        # １項目でも品質が悪い場合、０
        return response

    if param_shape > ((param_color - 2.8) ** 2 + 1.2):
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_11(param_color, param_shape):
    """
    右端で矩形となるregularの中に２つの放物線で囲まれたsuperiorが生成される
    """
    response = 0

    if param_color < 1 or param_shape < 1:
        # １項目でも品質が悪い場合、０
        return response

    shape_over_limit = 2.6 - 0.8 * (param_color - 2) ** 2
    shape_under_limit = 0.8 * (param_color - 2) ** 2 + 1.4
    if shape_under_limit <= param_shape and param_shape <= shape_over_limit:
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_12(param_color, param_shape):
    """
    中央で矩形となるregularの中に２つの放物線で囲まれたsuperiorが生成される
    """
    response = 0
    quarity_min = 0.8
    quarity_max = 2.8

    if param_color < quarity_min or param_shape < quarity_min:
        # １項目でも品質が悪い場合、０
        return response
    elif param_color > quarity_max or param_shape > quarity_max:
        # １項目でも品質が悪い場合、０
        return response

    shape_over_limit = 2.4 - 0.8 * (param_color - 1.8) ** 2
    shape_under_limit = 0.8 * (param_color - 1.8) ** 2 + 1.2
    if shape_under_limit <= param_shape and param_shape <= shape_over_limit:
        # 色相、形状ともに良い状態の場合、２
        response = 2
    else:
        # 一般的な品質の場合、１
        response = 1

    return response


def make_object_variable_rule_13(param_color, param_shape):
    """
    中央で円形となるregularの中に円形で囲まれたsuperiorが生成される
    """
    response = 0

    shape_over_limit1 = 2.8 - 0.8 * (param_color - 1.8) ** 2
    shape_over_limit2 = 2.4 - 1.2 * (param_color - 1.8) ** 2
    shape_under_limit1 = 0.8 * (param_color - 1.8) ** 2 + 1.2
    shape_under_limit2 = 1.2 * (param_color - 1.8) ** 2 + 1.6
    if shape_under_limit2 <= param_shape and param_shape <= shape_over_limit2:
        # 色相、形状ともに良い状態の場合、２
        response = 2
    elif shape_under_limit1 <= param_shape and param_shape <= shape_over_limit1:
        # 一般的な品質の場合、１
        response = 1
    else:
        response = 0

    return response


strategy = [
    make_object_variable_rule_1,
    make_object_variable_rule_2,
    make_object_variable_rule_3,
    make_object_variable_rule_4,
    make_object_variable_rule_5,
    make_object_variable_rule_6,
    make_object_variable_rule_7,
    make_object_variable_rule_8,
    make_object_variable_rule_9,
    make_object_variable_rule_10,
    make_object_variable_rule_11,
    make_object_variable_rule_12,
    make_object_variable_rule_13,
]


def start_action():
    """処理を開始します。"""
    file_name = common.make_path(common.DATA_PATH, None, common.EXPLANATORY)
    data = pd.read_csv(file_name, index_col=0)
    for i in common.PATTERN:
        target = []
        for color, shape in zip(data["color"].values, data["shape"].values):
            target.append(strategy[i - 1](color, shape))

        data["target"] = target
        data.index.name = "index"
        save_file = common.make_path(
            common.DATA_PATH, common.DATA_SUB_PATH, f"data_#{i:02}.csv")
        print(save_file)
        data.to_csv(save_file)


if __name__ == '__main__':
    start_action()
    print("Done.")
