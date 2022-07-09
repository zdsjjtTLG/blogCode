# -- coding: utf-8 --

"""表格操作的相关工具"""

import pandas as pd
from src.tools.matrix_tools import OpenOmx
from src.tools.matrix_tools import append_mapping
from src.tools.matrix_tools import append_matrix


# 三段表(一个value字段)转化矩阵(np.Ndarray)
def table_to_matrix(table_data=None, row_name='from', col_name='to', value_col='value'):
    """
    将三段表形式的矩阵转化为np.Ndarray格式的矩阵
    :param table_data: pd.DataFrame, 三段表
    :param row_name: str, 行索引的字段名
    :param col_name: str, 列索引的字段名
    :param value_col: str, 矩阵值的字段名
    :return: np.Ndarray

    example: ~

    ~~~Input~~~
    table_data:
    from   to  value
    --
    0      0      3
    0      1      2
    0      2      7
    1      0      1
    1      1     10
    1      2      3
    2      0      4
    2      1      3
    2      2      8

    ~~~Output~~~
    [[ 3  2  7]
    [ 1 10  3]
    [ 4  3  8]]
    """
    multi_index_series = table_data.groupby([row_name, col_name])[value_col].sum()

    unstack_data = multi_index_series.unstack()
    unstack_data.sort_index(ascending=True, axis=1, inplace=True)
    unstack_data.sort_index(ascending=True, axis=0, inplace=True)
    array_res = unstack_data.values
    return array_res


# 三段表(多个value字段)转化omx矩阵文件
def table_to_omx(table_path=None, row_name='from', col_name='to', value_col_name_list=None,
                 index_name='index', export_path=None):

    """
    将三段表(多个value字段)转化omx矩阵文件
    :param table_path: str, 表的路径
    :param row_name: str, 矩阵的行所在的列的名称
    :param col_name: str, 矩阵的列所在的列的名称
    :param value_col_name_list: list, 矩阵的值所在的列的名称列表
    :param index_name: str, 索引名称
    :param export_path: str, 输出路径
    :return:

    example: ~

    ~~~Input~~~
    table_data:
    from   to  value1   value2
    --
    0      0      3     12
    0      1      2     36
    0      2      7     1
    1      0      1     32
    1      1     10     16
    1      2      3     6
    2      0      4     7
    2      1      3     34
    2      2      8     3

    ~~~Output~~~
    在指定路径生成omx文件
    core_name: 'value1'
    [[ 3  2  7]
    [ 1 10  3]
    [ 4  3  8]]

    core_name: 'value2'
    [[ 12  36  1]
    [ 32 16  6]
    [ 7  34  3]]
    """

    # 读取三段表
    table_data = pd.read_csv(table_path)

    # 检查三段表
    check_row_col(table_data, row_name=row_name, col_name=col_name)

    # 取索引
    row_index_list = sorted(list(set(table_data[row_name].to_list())), reverse=False)
    col_index_list = sorted(list(set(table_data[col_name].to_list())), reverse=False)
    if len(col_index_list) >= len(col_index_list):
        index_list = col_index_list
    else:
        index_list = row_index_list

    # 获得value字段列表
    # 如果用户不指定value字段列表, 默认使用除开row_name、col_name以外的全部字段作为value字段列表
    if value_col_name_list is None:
        value_col_name_list = list(table_data.columns)
        value_col_name_list.remove(row_name)
        value_col_name_list.remove(col_name)

    with OpenOmx(export_path, 'w') as omx_file:
        for value_col_name in value_col_name_list:
            table_data_used = table_data[[row_name, col_name, value_col_name]].copy()

            transfer_matrix = table_to_matrix(table_data=table_data_used,
                                              row_name=row_name,
                                              col_name=col_name,
                                              value_col=value_col_name)
            append_matrix(omx_object=omx_file, core_name=value_col_name, matrix=transfer_matrix)

        append_mapping(omx_object=omx_file, index_name=index_name, index_list=index_list)


# 三段表转矩阵之前对三段表的检查
def check_row_col(table_data, row_name='from', col_name='to'):

    row_name_set = set(table_data[row_name].to_list())
    col_name_set = set(table_data[col_name].to_list())

    if col_name_set == row_name_set:
        pass
    else:
        if len(col_name_set) <= len(row_name_set):
            assert (col_name_set & row_name_set) == col_name_set, '三段表的行值、列值错误!'
        else:
            assert (col_name_set & row_name_set) == row_name_set, '三段表的行值、列值错误!'


# 解析模式层级表
def parse_mode_level(mode_level_data=None, group_mode_name=None, base_mode_name=None):
    """接受前端传入的模式划分表(最多两层), 将其映射为字典
    :param mode_level_data: pd.DataFrame, 模式层级表
    :param group_mode_name: str, 表中group_mode列的列名称
    :param base_mode_name: str, 表中base_mode_name列的列名称
    :return dict, {GroupMode: BaseMode}
    """

    # 获得模式群组名称
    group_list = list(set(mode_level_data[group_mode_name].to_list()))

    # 建立字典, 用于存放GroupMode: BaseMode
    group_base_dict = {}
    for group in group_list:
        base_mode_list = mode_level_data[mode_level_data[group_mode_name] == group][base_mode_name].to_list()
        group_base_dict[group] = base_mode_list
    return group_base_dict


# 表转字典
def table_to_dict(table_data=None, key_col_name=None, value_col_name=None):
    """
    传入一个pd.DataFrame格式的表格, 指定一列作为字典的键, 指定一列作为字典的值, 返回字典
    :param table_data: pd.DataFrame
    :param key_col_name: str, 表中将要作为字典的键的列的名称
    :param value_col_name: str, 表中将要作为字典的值的列的名称
    :return: dict
    """
    key_list = table_data[key_col_name].to_list()
    value_list = table_data[value_col_name].to_list()

    res_dict = {}

    for key, value in zip(key_list, value_list):
        if key in res_dict.keys():
            pass
        else:
            res_dict[key] = value
    return res_dict


if __name__ == '__main__':

    table_to_omx(table_path=r'C:\Users\Administrator\Desktop\omx_mtx_test\test1.csv',
                 row_name='row', col_name='col', value_col_name_list=['val1', 'val2'],
                 index_name='index', export_path=r'C:\Users\Administrator\Desktop\omx_mtx_test\test1.omx')

    table_to_omx(table_path=r'C:\Users\Administrator\Desktop\omx_mtx_test\test2.csv',
                 row_name='row', col_name='col', value_col_name_list=['val1', 'val2'],
                 index_name='index', export_path=r'C:\Users\Administrator\Desktop\omx_mtx_test\test2.omx')

    table_to_omx(table_path=r'C:\Users\Administrator\Desktop\omx_mtx_test\test3.csv',
                 row_name='row', col_name='col', value_col_name_list=['val1', 'val2'],
                 index_name='index', export_path=r'C:\Users\Administrator\Desktop\omx_mtx_test\test3.omx')