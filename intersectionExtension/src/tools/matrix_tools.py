# -- coding: utf-8 --
# @Time    : 2021/4/2 0001 16:46
# @Author  : TangKai
# @Team    : SuperModel
# @File    : matrix_tools.py

"""矩阵操作的相关工具"""

import openmatrix as omx
import pandas as pd
import os
import json
import time
from functools import wraps


meta_index = 'index'
meta_core = 'core_name'
meta_shape = 'shape'


def time_statistics(func):
    @wraps(func)
    def inner_func(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        consume = time.time() - st
        print(f'{func.__name__} cost {consume} secs!')
        return result

    return inner_func


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


# 单个矩阵转化为三段表
def matrix_to_table(matrix_data=None, row_name='from', col_name='to', value_name='value'):
    """
    将一个np.Ndarray格式的矩阵转化为三段表形式的矩阵
    :param matrix_data: np.Ndarray
    :param row_name: str, 行索引的字段名称
    :param col_name: str, 列索引的字段名称
    :param value_name: str, 矩阵值的字段名称
    :return: pd.DataFrame

    example: ~

    ~~~Input~~~
    matrix_data:
    [[ 3  2  7]
    [ 1 10  3]
    [ 4  3  8]]

    ~~~Output~~~
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
    """
    table_unstack = pd.DataFrame(matrix_data)

    series_stack = table_unstack.stack()

    dataframe_stack = pd.DataFrame(series_stack)
    dataframe_stack.reset_index(inplace=True)
    dataframe_stack.rename(columns={'level_0': row_name, 'level_1': col_name, 0: value_name}, inplace=True)

    return dataframe_stack


# 字典转为pd.DataFrame
def dict_to_table(input_dict=None, key_name='key', value_name='value'):
    """
    将字典转化为pd.DataFrame, 键值为一列, 值为一列
    :param input_dict: dict, 输入的字典
    :param key_name: str, 键的名称, 将作为键的那一列的列名称
    :param value_name: str, 值的名称, 将作为值的那一列的列名称
    :return: pd.DataFrame
    """

    key_list = list(input_dict.keys())
    value_list = [input_dict[key] for key in key_list]

    table_data = pd.DataFrame({key_name: key_list, value_name: value_list})

    return table_data


# 打开omx矩阵文件的上下文管理器类, 防止内存泄漏
class OpenOmx(object):
    def __init__(self, file_name, method):
        self.file_obj = omx.open_file(file_name, method)

    def __enter__(self):
        return self.file_obj

    def __exit__(self, exc_type, exc_value, tb):
        self.file_obj.close()


# 判别omx文件是否为空
def is_omx_null(omx_object):
    """
    判别omx文件是否为空
    :param omx_object: omx的文件对象
    :return: boolean
    """
    if len(omx_object.list_matrices()) < 1:
        return True
    else:
        return False


# 判别omx文件里面的矩阵是否为方阵
def is_omx_square(omx_object):
    """
    判别omx文件里面的矩阵是否为方阵(一个omx文件里面的各矩阵的维度必须一致, 这是openmatrix的规定)
    :param omx_object: omx的文件对象
    :return: boolean
    """
    if is_omx_null(omx_object):
        raise ValueError('omx文件为空!')
    else:
        dimension_tuple = omx_object.shape()
        if dimension_tuple[0] == dimension_tuple[1]:
            return True
        else:
            return False


# 取得omx文件的维度
def get_omx_shape(omx_object, axis=0):
    """
    取得omx文件的维度
    :param omx_object: omx_object: omx的文件对象
    :param axis: int, 取值为0时返回行的数目, 取值为1时返回列的数目
    :return: int
    """
    assert axis in [0, 1], '函数get_omx_shape中的axis参数指定错误!'

    if is_omx_null(omx_object):
        raise ValueError('omx文件为空!')
    else:
        dimension_tuple = omx_object.shape()
        if axis == 0:
            return dimension_tuple[0]
        else:
            return dimension_tuple[1]


# 向omx文件中(矩阵容器)新增矩阵
def append_matrix(omx_object, core_name=None, matrix=None):
    """
    向omx文件对象中新增矩阵, 若有同名矩阵, 则覆盖
    :param omx_object: omx_object, omx矩阵文件对象
    :param core_name: str, 矩阵的核心名称
    :param matrix: np.Ndarray, 矩阵数据
    :return:
    """
    if core_name in omx_object.list_matrices():
        omx_object.remove_node("/data", core_name)
    omx_object[core_name] = matrix


# 为omx文件新增索引
def append_mapping(omx_object, index_name=None, index_list=None):
    """
    为omx文件新增索引, 如果该名称的索引已经存在, 则覆盖
    :param omx_object: omx_object, omx矩阵文件对象
    :param index_name: str, 索引名称
    :param index_list: list, 索引值列表, 列表值必须为int类型
    :return:
    """

    if index_name in omx_object.list_mappings():
        omx_object.delete_mapping(index_name)
        omx_object.create_mapping(index_name, index_list)
    else:
        omx_object.create_mapping(index_name, index_list)


def save_res_to_omx(res_dict=None, fldr=None, csv=True, index_list=None, index_name=None, file_name=None):
    """

    :param res_dict: dict, dict, dict[str] = np.ndarray
    :param fldr: str
    :param file_name: str
    :param index_list: list[int]
    :param index_name: str
    :param csv: bool, 是否生成文本矩阵
    :return:
    """

    meta_data_dict = dict()

    # 元数据-记录核心矩阵名称
    meta_data_dict[meta_core] = list(res_dict.keys())

    with OpenOmx(os.path.join(fldr, file_name + '.omx'), 'w') as f:
        for core in res_dict.keys():
            f[core] = res_dict[core]

            # 元数据-记录矩阵大小
            meta_data_dict[meta_shape] = res_dict[core].shape

            if csv:
                mat_df = pd.DataFrame(res_dict[core], index=index_list, columns=index_list)
                mat_df.to_csv(os.path.join(fldr, f'{file_name}-{core}.csv'),
                              encoding='utf_8_sig', index=True)
        append_mapping(f, index_name=index_name, index_list=index_list)

        # 元数据-记录矩阵索引
        meta_data_dict[meta_index] = {}
        meta_data_dict[meta_index]['main_index'] = index_list

    # 存储元数据
    with open(os.path.join(fldr, f'{file_name}.json'), 'w') as f:
        json.dump(meta_data_dict, f)



