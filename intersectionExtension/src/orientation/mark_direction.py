# -- coding: utf-8 --
# @Time    : 2022/3/2 0002 10:45
# @Author  : TangKai
# @Team    : SuperModel
# @File    : mark_direction.py

import numpy as np
from shapely.geometry import LineString

# 线层数据、点层数据必需字段
length_field = 'length'  # 线层的长度, km
direction_field = 'dir'  # 线层的方向, 0, 1, -1
link_id_field = 'link_id'  # 线层的id
from_node_id_field = 'from_node'  # 线层的拓扑起始结点
to_node_id_field = 'to_node'  # 线层的拓扑终到结点
node_id_field = 'node_id'  # 点层的id
geometry_field = 'geometry'  # 几何属性字段
required_link_filed_list = \
    [link_id_field, from_node_id_field, to_node_id_field, length_field, direction_field, geometry_field]
required_node_filed_list = [node_id_field, geometry_field]

north_vector = np.array([0, 1])
south_vector = np.array([0, -1])
west_vector = np.array([-1, 0])
east_vector = np.array([1, 0])
orientation_vector_dict = {'north': north_vector,
                           'south': south_vector,
                           'west': west_vector,
                           'east': east_vector}

turn_type_dict = {'south-north': {'type': 701, 'description': '南直行'},
                  'south-west': {'type': 702, 'description': '南左转'},
                  'south-east': {'type': 703, 'description': '南右转'},
                  'north-south': {'type': 701, 'description': '北直行'},
                  'north-west': {'type': 703, 'description': '北右转'},
                  'north-east': {'type': 702, 'description': '北左转'},
                  'west-east': {'type': 701, 'description': '西直行'},
                  'west-north': {'type': 702, 'description': '西左转'},
                  'west-south': {'type': 703, 'description': '西右转'},
                  'east-west': {'type': 701, 'description': '东直行'},
                  'east-north': {'type': 703, 'description': '东右转'},
                  'east-south': {'type': 702, 'description': '东左转'},
                  'west-west': {'type': 704, 'description': '西掉头'},
                  'east-east': {'type': 704, 'description': '东掉头'},
                  'north-north': {'type': 704, 'description': '北掉头'},
                  'south-south': {'type': 704, 'description': '南掉头'},
                  }


def mark_intersection_inf(origin_intersection_gdf=None, delay_link_dict=None, intersection_node=None):
    """还需要考虑 × 型的路口
    :param origin_intersection_gdf: gpd.GeoDataFrame, 原始未拓展的交叉口
    :param delay_link_dict: dict,
    :param intersection_node: int, 交叉口节点ID
    :return:
    """
    # 首先计算各条路所处的方位
    # {link_id_1: 'north', link_id_2: 'south', link_id_1: 'west', link_id_1: 'east'}
    link_orientation_dict = get_link_orientation(origin_intersection_gdf=origin_intersection_gdf,
                                                 intersection_node=intersection_node)
    # print(link_orientation_dict)
    from_to_link_id_dict = {}
    # 先基于交叉口原路网得到{from_node}_{to_node}: link_id的映射
    for _index, row in origin_intersection_gdf.iterrows():
        if row[direction_field] in [1, 0]:
            from_to_link_id_dict['_'.join([str(row[from_node_id_field]), str(row[to_node_id_field])])] = row[
                link_id_field]
        if row[direction_field] in [-1, 0]:
            from_to_link_id_dict['_'.join([str(row[to_node_id_field]), str(row[from_node_id_field])])] = row[
                link_id_field]
    # print(from_to_link_id_dict)

    # 标记拓展道的信息
    mark_inf_dict = {}
    for delay_link in delay_link_dict.keys():
        origin_from_link = delay_link_dict[delay_link].split('-')[0]
        origin_to_link = delay_link_dict[delay_link].split('-')[1]

        origin_from_orientation = link_orientation_dict[(from_to_link_id_dict[origin_from_link])]
        origin_to_orientation = link_orientation_dict[(from_to_link_id_dict[origin_to_link])]
        mark_inf_dict[delay_link] = turn_type_dict[origin_from_orientation + '-' + origin_to_orientation]
    return mark_inf_dict


def get_link_orientation(origin_intersection_gdf=None, intersection_node=None):
    """

    :param origin_intersection_gdf:
    :param intersection_node:
    :return: Dict[int]
    """

    link_orientation_dict = {}
    # 分别计算每条link(以交叉口为起点的方向)和标准向量的夹角
    for _index, row in origin_intersection_gdf.iterrows():

        # 确保linestring的方向是从交叉口节点开始的
        if row[from_node_id_field] == intersection_node:
            linestring = row[geometry_field]
        else:
            linestring = LineString(list(row[geometry_field].coords)[::-1])
        line_coords = list(linestring.coords)
        link_vector = np.array(line_coords[-1]) - np.array(line_coords[0])
        angle_dict = {_key: np.arccos(link_vector.dot(orientation_vector) / (np.linalg.norm(link_vector) * np.linalg.norm(orientation_vector)))
                      for _key, orientation_vector in orientation_vector_dict.items()}

        link_orientation_dict[row[link_id_field]] = sorted(angle_dict.items(), key=lambda kv: (kv[1], kv[0]))[0][0]

    return link_orientation_dict


