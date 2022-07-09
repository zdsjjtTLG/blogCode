# -- coding: utf-8 --
# @Time    : 2021/6/15 0015 11:33
# @Author  : TangKai
# @Team    : SuperModel
# @File    : net_modification.py

"""编辑路网, epsg:32650"""

import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import linemerge
from itertools import groupby
from itertools import chain
import matplotlib.pyplot as plt


# 线层数据、点层数据必需字段
length_field = 'length'  # 线层的长度, km
direction_field = 'dir'  # 线层的方向, 0, 1, -1
link_id_field = 'link_id'  # 线层的id
from_node_id_field = 'from_node'  # 线层的拓扑起始结点
to_node_id_field = 'to_node'  # 线层的拓扑终到结点
node_id_field = 'node_id'  # 点层的id
geometry_field = 'geometry'  # 几何属性字段


# 编辑路网模块: 打断线功能主函数
def split_link(point_coord=None, link_gdf=None, node_gdf=None, match_link_index=None, node_columns_list=None,
               link_columns_list=None, tolerance=None, truncate_ratio=0.2):
    """传入用户点选的坐标点, 同时传入link表和node表, 执行打断功能
    :param point_coord: tuple, 用户点选的坐标, (x, y)
    :param link_gdf: gpd.GeoDataFrame, link线层数据
    :param node_gdf: gpd.GeoDataFrame, node点层数据
    :param match_link_index: int, 用户点匹配到的link的索引, 若不传入, 会自动搜索
    :param node_columns_list: list, node表除开必须字段'node_id', 'geometry'以外的字段
    :param link_columns_list: list, link表除开必须字段'link_id', 'from_node', 'to_node','dir', 'length', 'geometry'以外的字段
    :param tolerance: float, 搜索半径
    :param truncate_ratio: float, 截断比例
    :return: 若传入了match_link_index, 则为add_line函数调用该函数, 直接在link_gdf和node_gdf上修改
    若没有传入match_link_index, 则需要先做匹配, 会返回修改后的link_gdf和node_gdf

    编号规则为:
    新增加的一个节点id为node表中最大节点id + 1
    新增加的两条线的id分别为: 被打断的线的id、link表中最大link_id + 1
    """

    # 若没有进行近邻匹配, 则先做匹配, 这个是由用户传过来的坐标
    if match_link_index is None:

        # 先看用户点能否匹配到节点
        match_node_index_list = search_in_base_gdf(base_gdf=node_gdf,
                                                   points_loc_list=[point_coord],
                                                   tolerance=tolerance, base_gdf_crs='EPSG:32650',
                                                   point_crs='EPSG:32650')
        match_node_index = match_node_index_list[0]

        # 匹配不到节点
        if match_node_index is None:

            # 看能否匹配到线
            match_link_index_list = search_in_base_gdf(base_gdf=link_gdf,
                                                       points_loc_list=[point_coord],
                                                       tolerance=tolerance, base_gdf_crs='EPSG:32650',
                                                       point_crs='EPSG:32650')
            match_link_index = match_link_index_list[0]

            # 也匹配不到线, 无法打断
            if match_link_index is None:
                return link_gdf, node_gdf

            # 匹配到了线, 则
            else:
                split_link_in_link(point_coord=point_coord, link_gdf=link_gdf, node_gdf=node_gdf,
                                   match_link_index=match_link_index,
                                   node_columns_list=node_columns_list,
                                   link_columns_list=link_columns_list)
                return link_gdf, node_gdf

        # 匹配到了节点
        else:
            link_gdf, node_gdf, new_node_inf_dict = \
                split_link_in_node(link_gdf=link_gdf, node_gdf=node_gdf,
                                   match_node_index=match_node_index,
                                   node_columns_list=node_columns_list, truncate_ratio=truncate_ratio)

            return link_gdf, node_gdf

    # 若已经进行近邻匹配, 则代表该函数由add_link函数调用, 说明用户坐标无法匹配到节点, 匹配到了线
    else:
        new_node_id, new_node_coord_list = split_link_in_link(point_coord=point_coord, link_gdf=link_gdf,
                                                              node_gdf=node_gdf, match_link_index=match_link_index,
                                                              node_columns_list=node_columns_list,
                                                              link_columns_list=link_columns_list)
        return new_node_id, new_node_coord_list


# 编辑路网模块: 删除线功能主函数
def delete_link(coord_list=None, link_gdf=None, node_gdf=None):
    """
    删除线功能(删除和点最近的link)
    :param coord_list: list, 用户的坐标序列
    :param link_gdf: gpd.GeoDataFrame, link线层数据
    :param node_gdf: gpd.GeoDataFrame, node点层数据
    :return: 编辑后的link线层数据、node点层数据
    """

    match_link_index = []
    node_list = []

    # 记录距离每个点的最近link的index, 以及link的起始节点
    for coord in coord_list:
        now_point = Point(coord)
        link_gdf['__dis__'] = link_gdf[geometry_field].apply(lambda x: x.distance(now_point))
        link_gdf.sort_values(by='__dis__', inplace=True, ascending=True)
        now_index = list(link_gdf.iloc[[0], :].index)[0]
        match_link_index.append(now_index)
        node_list.append(link_gdf.at[now_index, from_node_id_field])
        node_list.append(link_gdf.at[now_index, to_node_id_field])

    link_gdf.drop('__dis__', axis=1, inplace=True)

    # 找出度为1的节点, 删除掉
    all_node_list_from = link_gdf[from_node_id_field].to_list()
    all_node_list_to = link_gdf[to_node_id_field].to_list()
    all_node_list = all_node_list_from + all_node_list_to
    delete_node_list = [node_id for node_id in node_list if all_node_list.count(node_id) <= 1]

    # 删除节点
    node_gdf.drop(node_gdf[node_gdf[node_id_field].isin(delete_node_list)].index, inplace=True)
    node_gdf.reset_index(inplace=True, drop=True)

    # 删除link
    link_gdf.drop(index=match_link_index, inplace=True)
    link_gdf.reset_index(inplace=True, drop=True)


# 逻辑子模块: 使用线上某点(非节点)打断link, split_link子模块, 直接在输入的线层数据和点层数据中直接修改
def split_link_in_link(point_coord=None, link_gdf=None, node_gdf=None, match_link_index=None,
                       node_columns_list=None, link_columns_list=None):
    """使用EPSG:32650, 已经确定了用户的输入点会匹配到某条link上(非节点), 执行打断操作, 在输入的线层数据和点层数据中直接修改
    :param point_coord:
    :param link_gdf:
    :param node_gdf:
    :param match_link_index:
    :param node_columns_list:
    :param link_columns_list:
    :return:
    """

    # 拿出匹配到的linestring对象
    match_link = link_gdf.at[match_link_index, geometry_field]

    # 用 离用户输入点 最近的 linestring对象上的点 对linestring进行切割
    split_linestring_list = split_line_by_nearest_point(point_obj=Point(point_coord),
                                                        linestring_obj=match_link)

    # 打断点为新节点, 返回的两条打断的线是按照拓扑方向依次排列在列表里
    split_linestring_a = split_linestring_list[0]
    split_linestring_b = split_linestring_list[1]
    new_node = split_linestring_a.intersection(split_linestring_b)

    # node表新增加节点
    new_node_id = node_gdf[node_id_field].max() + 1
    new_node_item = generate_node_item(node_id=new_node_id, geo_obj=new_node,
                                       non_required_field_list=node_columns_list)
    node_gdf.loc[len(node_gdf)] = new_node_item

    # 将该条link打断, 修改link表, 原来的link被打断
    origin_from_node = link_gdf.at[match_link_index, from_node_id_field]
    origin_to_node = link_gdf.at[match_link_index, to_node_id_field]
    origin_direction = link_gdf.at[match_link_index, direction_field]

    new_link_id_a = link_gdf.at[match_link_index, link_id_field]
    new_link_id_b = link_gdf[link_id_field].max() + 1

    new_link_item_a = generate_link_item(from_node_id=origin_from_node,
                                         to_node_id=new_node_id, link_id=new_link_id_a,
                                         direction=origin_direction,
                                         non_required_field_list=link_columns_list, geo_obj=split_linestring_a,
                                         geo_type=0)
    new_link_item_b = generate_link_item(from_node_id=new_node_id,
                                         to_node_id=origin_to_node, link_id=new_link_id_b,
                                         direction=origin_direction,
                                         non_required_field_list=link_columns_list, geo_obj=split_linestring_b,
                                         geo_type=0)

    # 删除原有link
    link_gdf.drop(index=match_link_index, inplace=True)

    # 插入打断后的两条link
    link_gdf.loc[match_link_index] = new_link_item_a
    link_gdf.loc[len(link_gdf)] = new_link_item_b

    return new_node_id, list(new_node.coords)


# 逻辑子模块: 使用节点打断link, split_link子模块
def split_link_in_node(link_gdf=None, node_gdf=None, match_node_index=None, node_columns_list=None, truncate_ratio=0.2):
    """
    已经确定了使用节点来打断link, 返回修改后的线层数据和点层数据
    若该节点的度为N, 需要原位修改N条link, 删除该匹配的节点, 新增N个新节点
    :param link_gdf: gpd.GeoDataFrame, 线层数据
    :param node_gdf: gpd.GeoDataFrame, 点层数据
    :param match_node_index: int, 匹配到的节点的索引
    :param node_columns_list: list, 节点表中除开必须字段以外的字段名称
    :param truncate_ratio: float, 截断比例(截断长度 / link长度)
    :return:
    """
    new_node_inf_dict = {}

    # 找出与此节点关联的link
    # 这里要注意BUG, 1-3、dir 1, 3-1、dir 1能否处理?这种情况应该被禁掉
    match_node_id = node_gdf.at[match_node_index, node_id_field]

    # 最大的节点ID
    max_node_id = node_gdf[node_id_field].max()

    new_node_num = 1

    # 从匹配点出发的link
    start_from_match_node_index = link_gdf[from_node_id_field].isin([match_node_id])
    link_gdf_a = link_gdf[start_from_match_node_index].copy()

    # 以匹配点结束的link
    end_in_match_node_index = link_gdf[to_node_id_field].isin([match_node_id])
    link_gdf_b = link_gdf[end_in_match_node_index].copy()

    # 用来记录新节点的字典, {new_node_id: point_geo_obj}
    new_node_dict = {}

    if link_gdf_a.empty:
        pass
    else:
        # 匹配节点为拓扑开始点的link, 截断, 修改几何列
        link_gdf_a[geometry_field] = link_gdf_a[geometry_field]. \
            apply(lambda x: truncate_line(linestring=x, start_position='from', truncate_ratio=truncate_ratio))

        # 匹配节点为拓扑开始点的link, 修改长度
        link_gdf_a[length_field] = link_gdf_a[geometry_field].apply(lambda x: x.length)

        # 删除原有link
        link_gdf.drop(index=link_gdf_a.index, inplace=True)

        # 记录新节点信息
        for index in list(link_gdf_a.index):
            link_gdf_a.loc[index, from_node_id_field] = max_node_id + new_node_num
            new_node_inf_dict[link_gdf_a.loc[index, to_node_id_field]] = max_node_id + new_node_num
            new_node_dict[max_node_id + new_node_num] = Point(link_gdf_a.at[index, geometry_field].coords[0])
            new_node_num += 1

    if link_gdf_b.empty:
        pass
    else:
        # 匹配节点为拓扑结束点的link, 截断, 修改几何列
        link_gdf_b[geometry_field] = link_gdf_b[geometry_field]. \
            apply(lambda x: truncate_line(linestring=x, start_position='to', truncate_ratio=truncate_ratio))

        # 修改长度
        link_gdf_b[length_field] = link_gdf_b[geometry_field].apply(lambda x: x.length)

        # 删除原有link
        link_gdf.drop(index=link_gdf_b.index, inplace=True)

        # 记录节点信息
        for index in list(link_gdf_b.index):
            link_gdf_b.loc[index, to_node_id_field] = max_node_id + new_node_num
            new_node_inf_dict[link_gdf_b.loc[index, from_node_id_field]] = max_node_id + new_node_num
            new_node_dict[max_node_id + new_node_num] = Point(link_gdf_b.at[index, geometry_field].coords[-1])
            new_node_num += 1

    # 拼接在一起
    link_gdf = link_gdf.append(link_gdf_a)
    link_gdf = link_gdf.append(link_gdf_b)

    # 修改节点表
    # 删除匹配到的节点
    node_gdf.drop(index=match_node_index, inplace=True)

    flag = 0
    for new_node_id, node_geo in new_node_dict.items():

        # 第一个插入的节点, 在节点表里面沿用刚刚被删除的节点的索引
        if flag == 0:

            new_node_item = generate_node_item(node_id=new_node_id,
                                               non_required_field_list=node_columns_list,
                                               geo_obj=node_geo)
            node_gdf.loc[match_node_index] = new_node_item

        else:
            new_node_item = generate_node_item(node_id=new_node_id,
                                               non_required_field_list=node_columns_list,
                                               geo_obj=node_geo)
            node_gdf.loc[len(node_gdf)] = new_node_item
        flag += 1

    return link_gdf, node_gdf, new_node_inf_dict


# 逻辑子模块: 截断功能, split_link子模块
def truncate_line(linestring=None, start_position='from', truncate_ratio=0.1):
    if start_position == 'from':
        truncate_length = linestring.length * truncate_ratio
        cut_line_list = cut(linestring, truncate_length)
        return cut_line_list[1]
    else:
        truncate_length = (1 - truncate_ratio) * linestring.length
        cut_line_list = cut(linestring, truncate_length)
        return cut_line_list[0]


# 逻辑子模块: 转化gdf的crs为’EPSG:32650‘
def check_crs(gdf=None, crs=None, is_geo=1):
    """
    检查传入的gdf是否是'EPSG:32650', 不是则转化
    :param gdf:
    :param crs:
    :param is_geo:
    :return:
    """
    if crs in ['EPSG:4326', 'epsg:4326']:
        gdf = gdf.to_crs('EPSG:32650')
        return gdf
    elif crs in ['EPSG:32650', 'epsg:32650']:
        return gdf
    else:
        raise ValueError('gdf的坐标系只能为EPSG:4326或EPSG:32650!')


# 逻辑子模块: 转化单个坐标点的crs为 'EPSG:32650'
def check_single_point_crs(point_coord=None, crs=None):
    """
    转化单个坐标点的crs为 'EPSG:32650'
    :param point_coord: tuple, (lon, lat)
    :param crs: str,
    :return:
    """
    # 如果点的坐标不是32650需要转化
    if crs == 'EPSG:4326':
        point_gdf = gpd.GeoDataFrame([], geometry=[Point(point_coord)], crs='EPSG:4326')
        point_gdf = point_gdf.to_crs('EPSG:32650')
        point_coord = (point_gdf.at[0, 'geometry'].x, point_gdf.at[0, 'geometry'].y)
        return point_coord
    elif crs == 'EPSG:32650':
        return point_coord
    else:
        raise ValueError('输入点的坐标系只能为EPSG:4326或EPSG:32650!')


# 逻辑子模块: 线截断
def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        xd = line.project(Point(p))
        if xd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if xd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


# 逻辑子模块: 最近点截断
def split_line_by_nearest_point(point_obj=None, linestring_obj=None):
    """
    给定一个linestring和一个点, 捕获linestring上和该点最近的点, 并用该点对linestring进行打断
    :param point_obj:
    :param linestring_obj:
    :return:
    """

    # 计算点a到linestring_obj开始点的距离, 点a为point_obj与linestring_obj的最近点
    dis = linestring_obj.project(point_obj)

    # 打断
    return cut(linestring_obj, dis)


# 逻辑子模块
def generate_link_item(from_node_id=None, to_node_id=None, link_id=None, direction=0,
                       coord_list=None, non_required_field_list=None, geo_type=1, geo_obj=None):
    """生成一个gdf的一行, 用字典表示
    :param from_node_id: int, 起始结点id
    :param to_node_id: int, 终到结点id
    :param link_id: int, link的id
    :param direction: int, 方向, 0, 1, -1
    :param coord_list: list, 线的坐标序列, [(lon1, lat1), (lon2, lat2),... ,(lonN, latN)]
    :param non_required_field_list: list, 除开必须字段以外的非必须字段名称列表, 值将置为空
    :param geo_type: int, 生成几何对象的方法, 默认为1, 即根据坐标在函数内部创建, 为0时, 需要直接传入几何对象
    :param geo_obj: shapely.gis, 几何对象
    :return: dict, 各键是字段名称, 值是字段值
    """

    # geometry列
    if geo_type == 1:
        geometry = LineString(coord_list)
    else:
        geometry = geo_obj

    # 计算距离, 米
    _length = geometry.length

    # 新增加link的属性
    if non_required_field_list is None or non_required_field_list == []:
        data_dict = dict()
        data_dict.update({length_field: _length, direction_field: direction, link_id_field: link_id,
                          from_node_id_field: from_node_id, to_node_id_field: to_node_id, geometry_field: geometry})
    else:

        data_dict = {x: np.nan for x in non_required_field_list}
        data_dict.update({length_field: _length, direction_field: direction, link_id_field: link_id,
                          from_node_id_field: from_node_id, to_node_id_field: to_node_id, geometry_field: geometry})

    return data_dict


# 逻辑子模块
def generate_node_item(node_id=None, non_required_field_list=None, geo_obj=None):
    """生成一个link
    :param node_id: int, 起始结点id
    :param geo_obj: 几何对象
    :param non_required_field_list: list, 除开必须字段以外的非必须字段名称列表, 值将置为空
    :return:
    """

    # 新增加node的属性
    if non_required_field_list is None or non_required_field_list == []:
        data_dict = {node_id_field: node_id, geometry_field: geo_obj}
    else:
        data_dict = {x: np.nan for x in non_required_field_list}
        data_dict.update({node_id_field: node_id, geometry_field: geo_obj})

    return data_dict


# 逻辑子模块
def search_in_base_gdf(base_gdf=None, points_loc_list=None, tolerance=100.0, base_gdf_crs='EPSG:4326',
                       point_crs='EPSG:4326'):
    """传入一组坐标点, 分别以这些点为中心创建buffer, 去一个base_gdf(node或者link)里面匹配最近的几何对象, 返回匹配到的几何对象的索引
    :param base_gdf: gpd.GeoDataFrame, 一个gdf, 可以是线层也可以代表点层
    :param points_loc_list: list, 一组点的坐标, [(lng1, lat1), (lng2, lat2), (lng3, lat3)......]
    :param tolerance: float, 搜索半径
    :param base_gdf_crs: str, base_gdf的坐标系
    :param point_crs: str, 坐标点所属的坐标系
    :return: list, [match_link_index1, match_link_index2, match_link_index3, ...]
    """

    # 这两个字段在sjon操作时, 是不允许连接数据中出现的
    assert 'index_right' not in list(base_gdf.columns), 'index_right or index_left can not be in the cols of base_gdf'
    assert 'index_left' not in list(base_gdf.columns), 'index_right or index_left can not be in the cols of base_gdf'

    # 坐标系检查
    if base_gdf_crs == 'EPSG:32650':
        pass
    else:
        if base_gdf_crs == 'EPSG:4326':
            base_gdf = base_gdf.to_crs('EPSG:32650')
        else:
            raise ValueError('base_gdf的坐标系只能为EPSG:4326或EPSG:32650!')

    # 当前搜索点的坐标, 转化为点的gdf, 并且转化坐标系
    now_points_list = list(MultiPoint(points_loc_list))
    if point_crs == 'EPSG:4326':
        now_points_gdf = gpd.GeoDataFrame([], geometry=now_points_list, crs='EPSG:4326')
        now_points_gdf = now_points_gdf.to_crs('EPSG:32650')
    else:
        if point_crs == 'EPSG:32650':
            now_points_gdf = gpd.GeoDataFrame([], geometry=now_points_list, crs='EPSG:32650')
        else:
            raise ValueError('点的坐标系只能为EPSG:4326或EPSG:32650!')

    # 以当前搜索点为半径建立buffer
    now_points_buffer_gdf = gpd.GeoDataFrame([], geometry=now_points_gdf.buffer(tolerance))

    # 给base_gdf加上新的几何列, 避免连接后几何列被抹去
    base_gdf['__base_geo__'] = base_gdf['geometry']

    # 搜索点buffer_gdf和base_gdf做sjoin, 筛选出在搜索范围内的base_geo
    join_data = gpd.sjoin(now_points_buffer_gdf, base_gdf, op='intersects', how='left')

    # 删除未在邻域内的行, 并且记录被删除的行索引, 也就是点的索引
    drop_base_index = list(join_data[join_data['__base_geo__'] == None].index)
    join_data.drop(drop_base_index, inplace=True)
    join_data['index_right'] = join_data['index_right'].astype(int)

    base_gdf.drop(columns='__base_geo__', inplace=True)

    # 为空的话, 所有点的邻域内都没有base_geo
    if join_data.empty:
        return [None] * len(points_loc_list)
    else:
        join_data['__center__'] = join_data['geometry'].apply(lambda x: x.centroid)
        join_data['__dis__'] = join_data[['__base_geo__', '__center__']].apply(lambda x: x[1].distance(x[0]), axis=1)
        match_base_index_list = []

        for i in range(0, len(points_loc_list)):
            if i in drop_base_index:
                match_base_index_list.append(None)
            else:
                now_match_data = join_data.loc[[i], :].copy()
                now_match_data.sort_values(by='__dis__', inplace=True, ascending=True)
                now_match_data.reset_index(inplace=True, drop=True)
                match_base_index_list.append(now_match_data.at[0, 'index_right'])

        return match_base_index_list


def get_edge_list(df=None, from_node_field=None, to_node_field=None, weight_field_list=None):
    """
    生成边列表用于创建图
    :param df: pd.DataFrame, 路网数据
    :param from_node_field: str, 起始节点字段名称
    :param to_node_field: str, 起始节点字段名称
    :param weight_field_list: list, 代表边权重的字段列表名称
    :return:
    """

    # 起终点
    from_list = [from_node for from_node in df[from_node_field].to_list()]
    to_list = [to_node for to_node in df[to_node_field].to_list()]

    if weight_field_list is not None:
        # 这一步非常重要, 保证迭代的顺序是按照用户传入的列顺序
        weight_data = df[weight_field_list].copy()

        # 获取权重字典
        weight_list = [list(item) for item in weight_data.itertuples(index=False)]

        # 边列表
        edge_list = [[from_node, to_node, dict(zip(weight_field_list, data))]
                     for from_node, to_node, data in zip(from_list, to_list, weight_list)]
    else:
        # 边列表
        edge_list = [[from_node, to_node] for from_node, to_node in zip(from_list, to_list)]

    return edge_list


if __name__ == '__main__':
    pass
