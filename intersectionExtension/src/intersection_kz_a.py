# -- coding: utf-8 --
# @Time    : 2021/7/13 0013 17:29
# @Author  : TangKai
# @File    : intersection_kz_a.py

"""不考虑掉头的路口扩展法"""

import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from src.tools.net_modification import split_link_in_node
import matplotlib.pyplot as plt


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


# 扩展交叉口结点主函数
# 只考虑link_id, dir, from_node, to_node, length, geometry
def extend_intersection(intersection_node_id=None, link_gdf=None, node_gdf=None):
    """扩展交叉口结点, 直接在线层数据和点层数据上修改
    :param intersection_node_id: int, 交叉口结点id
    :param link_gdf: gpd.GeoDataFrame, 线层数据
    :param node_gdf: gpd.GeoDataFrame, 点层数据
    :return:
    """
    # 交叉口节点的索引
    inter_node_index = list(node_gdf[node_gdf[node_id_field] == intersection_node_id].index)[0]

    # 得到和交叉口节点连接的局部路网
    sub_link_gdf = get_sub_shp(link_shp=link_gdf, intersection_node=intersection_node_id,
                               from_node_field=from_node_id_field, to_node_field=to_node_id_field)

    # 使用局部的网络图建立有向图
    sub_graph = get_graph_from_link_gdf(link_gdf=sub_link_gdf, dir_col=direction_field,
                                        from_node_col=from_node_id_field, to_node_col=to_node_id_field)

    # 找出除开交叉口节点之外的节点
    other_node_list = list(sub_graph.nodes)
    other_node_list.remove(intersection_node_id)
    path_dict = dict()

    # 寻找原交叉口中联通的结点
    for i in range(0, len(other_node_list)):
        for j in range(0, len(other_node_list)):
            if i != j:
                if nx.has_path(sub_graph, other_node_list[i], other_node_list[j]):
                    if other_node_list[i] in path_dict.keys():
                        path_dict[other_node_list[i]].append(other_node_list[j])
                    else:
                        path_dict[other_node_list[i]] = [other_node_list[j]]
                else:
                    pass

    # 在交叉口节点处打断, 交叉口结点被删除, 对于N路交叉路口, 生成N个新结点
    link_gdf, node_gdf, new_node_dict = split_link_in_node(link_gdf=link_gdf, node_gdf=node_gdf,
                                                           match_node_index=inter_node_index,
                                                           node_columns_list=None,
                                                           truncate_ratio=0.7)

    # 确定新生成结点的联通性
    from_node_list = []
    to_node_list = []
    for key in path_dict.keys():
        for value in path_dict[key]:
            if key in new_node_dict.keys() and value in new_node_dict.keys():
                from_node_list.append(new_node_dict[key])
                to_node_list.append(new_node_dict[value])

    new_link_df = pd.DataFrame({'from_node': from_node_list, 'to_node': to_node_list})

    new_link_df['from_to'] = new_link_df[['from_node', 'to_node']]. \
        apply(lambda x: '_'.join(map(str, sorted([x[0], x[1]]))), axis=1)

    def merge_single(single_df):
        used_df = single_df.copy()

        used_df[direction_field] = 1

        if len(used_df) == 1:
            return used_df
        else:
            used_df[direction_field] = 0
            return used_df.iloc[[0], :]

    # 双向路段融合
    new_link_df = new_link_df.groupby('from_to').apply(merge_single)
    new_link_df.reset_index(inplace=True, drop=True)
    new_link_df.drop(columns='from_to', axis=1, inplace=True)

    node_gdf.set_index(node_id_field, inplace=True)

    new_link_df['from_point'] = new_link_df['from_node'].apply(lambda x: node_gdf.loc[x, geometry_field])
    new_link_df['to_point'] = new_link_df['to_node'].apply(lambda x: node_gdf.loc[x, geometry_field])
    new_link_df[geometry_field] = new_link_df[['from_point', 'to_point']]. \
        apply(lambda x: LineString((x[0], x[1])), axis=1)
    new_link_df[length_field] = new_link_df[geometry_field].apply(lambda x: x.length / 1000)
    new_link_df.drop(columns=['from_point', 'to_point'], inplace=True, axis=1)
    node_gdf.reset_index(inplace=True, drop=False)
    new_link_gdf = gpd.GeoDataFrame(new_link_df, geometry=geometry_field)

    max_link_id = max(link_gdf[link_id_field].to_list())
    new_link_gdf[link_id_field] = [i + max_link_id for i in range(1, len(new_link_gdf) + 1)]

    # 其他列数据
    other_link_df = pd.DataFrame()
    new_link_gdf = pd.concat([new_link_gdf, other_link_df], axis=1)

    link_gdf = link_gdf.append(new_link_gdf)
    link_gdf.reset_index(inplace=True, drop=True)

    return link_gdf, node_gdf


# 将具有方向字段的路网格式转化为单向的路网格式(没有方向字段, 仅靠from_node, to_node即可判别方向)
def get_single_net(net_data=None, cols_field_name_list=None, dir_field_name=None,
                   from_node_name=None, to_node_name=None, geo_bool=True):
    """将具有方向字段的路网格式转化为单向的路网格式(没有方向字段, 仅靠from_node, to_node即可判别方向)
    :param net_data: pd.DataFrame, 原路网数据
    :param cols_field_name_list: list, 列名称列表
    :param dir_field_name: str, 原路网数据代表方向的字段名称
    :param from_node_name: str, 原路网数据代表拓扑起始结点的字段名称
    :param to_node_name: str, 原路网数据代表拓扑终端结点的字段名称
    :param geo_bool: bool, 路网数据是否带几何列
    :return: gpd.DatFrame or pd.DatFrame
    """
    if cols_field_name_list is None:
        cols_field_name_list = list(net_data.columns)

    # 找出双向字段, 双向字段都应该以_ab或者_ba结尾
    two_way_field_list = list()
    for cols_name in cols_field_name_list:
        if cols_name.endswith('_ab') or cols_name.endswith('_ba'):
            two_way_field_list.append(cols_name[:-3])
    two_way_field_list = list(set(two_way_field_list))
    ab_field_del = [x + '_ab' for x in two_way_field_list]
    ba_field_del = [x + '_ba' for x in two_way_field_list]
    ab_rename_dict = {x: y for x, y in zip(ab_field_del, two_way_field_list)}
    ba_rename_dict = {x: y for x, y in zip(ba_field_del, two_way_field_list)}

    # 方向为拓扑反向的
    net_negs = net_data[net_data[dir_field_name] == -1].copy()
    net_negs.drop(ab_field_del, axis=1, inplace=True)
    net_negs.rename(columns=ba_rename_dict, inplace=True)
    net_negs['temp'] = net_negs[from_node_name]
    net_negs[from_node_name] = net_negs[to_node_name]
    net_negs[to_node_name] = net_negs['temp']
    if geo_bool:
        net_negs[geometry_field] = net_negs[geometry_field].apply(lambda x: LineString(list(x.coords)[::-1]))
    net_negs.drop(['temp', dir_field_name], inplace=True, axis=1)

    # 方向为拓扑正向的
    net_poss = net_data[net_data[dir_field_name] == 1].copy()
    net_poss.drop(ba_field_del, axis=1, inplace=True)
    net_poss.rename(columns=ab_rename_dict, inplace=True)
    net_poss.drop([dir_field_name], inplace=True, axis=1)

    # 方向为拓扑双向的, 改为拓扑正向
    net_zero_poss = net_data[net_data[dir_field_name] == 0].copy()
    net_zero_poss[dir_field_name] = 1
    net_zero_poss.drop(ba_field_del, axis=1, inplace=True)
    net_zero_poss.rename(columns=ab_rename_dict, inplace=True)
    net_zero_poss.drop([dir_field_name], inplace=True, axis=1)

    # 方向为拓扑双向的, 改为拓扑反向
    net_zero_negs = net_data[net_data[dir_field_name] == 0].copy()
    net_zero_negs.drop(ab_field_del, axis=1, inplace=True)
    net_zero_negs.rename(columns=ba_rename_dict, inplace=True)
    net_zero_negs['temp'] = net_zero_negs[from_node_name]
    net_zero_negs[from_node_name] = net_zero_negs[to_node_name]
    net_zero_negs[to_node_name] = net_zero_negs['temp']
    if geo_bool:
        net_zero_negs[geometry_field] = net_zero_negs[geometry_field].apply(lambda x: LineString(list(x.coords)[::-1]))
    net_zero_negs.drop(['temp', dir_field_name], inplace=True, axis=1)

    net = net_poss.append(net_zero_poss, ignore_index=True)
    net = net.append(net_negs, ignore_index=True)
    net = net.append(net_zero_negs, ignore_index=True)

    return net


# 根据传入的交叉口节点id, 抽取出交叉口部分的shp数据
def get_sub_shp(link_shp=None, intersection_node=None, from_node_field=None, to_node_field=None):
    """
    根据传入的交叉口节点id, 抽取出交叉口部分的shp数据
    :param link_shp: gpd.DataFrame, 线层数据
    :param intersection_node: int, 交叉口的节点id
    :param from_node_field: str, 线层数据中代表link起始节点的字段名称
    :param to_node_field: str, 线层数据中代表link终到节点的字段名称
    :return: gpd.DataFrame, 交叉口局部的路网shp
    """

    # 取交叉口附近的数据
    sub_link_shp = link_shp[(link_shp[from_node_field] == intersection_node) |
                            (link_shp[to_node_field] == intersection_node)].copy()

    # 重设索引
    sub_link_shp.reset_index(inplace=True)

    return sub_link_shp


# 从线层数据中构造有向图
def get_graph_from_link_gdf(link_gdf=None, dir_col=None, from_node_col=None, to_node_col=None):
    """从线层数据中构造有向图
    :param link_gdf: gpd.DataFrame, 线层数据
    :param dir_col: str, 方向字段名称
    :param from_node_col: str, 路网线层数据中代表拓扑起始结点的字段名称
    :param to_node_col: str, 路网线层数据中代表拓扑终到结点的字段名称
    :return:
    """
    used_link_gdf = link_gdf.copy()

    def get_edge(direction, from_node, to_node):
        edge_list = list()
        if direction == 0:
            edge_list.append([from_node, to_node])
            edge_list.append([to_node, from_node])
        elif direction == -1:
            edge_list.append([to_node, from_node])
        else:
            edge_list.append([from_node, to_node])
        return edge_list

    used_link_gdf['edge_list'] = link_gdf[[dir_col, from_node_col, to_node_col]].\
        apply(lambda x: get_edge(x[0], x[1], x[2]), axis=1)

    di_edge_list = list(used_link_gdf['edge_list'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True))
    d_graph = nx.DiGraph()
    d_graph.add_edges_from(di_edge_list)

    return d_graph


def convert_to_gdf_by_wkt(df=None, wkt_cols=None, crs=None):
    """从带几何列数据的文本数据创建地理数据
    :param df: pd.DataFrame
    :param wkt_cols:
    :param crs:
    :return:
    """
    df['geometry'] = gpd.GeoSeries.from_wkt(df[wkt_cols])
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    return gdf


if __name__ == '__main__':
    node1 = Point(1, 1)
    node2 = Point(0, 1)
    node3 = Point(1, 0)
    node4 = Point(2, 1)
    node5 = Point(1, 2)
    link11 = LineString([node2, node1])
    link22 = LineString([node3, node1])
    link33 = LineString([node4, node1])
    link44 = LineString([node5, node1])

    link_geo = gpd.GeoDataFrame(pd.DataFrame({'link_id': [11, 22, 33, 44],
                                              'dir': [0, 0, 0, 0],
                                              'from_node': [2, 3, 4, 5],
                                              'to_node': [1, 1, 1, 1],
                                              'length': [link11.length, link22.length, link33.length, link44.length],
                                              'geometry': [link11, link22, link33, link44]}),
                                geometry='geometry')
    node_geo = gpd.GeoDataFrame(pd.DataFrame({'node_id': [1, 2, 3, 4, 5],
                                              'geometry': [node1, node2, node3, node4, node5]}),
                                geometry='geometry')
    print(link_geo)

    extend_link, extend_node = extend_intersection(intersection_node_id=1, link_gdf=link_geo, node_gdf=node_geo)

    print(extend_link)
    print(extend_node)

    ax = extend_link.plot()
    extend_node.plot(ax=ax)
    plt.show()





