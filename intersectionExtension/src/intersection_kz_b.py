import os
import math
import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
from src.tools.table_tools import table_to_dict
from shapely.geometry import Point, LineString
from src.orientation.mark_direction import mark_intersection_inf

'''考虑掉头的扩展法, 目前有两个亟待优化的地方: 
1.若需要扩展的两个节点是同一条link的两端, [会导致扩展记录的方位信息出错, 较先扩展的节点记录了节点在后面可能被分裂为其他节点], 好像不会出错
2.link的东南西北方位判断目前是采取每条Link和标准方位进行方位角比选, 对于叉字型路口可能出错'''

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

# 扩展路口的新增字段
_INTER_LABEL_FIELD = '_inter_label'
_INTER_ID_FIELD = '_inter_id'

# 交叉口记录表的字段
_MARK_INTER_ID_FIELD = 'intersection_id'
_MARK_ORIGIN_NODE_ID_FIELD = 'origin_node_id'
_MARK_TYPE_FIELD = 'type'
_MARK_DESCRIPTION_FIELD = 'description'


# 交叉口路网扩展主函数
def extend_intersection(link_gdf=None, intersection_df=None, node_gdf=None, out_fldr=None, check_conflict=False,
                        max_node_id=None, max_link_id=None, res_type='geojson'):
    """

    :param link_gdf:
    :param intersection_df:
    :param node_gdf:
    :param out_fldr: str,
    :param check_conflict: bool,
    :param max_node_id
    :param max_link_id
    :param res_type: str, geojson or shp or both
    :return:
    """
    # 先将坐标转化为EPSG: 32650
    link_gdf = link_gdf.to_crs('EPSG:32650')
    node_gdf = node_gdf.to_crs('EPSG:32650')

    # 检查交叉口冲突
    if check_conflict:
        check_inter_node_conflict()
    else:
        pass

    # 将node表的坐标信息转化为哈希表
    node_gdf['X'] = node_gdf['geometry'].x
    node_gdf['Y'] = node_gdf['geometry'].y

    # 将节点的坐标信息转化为坐标字典
    node_x_dict = table_to_dict(table_data=node_gdf, key_col_name=node_id_field, value_col_name='X')
    node_y_dict = table_to_dict(table_data=node_gdf, key_col_name=node_id_field, value_col_name='Y')
    if max_node_id is None or max_node_id is None:
        max_link_id = link_gdf[link_id_field].max()
        max_node_id = node_gdf[node_id_field].max()

    max_number_dict = {'max_link_id': max_link_id, 'max_node_id': max_node_id}
    orientation_df = pd.DataFrame({_MARK_INTER_ID_FIELD: [], _MARK_ORIGIN_NODE_ID_FIELD: [],
                                   _MARK_TYPE_FIELD: [], _MARK_DESCRIPTION_FIELD: [], link_id_field: [],
                                   from_node_id_field: [], to_node_id_field: []})

    # 扩展
    print('start extending intersection...')
    print('-------------------------------')
    for intersection in intersection_df['intersection_id'].unique():

        # 当前交叉口涉及到的节点
        intersection_node_list = intersection_df[intersection_df['intersection_id'] == intersection][
            node_id_field].to_list()
        print(f'extending intersection {intersection} which infers nodes {intersection_node_list}')

        for intersection_node in intersection_node_list:

            # print(link_gdf[[from_node_id_field, to_node_id_field]])
            # 待规避的BUG: 两个需要扩展的节点是同一条link两端
            # 筛选出和该节点关联的Link
            intersection_link_gdf = link_gdf[(link_gdf[from_node_id_field] == intersection_node) |
                                             (link_gdf[to_node_id_field] == intersection_node)].copy()

            # 首先做预处理, 将所有进出口道打断, 打断处距离交叉口节点距离使用alpha比率来表示
            if intersection_link_gdf.empty:
                print(intersection_node_list)
            intersection_link_gdf.reset_index(inplace=True, drop=True)

            # 返回当前路口扩展后的gdf\映射信息\方位信息
            new_intersection_link_gdf, map_inf_dict, mark_inf_df = get_extended_intersection(
                intersection_gdf=intersection_link_gdf,
                intersection_node_id=intersection_node,
                max_number_dict=max_number_dict,
                global_node_x_dict=node_x_dict,
                global_node_y_dict=node_y_dict,
                intersection_id=intersection)
            # 将原来的路口删除, 放入新的交叉口
            link_gdf = update_link(link_gdf=link_gdf,
                                   intersection_node_id=intersection_node,
                                   extended_intersection_gdf=new_intersection_link_gdf)

            # 拓展交叉口的信息表
            orientation_df = orientation_df.append(mark_inf_df)

    # 更新节点
    origin_node_list = node_gdf[node_id_field].to_list()
    new_node_list = list(set(list(node_x_dict.keys())) - set(origin_node_list))

    new_node_df = pd.DataFrame({node_id_field: new_node_list})
    new_node_df[geometry_field] = new_node_df[node_id_field].apply(lambda x: Point(node_x_dict[x], node_y_dict[x]))
    new_node_gdf = gpd.GeoDataFrame(new_node_df, geometry=geometry_field)
    node_gdf = node_gdf.append(new_node_gdf)
    node_gdf.reset_index(inplace=True, drop=True)

    # 删除原来的交叉口节点
    all_intersection_node_list = intersection_df[node_id_field].to_list()
    node_gdf.drop(index=node_gdf[node_gdf[node_id_field].isin(all_intersection_node_list)].index, axis=0, inplace=True)

    print('extending completed !!!!!!')

    # 坐标转化为84坐标
    link_gdf[length_field] = link_gdf[length_field].apply(lambda x: np.around(x, 1))
    link_gdf = link_gdf.to_crs('EPSG:4326')
    node_gdf = node_gdf.to_crs('EPSG:4326')

    link_field_list = list(link_gdf.columns)
    double_field_list = [x for x in link_field_list if (x.endswith('_ab') or x.endswith('_ba'))]
    print(double_field_list)
    rename_dict = {x: x[:-3:-1][::-1] + '_' + x[:-3] for x in double_field_list}

    if res_type == 'geojson':
        node_gdf.to_file(os.path.join(out_fldr, 'extended_node.geojson'), driver='GeoJSON', encoding='gbk')
        link_gdf.to_file(os.path.join(out_fldr, 'extended_link.geojson'), driver='GeoJSON', encoding='gbk')
    elif res_type == 'shp':
        # 所有的ab字段放在前面
        print(rename_dict)
        link_gdf.rename(columns=rename_dict, inplace=True)
        node_gdf.to_file(os.path.join(out_fldr, 'extended_node.shp'), encoding='gbk')
        link_gdf.to_file(os.path.join(out_fldr, 'extended_link.shp'), encoding='gbk')
    else:
        node_gdf.to_file(os.path.join(out_fldr, 'extended_node.geojson'), driver='GeoJSON', encoding='gbk')
        link_gdf.to_file(os.path.join(out_fldr, 'extended_link.geojson'), driver='GeoJSON', encoding='gbk')

        print(rename_dict)
        link_gdf.rename(columns=rename_dict, inplace=True)
        node_gdf.to_file(os.path.join(out_fldr, 'extended_node.shp'), encoding='gbk')
        link_gdf.to_file(os.path.join(out_fldr, 'extended_link.shp'), encoding='gbk')

    orientation_df.reset_index(inplace=True, drop=True)
    orientation_df.to_csv(os.path.join(out_fldr, 'orientation_inf.csv'), index=False, encoding='utf_8_sig')


def check_inter_node_conflict(link_gdf=None, node_gdf=None, intersection_df=None, col_inherit_dict=None):
    """
    :param link_gdf:
    :param node_gdf:
    :param intersection_df:
    :param col_inherit_dict
    :return:
    """
    # 找出连续相邻的Node

    pass


# 交叉口路网扩展子函数
def get_extended_intersection(intersection_gdf=None,
                              intersection_node_id=None,
                              max_number_dict=None,
                              global_node_x_dict=None, global_node_y_dict=None,
                              intersection_id=None):
    """传入一个关联某交叉口的路网, 返回扩展后的gpd.DataFrame
    :param intersection_gdf: gpd.GeoDataFrame, 和交叉口节点相邻的link
    :param intersection_node_id: int, 交叉口节点ID
    :param max_number_dict: dict,
    :param global_node_x_dict: dict[int], 全局节点x坐标哈希表
    :param global_node_y_dict: dict[int], 全局节点y坐标哈希表
    :param intersection_id: int, 交叉口id
    :return:
    """
    a = max_number_dict['max_link_id']
    print(f'扩展节点{intersection_node_id}前: 最大link_id:{a}')
    #  将交叉口的所有link转化为单向, 返回单向表示的gdf和双向字段列表
    single_intersection_gdf, two_way_field_list = prepare_net(net_data=intersection_gdf,
                                                              dir_field_name=direction_field,
                                                              from_node_name=from_node_id_field,
                                                              to_node_name=to_node_id_field)
    # print(single_intersection_gdf)
    # 建立交叉口附近的拓扑有向图
    ori_sub_graph = get_graph(sub_link_gdf=single_intersection_gdf,
                              node_x_loc_dict=global_node_x_dict,
                              node_y_loc_dict=global_node_y_dict)

    # 基于局部的拓扑文件, 进行交叉口的扩展
    # 得到扩展后的交叉口新增节点的坐标哈希表: Dict[int] = coords
    # 得到映射关系哈希表, 并且修改最大编号哈希表max_number_dict中的max_node_id
    new_node_x_dict, new_node_y_dict, map_inf_dict = \
        get_intersection_extend_information(origin_graph=ori_sub_graph.copy(),
                                            intersection_node_id=intersection_node_id,
                                            offset_type='min',
                                            vertical_ratio=0.06,
                                            straight_ratio=0.12,
                                            max_number_dict=max_number_dict)

    # 标记方位信息
    mark_inf_dict = mark_intersection_inf(origin_intersection_gdf=intersection_gdf,
                                          delay_link_dict=map_inf_dict['delay_link'],
                                          intersection_node=intersection_node_id)

    # 更新全局点位置
    global_node_x_dict.update(new_node_x_dict)
    global_node_y_dict.update(new_node_y_dict)

    # 依据映射信息, 生成新扩展后的交叉口gdf, 并且修改最大编号哈希表max_number_dict中的max_link_id
    extended_intersection_link_gdf = generate_intersection_link(origin_intersection_link_gdf=single_intersection_gdf,
                                                                map_inf_dict=map_inf_dict,
                                                                node_x_dict=global_node_x_dict,
                                                                node_y_dict=global_node_y_dict,
                                                                max_number_dict=max_number_dict,
                                                                two_way_field_list=two_way_field_list)
    b = max_number_dict['max_link_id']
    print(f'扩展节点{intersection_node_id}后: 最大link_id:{b}')

    # 新的link_gdf中添加标记信息
    # mark_inf_dict记录link_id的信息
    mark_inf_df = add_label(mark_inf_dict=mark_inf_dict, extended_intersection_link_gdf=extended_intersection_link_gdf,
                            intersection_node_id=intersection_node_id, intersection_id=intersection_id)

    return extended_intersection_link_gdf, map_inf_dict, mark_inf_df


def add_label(mark_inf_dict=None, extended_intersection_link_gdf=None,
              intersection_node_id=None, intersection_id=None):
    """
    为生成的交叉口线层添加标记信息, 原位修改
    完善交叉口的专项描述信息, 返回值
    :param mark_inf_dict:
    :param extended_intersection_link_gdf:
    :param intersection_node_id:
    :param intersection_id:
    :return:
    """

    # 为生成的交叉口线层添加标记信息 #
    extended_intersection_link_gdf[_INTER_ID_FIELD] = intersection_id

    def temp_func(key1=None, key2=None, val_dict=None, join_symbol='_', val_name='description'):
        try:
            return val_dict[join_symbol.join(map(str, [key1, key2]))][val_name]
        except:
            return None

    extended_intersection_link_gdf[_INTER_LABEL_FIELD] = \
        extended_intersection_link_gdf[[from_node_id_field, to_node_id_field]].apply(
            lambda x: temp_func(x[0], x[1], mark_inf_dict), axis=1)

    # 完善交叉口的专项描述信息 #
    # 重设索引方便后面取值, 记得还原
    extended_intersection_link_gdf.set_index([from_node_id_field, to_node_id_field], inplace=True)

    orientation_df = pd.DataFrame(mark_inf_dict).T.reset_index(drop=False).rename(
        columns={'index': 'from_to'})
    orientation_df[_MARK_INTER_ID_FIELD] = intersection_id
    orientation_df[_MARK_ORIGIN_NODE_ID_FIELD] = intersection_node_id
    orientation_df[from_node_id_field] = orientation_df['from_to'].apply(lambda x: int(x.split("_")[0]))
    orientation_df[to_node_id_field] = orientation_df['from_to'].apply(lambda x: int(x.split("_")[1]))
    orientation_df[link_id_field] = orientation_df[[from_node_id_field, to_node_id_field]].apply(
        lambda x: extended_intersection_link_gdf.at[(x[0], x[1]), link_id_field], axis=1)

    orientation_df.drop(columns='from_to', inplace=True, axis=1)

    extended_intersection_link_gdf.reset_index(drop=False, inplace=True)

    return orientation_df


def prepare_net(net_data=None, cols_field_name_list=None, dir_field_name=None,
                from_node_name=None, to_node_name=None, save_link_id=False):
    """将具有方向字段的路网格式转化为单向的路网格式(没有方向字段, 仅靠from_node, to_node即可判别方向)
    :param net_data: pd.DataFrame, 原路网数据
    :param cols_field_name_list: list, 列名称列表
    :param dir_field_name: str, 原路网数据代表方向的字段名称
    :param from_node_name: str, 原路网数据代表拓扑起始结点的字段名称
    :param to_node_name: str, 原路网数据代表拓扑终端结点的字段名称
    :param save_link_id: bool, 是否输出link_id
    :return: pd.DatFrame
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
    net_negs.drop(['temp', dir_field_name], inplace=True, axis=1)
    # 几何列反向
    net_negs[geometry_field] = net_negs[geometry_field].apply(lambda x: LineString(list(x.coords)[::-1]))

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
    net_zero_negs.drop(['temp', dir_field_name], inplace=True, axis=1)
    # 几何列反向
    net_zero_negs[geometry_field] = net_zero_negs[geometry_field].apply(lambda x: LineString(list(x.coords)[::-1]))

    net = net_poss.append(net_zero_poss, ignore_index=True)
    net = net.append(net_negs, ignore_index=True)
    net = net.append(net_zero_negs, ignore_index=True)
    net.reset_index(inplace=True, drop=True)

    return net, two_way_field_list


def get_intersection_extend_information(origin_graph=None, intersection_node_id=None,
                                        offset_type=None, vertical_ratio=0.07, straight_ratio=0.1,
                                        max_number_dict=None):
    """对传入的交叉口进行扩展, 返回扩展映射信息, 直接修改最大编号信息
    :param origin_graph: nx.graph, 原交叉路口关联link的图
    :param intersection_node_id: int, 交叉口节点id
    :param offset_type: str, 偏移类型
    :param vertical_ratio: float, 沿直线方向的偏移距离的比率
    :param straight_ratio: float, 沿直线垂直方向的偏移距离的比率
    :param max_number_dict: int,
    :return:

    对于子图的修改是不需要做的, 只要记录映射信息就可以了

    origin_graph:
    未扩展的交叉口拓扑图

    exit_entrance_link_dict:
    字典, 键(字符串)代表一条link(原来的进口道、出口道),
    值(字符串)也代表一条link(路口扩展后的进口道、出口道)
    新的进口道、出口道link会继承原来进口道、出口道的路段信息
    link的表达方式为{from_node}_{to_node}

    delay_link_dict:
    字典, 键(字符串)代表一条link(路口扩展后用于表示路口延误的link),
    值(字符串)代表一条link到一条link(路口扩展前表示一条link到一条link的延误)
    新的进口道、出口道的link会继承原来原来进口道、出口道的路段信息
    link的表达方式为{from_node}_{to_node}
    一条link到一条link{from_node1}_{to_node2}-{from_node2}_{to_node3}

    new_node_loc_x:
    字典, 键代表新生成的结点的node_id, 值为x坐标(经度)

    new_node_loc_y:
    字典, 键代表新生成的结点的node_id, 值为y坐标(纬度)
    """

    # 不指定偏移类型, 则使用link的长度为基准
    if offset_type is None:
        base_length = 1

    # 指定了偏移类型
    else:
        base_length = get_base_length(graph=origin_graph, offset_type=offset_type)

    # 其他路口结点
    other_node_list = list(origin_graph.nodes)
    other_node_list.remove(intersection_node_id)

    # 新节点的坐标字典
    new_node_loc_x = {}
    new_node_loc_y = {}

    # 交叉结点口的坐标
    intersection_node_x = origin_graph.nodes[intersection_node_id]['X']
    intersection_node_y = origin_graph.nodes[intersection_node_id]['Y']

    # 记录进口道的新结点对应的原始进口道起始结点
    entrance_new_node_dict = {}

    # 记录出口道的新结点对应的原始出口道终到结点
    exit_new_node_dict = {}

    # 在进口道、出口道上新增结点
    for node_id in other_node_list:

        node_x = origin_graph.nodes[node_id]['X']
        node_y = origin_graph.nodes[node_id]['Y']

        # 规定的命名方式, 新增加的结点的id为整个路网的最大结点id加1
        new_node_r = max_number_dict['max_node_id'] + 1
        new_node_l = max_number_dict['max_node_id'] + 2
        new_node_s = max_number_dict['max_node_id'] + 1

        # 如果基准长度类别没有指定, 则使用当前link的长度
        if offset_type is None:
            base_length = get_dis(node_x, node_y, intersection_node_x, intersection_node_y)

        # 如果是双向道路
        if (node_id, intersection_node_id) in origin_graph.edges and \
                (intersection_node_id, node_id) in origin_graph.edges:

            # 获取新节点坐标, 进交叉口方向右侧
            right_x, right_y = get_new_node_around_link(x_from=intersection_node_x, y_from=intersection_node_y,
                                                        x_to=node_x, y_to=node_y,
                                                        clock_direction=1,
                                                        straight_offset_length=base_length * straight_ratio,
                                                        vertical_offset_length=base_length * vertical_ratio)

            # 获取新节点坐标, 进交叉口方向左侧
            left_x, left_y = get_new_node_around_link(x_from=intersection_node_x, y_from=intersection_node_y,
                                                      x_to=node_x, y_to=node_y,
                                                      clock_direction=0,
                                                      straight_offset_length=base_length * straight_ratio,
                                                      vertical_offset_length=base_length * vertical_ratio)

            # 打断原有link, 插入新结点
            # break_line(graph=origin_graph, from_node_id=intersection_node_id, to_node_id=node_id,
            #            new_node_id=new_node_l, new_node_x=left_x, new_node_y=left_y)
            #
            # break_line(graph=origin_graph, from_node_id=node_id, to_node_id=intersection_node_id,
            #            new_node_id=new_node_r, new_node_x=right_x, new_node_y=right_y)

            # 记录新结点的坐标
            new_node_loc_x[new_node_l] = left_x
            new_node_loc_y[new_node_l] = left_y
            new_node_loc_x[new_node_r] = right_x
            new_node_loc_y[new_node_r] = right_y

            # 更新最大节点编号
            max_number_dict['max_node_id'] = new_node_l

            # 记录进口道、出口道上新的结点信息
            entrance_new_node_dict[new_node_r] = node_id
            exit_new_node_dict[new_node_l] = node_id

            print(f'{intersection_node_id}到{node_id}路段新增加节点{new_node_l}')
            print(f'{node_id}到{intersection_node_id}路段新增加节点{new_node_r}')

        # 单向进口道
        elif (node_id, intersection_node_id) in origin_graph.edges:

            # 获取新节点坐标, 原来link上
            x, y = get_new_node_around_link(x_from=intersection_node_x, y_from=intersection_node_y,
                                            x_to=node_x, y_to=node_y,
                                            clock_direction=1,
                                            straight_offset_length=base_length * straight_ratio,
                                            vertical_offset_length=0)
            # # 打断
            # break_line(graph=origin_graph, from_node_id=node_id, to_node_id=intersection_node_id,
            #            new_node_id=new_node_s, new_node_x=x, new_node_y=y)

            # 记录进口道上新的结点信息
            entrance_new_node_dict[new_node_s] = node_id

            # 更新最大节点ID
            max_number_dict['max_node_id'] = new_node_s

            # 记录新结点的坐标
            new_node_loc_x[new_node_s] = x
            new_node_loc_y[new_node_s] = y

            print(f'{node_id}到{intersection_node_id}路段新增加节点{new_node_s}')

        # 单向出口道
        elif (intersection_node_id, node_id) in origin_graph.edges:
            # 获取新节点坐标, 原来link上
            x, y = get_new_node_around_link(x_from=intersection_node_x, y_from=intersection_node_y,
                                            x_to=node_x, y_to=node_y,
                                            clock_direction=1,
                                            straight_offset_length=base_length * straight_ratio,
                                            vertical_offset_length=0)

            # # 打断
            # break_line(graph=origin_graph, from_node_id=intersection_node_id, to_node_id=node_id,
            #            new_node_id=new_node_s, new_node_x=x, new_node_y=y)

            # 记录进口道上新的结点信息
            exit_new_node_dict[new_node_s] = node_id

            # 更新最大节点ID
            max_number_dict['max_node_id'] = new_node_s

            # 记录新结点的坐标
            new_node_loc_x[new_node_s] = x
            new_node_loc_y[new_node_s] = y

            print(f'{intersection_node_id}到{node_id}路段新增加节点{new_node_s}')
        else:
            raise ValueError

    # 新旧进出口道的记录字典
    exit_entrance_link_dict = {}

    # 记录新的进口道和原来进口道的对应关系
    for new_node_id in entrance_new_node_dict.keys():
        ori_entrance_start_node = entrance_new_node_dict[new_node_id]
        ori_entrance_link = '_'.join([str(ori_entrance_start_node), str(intersection_node_id)])
        new_entrance_link = '_'.join([str(ori_entrance_start_node), str(new_node_id)])
        exit_entrance_link_dict[ori_entrance_link] = new_entrance_link

    # 记录新的出口道和原来出口道的对应关系
    for new_node_id in exit_new_node_dict.keys():
        ori_exit_end_node = exit_new_node_dict[new_node_id]
        ori_exit_link = '_'.join([str(intersection_node_id), str(ori_exit_end_node)])
        new_exit_link = '_'.join([str(new_node_id), str(ori_exit_end_node)])
        exit_entrance_link_dict[ori_exit_link] = new_exit_link

    # 增加用于表示左转、右转、直行、掉头延误的link
    delay_link_list = []

    # 记录延误信息
    delay_link_dict = {}

    for entrance_node_id in entrance_new_node_dict.keys():
        for exit_node_id in exit_new_node_dict.keys():
            delay_link_list.append((entrance_node_id, exit_node_id))

            pre_node_id = entrance_new_node_dict[entrance_node_id]
            lat_node_id = exit_new_node_dict[exit_node_id]

            new_link = '_'.join([str(entrance_node_id), str(exit_node_id)])
            origin_from_link = '_'.join([str(pre_node_id), str(intersection_node_id)])
            origin_to_link = '_'.join([str(intersection_node_id), str(lat_node_id)])

            delay_link_dict[new_link] = '-'.join([origin_from_link, origin_to_link])

    # 添加所有的新link
    origin_graph.add_edges_from(delay_link_list)
    origin_graph.remove_node(intersection_node_id)

    # 将所有的信息封装到一个哈希表
    map_inf_dict = {'origin_graph': origin_graph, 'exit_entrance_link': exit_entrance_link_dict,
                    'entrance_new_node': entrance_new_node_dict, 'exit_new_node': exit_new_node_dict,
                    'delay_link': delay_link_dict}

    return new_node_loc_x, new_node_loc_y, map_inf_dict


def get_base_length(graph=None, offset_type=None):

    assert offset_type in ['avg', 'max', 'min'], 'base_length_type指定有误!'

    link_length_list = []
    already_check_list = []

    for link in graph.edges:
        print(already_check_list)
        _from = link[0]
        _to = link[1]
        if (_from, _to) not in already_check_list and (_to, _from) not in already_check_list:
            _from_x = graph.nodes[_from]['X']
            _from_y = graph.nodes[_from]['Y']
            _to_x = graph.nodes[_to]['X']
            _to_y = graph.nodes[_to]['Y']
            link_length_list.append(get_dis(_from_x, _from_y, _to_x, _to_y))


    if offset_type == 'avg':
        return sum(link_length_list) / len(link_length_list)
    elif offset_type == 'min':
        return min(link_length_list)
    elif offset_type == 'max':
        return max(link_length_list)


def get_new_node_around_link(x_from=None, y_from=None, x_to=None, y_to=None,
                             clock_direction=None, straight_offset_length=None, vertical_offset_length=None):
    """
    在一条link上的指定位置, 沿link指定方向偏移一定距离后沿着该link的垂直方向偏移一定距离生成一个点, 返回该点的坐标
    :param x_from: float, 起始结点的x坐标
    :param y_from: float, 起始结点的y坐标
    :param x_to: float, 终到结点的x坐标
    :param y_to: float, 终到结点的y坐标
    :param vertical_offset_length: float, 垂直方向偏移的长度
    :param straight_offset_length: float, 直线方向偏移的长度
    :param clock_direction: int, 代表时针方向的数字, 0: 顺时针, 1: 逆时针, 用于确定偏移方向
    :return:
    """

    # 基准向量
    base_vector = np.array([x_to - x_from, y_to - y_from])

    # 获得link上的偏移点的坐标
    x_o, y_o = get_loc_by_ratio(x_from, y_from, x_to, y_to, offset_length=straight_offset_length)

    # 获取的偏移方向的方向向量
    vertical_vector = get_vertical_vector(base_vector=base_vector,
                                          clock_direction=clock_direction,
                                          length=vertical_offset_length)

    # 起点到偏移点的向量
    vector_a = np.array([x_o - x_from, y_o - y_from])

    # 法向量
    new_vector = vector_a + vertical_vector

    return x_from + new_vector[0], y_from + new_vector[1]


def get_dis(x1=None, y1=None, x2=None, y2=None):
    """
    计算笛卡尔坐标系下的两点距离
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_loc_by_ratio(from_x=None, from_y=None, to_x=None, to_y=None, offset_length=None):
    """
    给定一个起终点坐标, 求出线段上某点的坐标, 该点到起点的距离为offset_length
    :param from_x:
    :param from_y:
    :param to_x:
    :param to_y:
    :param offset_length:
    :return:
    """

    x_vector = np.array([1, 0])
    y_vector = np.array([0, 1])
    link_vector = np.array([to_x - from_x, to_y - from_y])

    y_para = 0
    x_para = 0

    if np.dot(x_vector, link_vector) < 0:
        x_para = -1
    elif np.dot(x_vector, link_vector) > 0:
        x_para = 1
    else:
        pass

    if np.dot(y_vector, link_vector) < 0:
        y_para = -1
    elif np.dot(y_vector, link_vector) > 0:
        y_para = 1
    else:
        y_para = 1

    if (to_x - from_x) == 0:
        xo = from_x
        yo = from_y + y_para * offset_length
    else:
        k = (to_y - from_y) / (to_x - from_x)
        theta = math.atan(k)
        xo = from_x + offset_length * abs(math.cos(theta)) * x_para
        yo = from_y + offset_length * abs(math.sin(theta)) * y_para
    return xo, yo


def get_vertical_vector(base_vector=None, clock_direction=None, length=1.0):
    """
    基于一个向量, 计算特定方向、特定长度的法向量
    :param base_vector: np.Ndarray, 二维向量
    :param clock_direction: int, 代表时针方向的数字, 0: 顺时针, 1: 逆时针
    :param length: float, 返回的结果向量的模
    :return: np.Ndarray

    example: ~

    ~~~ Input ~~~
    base_vector = [1, 0]: ---->
    clock_direction = 0, 代表选取base_vector的两个法向量中和base_vector构成顺时针方向的那个
    base_vector的法向量有两个方向, 向上的和向下的, 前者和base_vector构成逆时针方向, 后者和base_vector构成顺时针方向,

    ~~~ output ~~~
    [0, 1]
    """

    assert clock_direction in [0, 1], 'clock_direction参数指定有误!'

    if base_vector[0] == 0:
        normal_vector_a = np.array([-1, 0])
        normal_vector_b = np.array([1, 0])
        length_ratio = length
    elif base_vector[1] == 0:
        normal_vector_a = np.array([0, -1])
        normal_vector_b = np.array([0, 1])
        length_ratio = length
    else:
        k = base_vector[1] / base_vector[0]
        normal_vector_a = np.array([1, -1 / k])
        normal_vector_b = np.array([-1, 1 / k])
        length_ratio = length / math.sqrt(1 + (1 / k)**2)

    # 要求返回顺时针的法向量
    if clock_direction == 0:
        if (base_vector[0] * normal_vector_a[1] - normal_vector_a[0] * base_vector[1]) < 0:
            return normal_vector_a * length_ratio
        else:
            return normal_vector_b * length_ratio

    # 要求返回逆时针的法向量
    elif clock_direction == 1:
        if (base_vector[0] * normal_vector_a[1] - normal_vector_a[0] * base_vector[1]) < 0:
            return normal_vector_b * length_ratio
        else:
            return normal_vector_a * length_ratio


# 建立拓扑图有向图
def get_graph(sub_link_gdf=None, node_x_loc_dict=None, node_y_loc_dict=None):
    """
    建立有向图
    :param sub_link_gdf: gpd.GeoDtaFrame
    :param node_x_loc_dict: Dict[int]
    :param node_y_loc_dict: Dict[int]
    :return: nx.graph
    """
    # 交叉口附近的节点, 构成子网络
    inter_node_list = list(set(sub_link_gdf[from_node_id_field].to_list() +
                               sub_link_gdf[to_node_id_field].to_list()))

    # 基于intersection_node建立自网络
    sub_net = nx.DiGraph()

    sub_node_list = [(node_id, {'X': node_x_loc_dict[node_id], 'Y': node_y_loc_dict[node_id]})
                     for node_id in inter_node_list]

    sub_link_list = []
    for i in range(0, len(sub_link_gdf)):
        from_node = sub_link_gdf.at[i, from_node_id_field]
        to_node = sub_link_gdf.at[i, to_node_id_field]
        distance = get_dis(node_x_loc_dict[from_node], node_y_loc_dict[from_node],
                           node_x_loc_dict[to_node], node_y_loc_dict[to_node])
        sub_link_list.append((from_node, to_node, {'length': distance}))

    sub_net.add_nodes_from(sub_node_list)
    sub_net.add_edges_from(sub_link_list)

    return sub_net


# 依据新交叉口和原交叉口的映射关系, 生成新的交叉口gpd.GeoDataFrame
def generate_intersection_link(origin_intersection_link_gdf=None, map_inf_dict=None,
                               node_x_dict=None, node_y_dict=None, max_number_dict=None, two_way_field_list=None):
    """
    依据拓展路口的映射关系在原双向表示的路网中加入新的交叉口扩展路网, 删除原来的交叉口路网
    有一个col_inherit_dict参数, 指定字段的继承形式, 有copy(复制继承, 针对数值类型和str类型)和proportion(按照长度比例折减继承, 对于数值类型)两种类型
    :param origin_intersection_link_gdf: gpd.GeoDataFrame, 扩展前的交叉口路网(单向表示的)
    :param map_inf_dict: Dict, 扩展映射信息哈希表
    :param max_number_dict: Dict,
    :param node_x_dict: Dict,
    :param node_y_dict: Dict,
    :param two_way_field_list: List, 双向字段列表
    :return:
    """

    #  ------ 将映射关系从map_inf_dict中取出来 ------ #
    exit_entrance_link_dict = map_inf_dict['exit_entrance_link']
    delay_link_dict = map_inf_dict['delay_link']
    a = max_number_dict['max_link_id']
    print(f'继承前的最大link_id:{a}')
    #  ------ 继承进出口道的路网信息 ------ #
    # 所有信息都复制继承, 考虑是同一条link
    # 新的进出口道link
    new_exit_entrance_link_list = list(exit_entrance_link_dict.values())
    # 原来的进出口道link
    ori_exit_entrance_link_list = list(exit_entrance_link_dict.keys())

    # 生成新进出口道的几何信息
    exit_entrance_link_data = pd.DataFrame({'new_from_to': new_exit_entrance_link_list,
                                            'ori_from_to': ori_exit_entrance_link_list})
    exit_entrance_link_data['new_from'] = exit_entrance_link_data['new_from_to'].apply(lambda x: int(x.split('_')[0]))
    exit_entrance_link_data['new_to'] = exit_entrance_link_data['new_from_to'].apply(lambda x: int(x.split('_')[1]))
    exit_entrance_link_data['ori_from'] = exit_entrance_link_data['ori_from_to'].apply(lambda x: int(x.split('_')[0]))
    exit_entrance_link_data['ori_to'] = exit_entrance_link_data['ori_from_to'].apply(lambda x: int(x.split('_')[1]))

    # 按照新的起终点坐标转化为shp(LineString类型), 仍然保留了原来的起终点信息
    exit_entrance_link_gdf = get_line_string_data(from_to_data=exit_entrance_link_data,
                                                  node_x_loc_dict=node_x_dict, node_y_loc_dict=node_y_dict,
                                                  from_name='new_from', to_name='new_to')

    # 从原来的交叉口信息中选出非几何字段
    selected_columns = list(origin_intersection_link_gdf.columns)
    selected_columns.remove('geometry')
    selected_origin_sub_df = origin_intersection_link_gdf[selected_columns]

    # link_id不可以直接继承, 不然会有重复值, 其他属性直接继承
    new_exit_entrance_link_gdf = \
        pd.merge(exit_entrance_link_gdf, selected_origin_sub_df,
                 left_on=['ori_from', 'ori_to'], right_on=[from_node_id_field, to_node_id_field],
                 how='left')

    # 新的进出口link的shp文件, 这里可能有BUG, 用户输入的起始结点名称如果不是 'FROM' 和 'TO', 要先转化
    new_exit_entrance_link_gdf. \
        drop(['ori_from', 'ori_to', 'new_from_to', 'ori_from_to', from_node_id_field, to_node_id_field], axis=1,
             inplace=True)

    new_exit_entrance_link_gdf.rename(columns={'new_from': from_node_id_field, 'new_to': to_node_id_field},
                                      inplace=True)
    # 方向都是1
    new_exit_entrance_link_gdf[direction_field] = 1

    # 双向字段_ab有值, _ba无值为空
    new_exit_entrance_link_gdf.rename(columns={two_way_field: two_way_field + '_ab'
                                               for two_way_field in two_way_field_list}, inplace=True)
    new_exit_entrance_link_gdf[[two_way_field + '_ba' for two_way_field in two_way_field_list]] = np.nan

    #  ------ 生成左转、右转、直行专用道 ------ #
    # 新的延误link的起终点id
    delay_link_list = list(delay_link_dict.keys())

    # 建立延误link的shp文件
    # 只有from_field和to_field字段
    delay_link_data = pd.DataFrame({'delay_from_to': delay_link_list})
    delay_link_data['delay_from'] = delay_link_data['delay_from_to'].apply(lambda x: int(x.split('_')[0]))
    delay_link_data['delay_to'] = delay_link_data['delay_from_to'].apply(lambda x: int(x.split('_')[1]))
    delay_link_data.rename(columns={'delay_from': from_node_id_field, 'delay_to': to_node_id_field}, inplace=True)
    delay_link_data['origin_from_to_link'] = delay_link_data['delay_from_to'].apply(lambda x: delay_link_dict[x])
    delay_link_data.drop(['delay_from_to', 'origin_from_to_link'], axis=1, inplace=True)

    # 建立shp
    delay_link_gdf = get_line_string_data(from_to_data=delay_link_data,
                                          node_x_loc_dict=node_x_dict, node_y_loc_dict=node_y_dict,
                                          from_name=from_node_id_field, to_name=to_node_id_field)

    # 生成length
    delay_link_gdf[length_field] = delay_link_gdf[geometry_field].apply(lambda x: round(x.length, 2))

    new_link_gdf = new_exit_entrance_link_gdf.append(delay_link_gdf)
    new_link_gdf.reset_index(inplace=True, drop=True)

    # dir字段
    new_link_gdf[direction_field] = 1

    # 统一更新link_id
    new_link_gdf[link_id_field] = [max_number_dict['max_link_id'] + i for i in range(1, len(new_link_gdf) + 1)]

    # 修改最大link_id
    max_number_dict['max_link_id'] += len(new_link_gdf)
    b = max_number_dict['max_link_id']
    print(f'继承前的最大link_id:{b}, 新增{len(new_link_gdf)}条!')

    # 修改当前后：
    print(new_link_gdf[[link_id_field, from_node_id_field, to_node_id_field]])
    print(new_link_gdf.columns)
    # 长度字段保留两位小数
    return new_link_gdf


def update_link(link_gdf=None, intersection_node_id=None, extended_intersection_gdf=None):
    """更新路网信息"""

    # 删除原有路口
    link_gdf.drop(index=link_gdf[(link_gdf[from_node_id_field] == intersection_node_id) |
                        (link_gdf[to_node_id_field] == intersection_node_id)].index, inplace=True, axis=0)
    # 加上新的路口
    link_gdf = link_gdf.append(extended_intersection_gdf)
    link_gdf.reset_index(inplace=True, drop=True)
    return link_gdf


def get_line_string_data(from_to_data=None, node_x_loc_dict=None, node_y_loc_dict=None, from_name=None, to_name=None):
    """
    生成gpd.GeoDataFrame
    :param from_to_data:
    :param node_x_loc_dict:
    :param node_y_loc_dict:
    :param from_name:
    :param to_name:
    :return:
    """
    from_to_data['from_x'] = from_to_data[from_name].apply(lambda x: node_x_loc_dict[x])
    from_to_data['from_y'] = from_to_data[from_name].apply(lambda x: node_y_loc_dict[x])
    from_to_data['to_x'] = from_to_data[to_name].apply(lambda x: node_x_loc_dict[x])
    from_to_data['to_y'] = from_to_data[to_name].apply(lambda x: node_y_loc_dict[x])

    from_to_data['from_point'] = from_to_data[['from_x', 'from_y']].apply(lambda x: Point(x[0], x[1]), axis=1)
    from_to_data['to_point'] = from_to_data[['to_x', 'to_y']].apply(lambda x: Point(x[0], x[1]), axis=1)
    from_to_data['geometry'] = from_to_data[['from_point', 'to_point']].apply(lambda x: LineString([x[0], x[1]]),
                                                                              axis=1)

    from_to_data.drop(['from_x', 'from_y', 'to_x', 'to_y', 'from_point', 'to_point'], axis=1, inplace=True)

    link_shp = gpd.GeoDataFrame(from_to_data, crs="EPSG:4326")
    return link_shp

