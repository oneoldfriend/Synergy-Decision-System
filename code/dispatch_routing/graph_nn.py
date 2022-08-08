import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv


class Network(torch.nn.Module):
    def __init__(self, node_feature_dim, manual_feature_dim, hidden_layer1_size, hidden_layer2_size,
                 graph_embedding_size, hidden_layer3_size):
        super().__init__()
        # self.hidden_layer1 = GCNConv(node_feature_dim, hidden_layer1_size)
        self.hidden_layer1 = SAGEConv(node_feature_dim, hidden_layer1_size)
        # self.hidden_layer2 = GCNConv(hidden_layer1_size, hidden_layer2_size)
        self.hidden_layer2 = SAGEConv(hidden_layer1_size, hidden_layer2_size)
        # self.graph_embedding = GCNConv(hidden_layer2_size, graph_embedding_size)
        # self.graph_embedding = SAGEConv(hidden_layer2_size, graph_embedding_size)
        # 隐藏层
        self.hidden_layer3 = torch.nn.Linear(graph_embedding_size + manual_feature_dim,
                                             hidden_layer3_size)
        # 线性回归层
        self.linear_layer = torch.nn.Linear(hidden_layer3_size, 1)

    def forward(self, input_graph, manual_features):
        x, edge_index = input_graph.x, input_graph.edge_index
        flow = F.leaky_relu(self.hidden_layer1(x, edge_index))
        flow = F.leaky_relu(self.hidden_layer2(flow, edge_index))
        # flow = F.relu(self.graph_embedding(flow, edge_index))
        flow = torch.mean(flow, dim=0)
        flow = torch.hstack((flow, torch.tensor(manual_features, dtype=torch.float)))
        flow = F.relu(self.hidden_layer3(flow))
        # flow = F.relu(self.hidden_layer4(flow))
        return self.linear_layer(flow)


def post_state_to_feature(post_state_list, env_parameters):
    graph_list = []
    for state, executed_courier_id in post_state_list:
        # 将决策后状态转为图数据
        pos_id_index_list = [150]
        x = []
        edge_source = []
        edge_target = []
        for courier_id in range(len(state['routes'])):
            route = state['routes'][courier_id]

            if courier_id == executed_courier_id:
                # 决策影响图的联通性
                one_direction_index = state['current_positions_index'][
                                          courier_id] + 1
            else:
                one_direction_index = state['current_positions_index'][
                    courier_id]
            route_edge_source, route_edge_target = route_to_graph(route, x,
                                                                  one_direction_index,
                                                                  env_parameters,
                                                                  pos_id_index_list)
            edge_source += route_edge_source
            edge_target += route_edge_target
        edge_index = [edge_source, edge_target]
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        graph_list.append(Data(x=x, edge_index=edge_index))
    return graph_list


def state_to_feature(state_reward_list, env_parameters):
    graph_list = []
    for state, reward, _ in state_reward_list:
        # 将当前状态转为图数据
        pos_id_index_list = [150]
        x = []
        edge_source = []
        edge_target = []
        for courier_id in range(len(state['routes'])):
            route = state['routes'][courier_id]
            one_direction_index = state['current_positions_index'][
                courier_id]
            route_edge_source, route_edge_target = route_to_graph(route, x,
                                                                  one_direction_index,
                                                                  env_parameters,
                                                                  pos_id_index_list)
            edge_source += route_edge_source
            edge_target += route_edge_target
        edge_index = [edge_source, edge_target]
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor([reward], dtype=torch.float)
        graph_list.append(Data(x=x, y=y, edge_index=edge_index))
    return graph_list


def node_to_feature(node, env_parameters):
    if node['is_origin'] is True:
        node_type = 1
    else:
        node_type = 0
    # 提取点的特征
    node_x = env_parameters['position_matrix'][node['position_id'], 0]
    node_y = env_parameters['position_matrix'][node['position_id'], 1]
    return [node_type,
            node_x / env_parameters['graph_size'][0],
            node_y / env_parameters['graph_size'][1],
            float(node['start_time']) / env_parameters['max_work_time'],
            float(node['end_time']) / env_parameters['max_work_time'], ]


def route_to_graph(route, x, one_direction_index,
                   env_parameters, pos_id_index_list):
    route_edge_source = []
    route_edge_target = []
    for position_index in range(len(route)):
        # 当前点特征
        source_node = route[position_index]
        source_node_feature = node_to_feature(source_node, env_parameters)
        if source_node_feature not in x:
            x.append(source_node_feature)
        # 加入与当前点相连的边信息
        edge_source_for_this_node, edge_target_for_this_node = edge_index_generation(
            position_index, route, one_direction_index, pos_id_index_list)
        route_edge_source += edge_source_for_this_node
        route_edge_target += edge_target_for_this_node
    return route_edge_source, route_edge_target


def edge_index_generation(position_index, route, one_direction_index,
                          pos_id_index_list):
    edge_source = []
    edge_target = []
    source_node = route[position_index]
    source_node_index = get_node_index(source_node, pos_id_index_list)
    if position_index < len(route) - 1:
        # 加入与后续点连接情况
        next_position_index = position_index + 1
        if position_index > one_direction_index:
            # 对未确定访问顺序的点，与当前点以全连通方式加入图中
            while next_position_index < len(route):
                # target node index确定
                target_node = route[next_position_index]
                target_node_index = get_node_index(target_node,
                                                   pos_id_index_list)
                # 双向边加入
                edge_source.append(source_node_index)
                edge_target.append(target_node_index)
                edge_source.append(target_node_index)
                edge_target.append(source_node_index)
                next_position_index += 1
        elif position_index < one_direction_index:
            # target node index确定
            target_node = route[next_position_index]
            target_node_index = get_node_index(target_node, pos_id_index_list)
            # 单向边加入
            edge_source.append(source_node_index)
            edge_target.append(target_node_index)
        else:
            # 最后一个确定点与后续点均可单向连接
            while next_position_index < len(route):
                target_node = route[next_position_index]
                target_node_index = get_node_index(target_node,
                                                   pos_id_index_list)
                edge_source.append(source_node_index)
                edge_target.append(target_node_index)
                next_position_index += 1
    return edge_source, edge_target


def get_node_index(node, node_index_list):
    # node index确定
    if node['position_id'] not in node_index_list:
        node_index_list.append(node['position_id'])
    return node_index_list.index(node['position_id'])
