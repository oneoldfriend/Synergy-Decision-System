import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

import dispatch_routing.graph_nn as network


class MarkovDecisionProcess:
    def __init__(self, gnn, simulation_data, env_parameters, myopic):
        self.value_function = gnn
        self.step_length = 10
        self.state = {'current_time': 0,
                      'current_positions_index': [],
                      'routes': [],
                      'new_orders': [],
                      'picked_orders': [],
                      'remaining_orders': []}
        self.myopic = myopic
        self.outsourced_orders = []
        self.states_cost_stored = []
        self.simulation_data = simulation_data
        self.env_parameters = env_parameters
        self.selection_listed = {'greedy': greedy_selection,
                                 'epsilon_greedy': epsilon_greedy_selection,
                                 'softmax': softmax_selection}
        self.selection = 'greedy'
        self.epsilon = 2e-1

    def state_init(self):
        # 初始化状态
        for _ in range(self.env_parameters['couriers_no']):
            self.state['current_positions_index'].append(0)
            self.state['routes'].append(
                [{'position_id': self.env_parameters['depot_id'],
                  'is_origin': True,
                  'arrival_time': 0,
                  'departure_time': 0,
                  'is_visited': True,
                  'order_id': -1,
                  'start_time': 0,
                  'end_time': self.env_parameters[
                      'max_work_time']}])
        # 初始订单检查
        for order in self.simulation_data:
            if order['appearing_time'] == 0:
                self.state['new_orders'].append(order)
        self.dispatch(self.state)

    def take_action(self):
        # 派单
        if len(self.state['new_orders']) > 0:
            self.dispatch(self.state)
        # 路由
        acc_pos_list = self.get_accessible_positions()
        for courier_id in range(self.env_parameters['couriers_no']):
            if len(acc_pos_list[courier_id]) == 0:
                # 该配送员无路由可进行
                continue
            # 生成候选动作集对应的（状态-动作）信息
            post_state_list = []
            for acc_pos_index in acc_pos_list[courier_id]:
                temp_state = copy.deepcopy(self.state)
                if self.execute_action(temp_state, courier_id,
                                       acc_pos_index) is True:
                    # 动作成功执行
                    post_state_list.append([temp_state, courier_id])
            # 评估所有动作并执行所选动作
            self.state = self.post_state_evaluation(post_state_list,
                                                    self.state_to_manual_feature([i[0] for i in post_state_list]))

    def transition(self):
        last_current_time = self.state['current_time']
        # 更新当前时间
        self.state['current_time'] += self.step_length
        # 更新当前时间步产生的订单
        for order in self.simulation_data:
            if last_current_time < order['appearing_time'] <= self.state[
                'current_time']:
                self.state['new_orders'].append(order)
        # 更新配送员当前位置及路径中相关信息
        for courier_id in range(self.env_parameters['couriers_no']):
            current_positions_index = self.state['current_positions_index'][
                courier_id]
            couriers_route = self.state['routes'][courier_id]
            for position_index in range(current_positions_index + 1,
                                        len(couriers_route)):
                last_position = couriers_route[position_index - 1]
                position_to_check = couriers_route[position_index]
                # 检查该配送员的后续地点是否已服务
                if (last_position['departure_time'] <=
                        self.state['current_time']):
                    # 更新该地点访问状态
                    position_to_check['is_visited'] = True
                    # 更新配送员的当前地点
                    self.state['current_positions_index'][
                        courier_id] = position_index
                    # 更新当前状态的已取订单
                    if position_to_check['is_origin'] is True:
                        # 已取到该订单
                        self.state['picked_orders'].append(
                            position_to_check['order_id'])
                    else:
                        # 已送达该订单
                        self.state['picked_orders'].remove(
                            position_to_check['order_id'])
                    self.state['current_positions_index'][
                        courier_id] = position_index
                else:
                    break
            # 保证当前地点的离开时间不小于当前状态的时间
            old_departure_time = couriers_route[self.state['current_positions_index'][courier_id]]['departure_time']
            couriers_route[self.state['current_positions_index'][courier_id]][
                'departure_time'] = max(
                self.state['current_time'], old_departure_time)
        self.states_cost_stored.append(
            [copy.deepcopy(self.state), self.calc_cost(self.state['routes']),
             self.state_to_manual_feature([self.state])[0]])

    def is_finished(self):
        if self.state['current_time'] > self.env_parameters['max_work_time']:
            return True
        else:
            return False

    def get_accessible_positions(self):
        acc_pos_index_list = []
        for courier_id in range(self.env_parameters['couriers_no']):
            # 生成该配送员可访问地点的索引list
            acc_pos_index = []
            route = self.state['routes'][courier_id]
            current_positions_index = self.state['current_positions_index'][
                courier_id]
            max_position_index = len(route) - 1
            if current_positions_index < max_position_index:
                # 配送员存在后续地点
                for next_position_index in range(current_positions_index + 1,
                                                 max_position_index + 1):
                    # 检查所有后续地点是否可访问
                    next_position = route[next_position_index]
                    if next_position['is_origin'] is True or (next_position[
                                                                  'order_id'] in
                                                              self.state[
                                                                  'picked_orders']):
                        # 将可访问地点加入list中
                        acc_pos_index.append(next_position_index)
            acc_pos_index_list.append(acc_pos_index)
        return acc_pos_index_list

    def post_state_evaluation(self, post_state_list, manual_features_list):
        prediction_list = []
        if self.myopic:
            # 为myopic policy, 直接使用当前cost作为预测值
            for features in manual_features_list:
                prediction_list.append(features[0])
        else:
            # 状态数据转换为图输入数据
            graph_list = network.post_state_to_feature(post_state_list,
                                                       self.env_parameters)
            for input_graph in graph_list:
                manual_features = manual_features_list.pop(0)
                # 依次输入每个图并记录预测值
                net_output = self.value_function(input_graph, manual_features)
                prediction_list.append(torch.mean(net_output, dim=0).item())
        # 根据policy选择动作
        best_post_state_index = self.selection_listed.get(self.selection)(
            prediction_list)
        best_post_state = post_state_list[best_post_state_index][0]
        return best_post_state

    def execute_action(self, state, courier_id, next_pos_index):
        courier_route = state['routes'][courier_id]
        # 拿出要访问的下一个地点
        position_to_be_inserted = courier_route.pop(next_pos_index)
        # 将其插入到配送员当前所在地点的后面
        insert_position_index = state['current_positions_index'][courier_id] + 1
        courier_route.insert(insert_position_index, position_to_be_inserted)
        # 更新路径信息
        self.update_route([courier_route],
                          [state['current_positions_index'][courier_id]])
        return self.routing_check([courier_route])

    def routing_check(self, route_list):
        for route in route_list:
            if (route[len(route) - 1]['departure_time']
                    > self.env_parameters['max_work_time']):
                # 超出最大工作时长
                return False
        return True

    def dispatch(self, state):
        unassigned_orders = state['new_orders'] + state['remaining_orders']
        remaining_orders = []
        for order in unassigned_orders:
            best_cost = 1e+5
            best_courier_id = -1
            best_origin_pos_index = -1
            best_dest_pos_index = -1
            # 订单地点信息初始化
            origin = {'position_id': order['origin_id'],
                      'is_origin': True,
                      'arrival_time': 0,
                      'departure_time': 0,
                      'is_visited': False,
                      'order_id': order['order_id'],
                      'start_time': order['start_time'],
                      'end_time': order['end_time']}
            dest = {'position_id': order['destination_id'],
                    'is_origin': False,
                    'arrival_time': 0,
                    'departure_time': 0,
                    'is_visited': False,
                    'order_id': order['order_id'],
                    'start_time': order['start_time'],
                    'end_time': order['end_time']}
            # 遍历这个订单的所有可插入位置
            for courier_id in range(len(state['routes'])):
                route = state['routes'][courier_id]
                for origin_pos_index in range(
                        state['current_positions_index'][courier_id] + 1,
                        len(route) + 1):
                    # 遍历起点可插入位置
                    route.insert(origin_pos_index, origin)
                    for dest_pos_index in range(origin_pos_index + 1,
                                                len(route) + 1):
                        # 遍历终点可插入位置
                        route.insert(dest_pos_index, dest)
                        # 更新新路径
                        cost = self.update_route([route], [
                            state['current_positions_index'][courier_id]])
                        # 评估该位置
                        if (self.routing_check(
                                [route]) is True) and cost < best_cost:
                            best_cost = cost
                            best_origin_pos_index = origin_pos_index
                            best_dest_pos_index = dest_pos_index
                            best_courier_id = courier_id
                        # 路径复原
                        route.pop(dest_pos_index)
                    route.pop(origin_pos_index)
            if best_courier_id == -1:
                # 未能分配该订单
                if order in state['new_orders']:
                    # 订单留到下一次分配
                    remaining_orders.append(order)
                else:
                    # 将订单外包
                    self.outsourced_orders.append(order)
            else:
                # 将该订单插入其当前最优位置
                state['routes'][best_courier_id].insert(best_origin_pos_index,
                                                        origin)
                state['routes'][best_courier_id].insert(best_dest_pos_index,
                                                        dest)
            # 更新该配送员路径
            self.update_route([state['routes'][best_courier_id]], [
                state['current_positions_index'][best_courier_id]])
        self.state['new_orders'].clear()
        self.state['remaining_orders'] = remaining_orders

    def calc_cost(self, routes_list):
        travel_cost = 0
        wait_time = 0
        penalty = 0
        for route in routes_list:
            for pos_index in range(1, len(route)):
                last_position = route[pos_index - 1]
                position = route[pos_index]
                travel_cost += self.env_parameters['dis_matrix'][
                    last_position['position_id'], position['position_id']]
                wait_time += max(0,
                                 position['start_time'] - position[
                                     'arrival_time'])
                penalty += max(0, self.env_parameters['penalty_factor'] * (
                        position['arrival_time'] - position['end_time']))
        cost = travel_cost + wait_time + penalty
        return cost, travel_cost, wait_time, penalty

    def state_to_manual_feature(self, state_list):
        # 获取状态的时间相关特征列表
        manual_features_list = []
        for state in state_list:
            # 所有配送员剩余可工作时间
            leftover_time = 0
            # 当前状态时间
            current_time = state['current_time']
            # cost信息（路程时间，等待时间，迟到时间）
            cost_info = list(self.calc_cost(state['routes']))
            for routes in state['routes']:
                leftover_time += self.env_parameters['max_work_time'] - routes[len(routes) - 1]['departure_time']
            cost_info.append(leftover_time)
            cost_info.append(current_time)
            manual_features_list.append(cost_info)
        return manual_features_list

    def update_route(self, routes_list, current_positions_index):
        cost = 0
        for courier_id in range(len(routes_list)):
            route = routes_list[courier_id]
            for pos_index in range(current_positions_index[courier_id] + 1,
                                   len(route)):
                last_position = route[pos_index - 1]
                position = route[pos_index]
                # 获取与上一地点距离
                dis = self.env_parameters['dis_matrix'][
                    last_position['position_id'], position['position_id']]
                # 更新到达时间和出发时间
                position['arrival_time'] = last_position['departure_time'] + dis
                position['departure_time'] = (max(position['start_time'],
                                                  position['arrival_time']) +
                                              self.env_parameters[
                                                  'service_time'])
                # cost记录
                cost += dis
                cost += max(0,
                            position['start_time'] - position['arrival_time'])
                cost += max(0, self.env_parameters['penalty_factor'] * (
                        position['arrival_time'] - position['end_time']))
        return cost


def greedy_selection(prediction_list):
    return prediction_list.index(min(prediction_list))


def epsilon_greedy_selection(prediction_list):
    epsilon = 0.2
    select_index = prediction_list.index(min(prediction_list))
    prop = random.random()
    if prop <= epsilon:
        select_index = random.randint(0, len(prediction_list))
    return select_index


def softmax_selection(prediction_list):
    prediction_list_np = np.array(prediction_list)
    sample_prob = F.softmax(torch.from_numpy(prediction_list_np),
                            dim=1).numpy()
    select_index = np.random.choice(
        range(0, len(prediction_list)),
        1,
        p=sample_prob)[0]
    return select_index
