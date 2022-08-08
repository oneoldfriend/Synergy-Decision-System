import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import dispatch_routing.graph_nn as network
import dispatch_routing.markov_decision_process as markov_decision_process


class DispatchRouting:
    def __init__(self, cvr_prediction_matrix, users_size, restaurants_size):
        # 网络参数
        self.node_features_dim = 5
        self.manual_features_dim = 6
        self.hidden_layer1_size = 32
        self.hidden_layer2_size = 64
        self.graph_embedding_size = 64
        self.hidden_layer3_size = 128
        self.learning_rate = 1e-3
        self.reward_decay = 0.9
        self.episodes = 10000
        self.test_size = 1000
        self.batch_size = 128
        self.network = network.Network(self.node_features_dim,
                                       self.manual_features_dim,
                                       self.hidden_layer1_size,
                                       self.hidden_layer2_size,
                                       self.graph_embedding_size,
                                       self.hidden_layer3_size)
        self.cvr_prediction_matrix = cvr_prediction_matrix
        self.cvr_groundtruth_matrix = np.loadtxt('dataset/user_cvr_matrix.csv',
                                                 delimiter=',')
        self.myopic = False
        # 模拟环境参数
        self.env_parameters = {'orders_no': 50,
                               'couriers_no': 3,
                               'max_work_time': 720,
                               'time_window_length': 60,
                               'service_time': 1,
                               'penalty_factor': 2,
                               'users_size': users_size,
                               'restaurants_size': restaurants_size,
                               'dis_matrix': np.loadtxt(
                                   'dataset/dis_matrix.csv', delimiter=','),
                               'position_matrix': np.loadtxt(
                                   'dataset/position_matrix.csv',
                                   delimiter=','),
                               'graph_size': [100, 100],
                               'depot_id': 150}
        self.experience_list = []

    def gnn_training(self, input_graph_list, manual_feature_list):
        optimizer = torch.optim.Adam(self.network.parameters(),
                                     self.learning_rate)
        loss_func = torch.nn.MSELoss()
        self.network.train()
        prediction_list = []
        label_list = []
        for input_graph in input_graph_list:
            manual_feature = manual_feature_list.pop(0)
            # 依次输入每个图并记录预测值
            optimizer.zero_grad()
            net_output = self.network(input_graph, manual_feature)
            prediction_list.append(net_output)
            label_list.append(input_graph.y.to(torch.float32))
        # 对这一batch的预测结果计算loss
        loss = loss_func(torch.cat(prediction_list), torch.cat(label_list))
        # 更新
        loss.backward()
        optimizer.step()
        return torch.mean(torch.cat(label_list)), loss

    def training(self, model_path):
        for episode in range(self.episodes):
            mdp = markov_decision_process.MarkovDecisionProcess(self.network,
                                                                self.data_generation(
                                                                    True),
                                                                self.env_parameters, False)
            # 开始MDP模拟
            mdp.state_init()
            while mdp.is_finished() is False:
                mdp.take_action()
                mdp.transition()
            # 对模拟结果处理成state-reward形式
            self.state_cost_list_processor(mdp.states_cost_stored)
            # 保证experience_list的size
            while len(self.experience_list) > self.batch_size:
                # 移除老的经验样本
                self.experience_list.pop(0)
            # 以batch size训练网络
            if len(self.experience_list) == self.batch_size:
                # 将state-reward形式转为图输入
                input_graph_list = network.state_to_feature(
                    self.experience_list,
                    self.env_parameters)
                loss_info = self.gnn_training(input_graph_list, [i[2] for i in self.experience_list])
                print(episode, loss_info)
            if (episode + 1) / 100 == 0:
                # 保存模型
                torch.save(self.network.state_dict(), model_path)

    def state_cost_list_processor(self, state_cost_list):
        # 对模拟结果处理成state-reward形式
        for index in range(len(state_cost_list) - 1, 0, -1):
            # 获取各个状态对应的current_reward
            previous_cost = state_cost_list[index - 1][1][0]
            state_cost_list[index][1] = state_cost_list[index][1][
                                            0] - previous_cost
        state_cost_list[0][1] = state_cost_list[0][1][0]
        for index in range(len(state_cost_list) - 1, 0, -1):
            # 倒序将current_reward以衰减相加得到各状态对应的future reward
            current_reward = state_cost_list[index][1]
            state_cost_list[index - 1][1] += self.reward_decay * current_reward
        self.experience_list += state_cost_list

    def test(self):
        model_path = 'model/dis_routing_model.pth'
        if not Path(model_path).exists():
            # 训练并保存推荐模型
            self.training(model_path)
        else:
            # 加载模型
            state_dict = torch.load(model_path)
            self.network.load_state_dict(state_dict)
        test_set_list = self.read_test_data()
        result_list = []
        for episode in range(self.test_size):
            test_set = []
            for _ in range(self.env_parameters['orders_no']):
                test_set.append(test_set_list.pop(0))
            mdp = markov_decision_process.MarkovDecisionProcess(self.network,
                                                                test_set,
                                                                self.env_parameters, self.myopic)
            # 开始MDP模拟
            mdp.state_init()
            while mdp.is_finished() is False:
                mdp.take_action()
                mdp.transition()
            # 记录本次test结果
            result_list.append([mdp.calc_cost(mdp.state['routes'])])
        result_np = np.array(result_list)
        # 求出test均值
        result_np = np.mean(result_np, axis=0)
        print(result_np)

    def data_generation(self, is_training):
        data_list = []
        # 下单用户id生成
        order_users_sample = np.random.choice(
            range(self.env_parameters['users_size']),
            self.env_parameters['orders_no'])
        # 计算用户-饭店点击概率矩阵
        if is_training is True:
            sample_matrix = F.softmax(
                torch.from_numpy(self.cvr_prediction_matrix),
                dim=1).numpy()
        else:
            sample_matrix = F.softmax(
                torch.from_numpy(self.cvr_groundtruth_matrix),
                dim=1).numpy()
        order_id = 0
        for user_id in order_users_sample:
            order_id += 1
            while True:
                # 点击饭店id生成
                restaurant_id = np.random.choice(
                    range(0, self.env_parameters['restaurants_size']),
                    1,
                    p=sample_matrix[user_id])[0]
                # 下单概率采样
                order_prob = self.cvr_prediction_matrix[user_id, restaurant_id]
                sample_prob = np.random.rand()
                if sample_prob <= order_prob:
                    # 用户在饭店完成下单
                    appearing_time = np.random.randint(0, self.env_parameters[
                        'max_work_time'] - self.env_parameters[
                                                           'time_window_length'])
                    # 生成订单数据
                    data_list.append(
                        {'order_id': order_id,
                         'appearing_time': appearing_time,
                         'destination_id': restaurant_id + 50,
                         'origin_id': user_id,
                         'start_time': appearing_time,
                         'end_time': appearing_time + self.env_parameters[
                             'time_window_length']})
                    # 退出while进行下一个订单模拟
                    break
        data_list.sort(key=lambda x: x['appearing_time'])
        return data_list

    def read_test_data(self):
        test_set_path = 'dataset/test_set-' + str(self.env_parameters['orders_no']) + '.txt'
        test_set_list = []
        if not Path(test_set_path).exists():
            # 生成测试集
            for _ in range(self.test_size):
                test_set_list += self.data_generation(False)
            file = open(test_set_path, 'w')
            for order_dict in test_set_list:
                file.write(str(order_dict))
                file.write('\n')
            file.close()
        # 加载测试集数据
        file = open(test_set_path, 'r')
        for line in file:
            test_set_list.append(eval(line))
        return test_set_list
