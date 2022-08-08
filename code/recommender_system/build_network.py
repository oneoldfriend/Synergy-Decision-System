"""
-*- coding: utf-8 -*-
@Time : 2021/1/28 17:55
@Author : 01398085
@File : build_network.py
@Comment :
"""

import torch
import torch.nn.functional as F


class Network(torch.nn.Module):

    def __init__(self, hidden_layer1_size, hidden_layer2_size, users_size, restaurants_size,
                 embedding_size):
        super().__init__()
        # 用户，饭店的embedding层
        self.user_embedding = torch.nn.Embedding(users_size,
                                                 embedding_size)
        self.restaurant_embedding = torch.nn.Embedding(restaurants_size,
                                                       embedding_size)
        # 两层全连接隐藏层
        self.hidden_layer1 = torch.nn.Linear(2 * embedding_size,
                                             hidden_layer1_size)
        self.hidden_layer2 = torch.nn.Linear(hidden_layer1_size,
                                             hidden_layer2_size)
        # 输出层
        self.output_layer = torch.nn.Linear(hidden_layer2_size, 1)

    def forward(self, net_input):
        # 分别读取用户和饭店embedding
        embedding_list = [self.user_embedding(net_input[:, 0])]
        # 用户对应饭店集合的field embedding获取（多个饭店embedding的均值embedding）
        restaurants_field_embedding = torch.mean(
            self.restaurant_embedding(net_input[:, 1:]), 1)
        embedding_list.append(restaurants_field_embedding)
        # 将两个embedding concat后作为下一层输入
        flow = torch.cat(embedding_list, 1)
        # Tensor flowing！
        flow = F.relu(self.hidden_layer1(flow))
        flow = F.relu(self.hidden_layer2(flow))
        # CVR预测值输出
        return F.sigmoid(self.output_layer(flow))
