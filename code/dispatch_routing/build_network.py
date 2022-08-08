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

    def __init__(self, hidden_layer1_size, hidden_layer2_size, feature_size):
        super().__init__()
        # 两层全连接隐藏层
        self.hidden_layer1 = torch.nn.Linear(feature_size,
                                             hidden_layer1_size)
        self.hidden_layer2 = torch.nn.Linear(hidden_layer1_size,
                                             hidden_layer2_size)
        # 输出层
        self.output_layer = torch.nn.Linear(hidden_layer2_size, 1)

    def forward(self, net_input):
        # Tensor flowing！
        flow = F.relu(self.hidden_layer1(net_input))
        flow = F.relu(self.hidden_layer2(flow))
        # 预测值输出
        return self.output_layer(flow)

    def state_to_feature(self, state_list):
        # 状态转为特征向量
        pass

    def post_state_to_feature(self, state_list):
        # 后决策状态转为特征向量
        pass
