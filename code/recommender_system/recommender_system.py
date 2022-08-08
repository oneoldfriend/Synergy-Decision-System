"""
-*- coding: utf-8 -*-
@Time : 2021/1/28 9:35
@Author : 01398085
@File : recommender_system.py
@Comment :
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pathlib import Path

import recommender_system.build_network as build_network
import recommender_system.data_processor as data_processor


class RecommenderSystem:

    def __init__(self, users_size, restaurants_size):
        # 网络参数
        self.hidden_layer1 = 256
        self.hidden_layer2 = 128
        self.users_size = users_size
        self.restaurants_size = restaurants_size
        self.embedding_size = 64
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.episodes = 20000
        self.network = build_network.Network(self.hidden_layer1,
                                             self.hidden_layer2,
                                             self.users_size,
                                             self.restaurants_size,
                                             self.embedding_size)
        self.dataset = data_processor.MyDataset(self.users_size,
                                                self.restaurants_size)

    def training(self, rec_sys_model_path):
        data_loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True)
        optimizer = torch.optim.Adam(self.network.parameters(),
                                     self.learning_rate)
        loss_func = torch.nn.MSELoss()
        epoch = 0
        while epoch < self.episodes:
            for episode, data in enumerate(data_loader):
                epoch += 1
                x, label = data
                x, label = torch.LongTensor(x.long()), Variable(label)
                optimizer.zero_grad()
                prediction = self.network(x)
                loss = loss_func(prediction.to(torch.float32),
                                 label.to(torch.float32))
                print(epoch, self.evaluation(), loss)
                loss.backward()
                optimizer.step()
            torch.save(self.network.state_dict(), rec_sys_model_path)

    def evaluation(self):
        ground_truth = np.loadtxt('dataset/user_cvr_matrix.csv', delimiter=',')
        cosine_dis = []
        # 计算每个用户的预测CVR向量与ground truth的余弦距离
        for user_id in range(0, self.users_size):
            x = np.hstack(
                (np.full((self.restaurants_size, 1), user_id),
                 np.array(list(range(0, self.restaurants_size))).reshape(
                     (self.restaurants_size, 1))))
            prediction = self.network.forward(
                torch.LongTensor(x)).detach().numpy()
            similarity = np.dot(ground_truth[user_id, ...], prediction) / (
                    np.linalg.norm(ground_truth[user_id, ...]) *
                    np.linalg.norm(prediction))
            cosine_dis.append(1 - similarity)
        # 输出所有用户的平均余弦距离
        return np.mean(cosine_dis)

    def prediction(self):
        model_path = 'model/rec_sys_model.pth'
        if not Path(model_path).exists():
            # 训练并保存推荐模型
            self.training(model_path)
        else:
            # 加载模型
            state_dict = torch.load(model_path)
            self.network.load_state_dict(state_dict)
        cvr_prediction_matrix = np.zeros((0, 100))
        for user_id in range(0, self.users_size):
            x = np.hstack(
                (np.full((self.restaurants_size, 1), user_id),
                 np.array(list(range(0, self.restaurants_size))).reshape(
                     (self.restaurants_size, 1))))
            prediction = self.network.forward(
                torch.LongTensor(x)).detach().numpy().T
            cvr_prediction_matrix = np.vstack((cvr_prediction_matrix,
                                               prediction))
        print(self.evaluation())
        np.savetxt('dataset/cvr_prediction_matrix.csv',
                   cvr_prediction_matrix[:],
                   delimiter=',')

    def load_model(self, rec_sys_model_path):
        state_dict = torch.load(rec_sys_model_path)
        self.network.load_state_dict(state_dict)
