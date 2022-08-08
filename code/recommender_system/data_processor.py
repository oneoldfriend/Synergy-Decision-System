from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, users_size, restaurants_size):
        self.dataset_path = 'dataset/'
        self.users_size = users_size
        self.restaurants_size = restaurants_size
        self.training_set_np = self.read_training_set()
        self.x = torch.from_numpy(self.training_set_np[:, 0:2].astype(int))
        self.label = torch.from_numpy(self.training_set_np[:, -1:])

    def __getitem__(self, item):
        return self.x[item], self.label[item]

    def __len__(self):
        return self.training_set_np.shape[0]

    def data_generation(self):
        # 数据集初始化
        dataset = np.empty(shape=(0, 3))
        # 每个用户的样本数
        no_clicks_for_user = 1000
        # 读取用户-饭店下单概率矩阵
        user_cvr_matrix = np.loadtxt(self.dataset_path + 'user_cvr_matrix.csv',
                                     delimiter=',')
        # 计算用户-饭店点击概率矩阵
        sample_matrix = F.softmax(torch.from_numpy(user_cvr_matrix),
                                  dim=1).numpy()
        for user_id in range(0, self.users_size):
            # 对每个用户根据点击概率获得其点击饭店样本
            restaurant_samples = np.random.choice(
                range(0, self.restaurants_size),
                no_clicks_for_user,
                p=sample_matrix[user_id])
            # 对用户点击的每个饭店采样其下单样本
            for restaurant_id in restaurant_samples:
                order_prob = user_cvr_matrix[user_id, restaurant_id]
                sample_prob = np.random.rand()
                if sample_prob <= order_prob:
                    label = 1
                else:
                    label = 0
                # 获得 用户-饭店-下单 样本
                order_sample = np.array([user_id, restaurant_id, label])
                # 将该样本加入数据集中
                dataset = np.vstack((dataset, order_sample))
        user_order_matrix = np.zeros((self.users_size, self.restaurants_size))
        # 将 用户-饭店-下单 样本输出到文件
        np.savetxt(self.dataset_path + 'user_order_samples.csv', dataset[:],
                   delimiter=',')
        for row in range(0, dataset.shape[0]):
            # 获得用户下单次数矩阵
            user_order_matrix[
                int(dataset[row, 0]), int(dataset[row, 1])] += int(
                dataset[row, 2])
        # 将下单矩阵输出到文件
        np.savetxt(self.dataset_path + 'user_order_matrix.csv',
                   user_order_matrix[:],
                   delimiter=',')

    def read_training_set(self):
        file_path = self.dataset_path + 'user_order_samples.csv'
        if not Path(file_path).exists():
            # 数据不存在则生成数据
            self.data_generation()
        # 读取用户下单矩阵
        user_order_samples = np.loadtxt(file_path, delimiter=',')
        return user_order_samples

    def read_test_set(self):
        file_path = self.dataset_path + 'test_set.csv'
        if not Path(file_path).exists():
            self.data_generation()
        return np.loadtxt(file_path, delimiter=',').astype(int)
