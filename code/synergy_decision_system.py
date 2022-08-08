"""
-*- coding: utf-8 -*-
@Time : 2021/2/2 9:52
@Author : 01398085
@File : synergy_decision_system.py
@Comment :
"""
from pathlib import Path

import numpy as np

import dispatch_routing.dispatch_routing as dispatch_routing
import recommender_system.recommender_system as recommender_system

if __name__ == '__main__':
    users_no = 50
    restaurants_no = 100
    # 检查是否存在推荐预测结果
    file_path = 'dataset/cvr_prediction_matrix.csv'
    recommender_system = recommender_system.RecommenderSystem(users_no, restaurants_no)
    if not Path(file_path).exists():
        # 不存在推荐预测结果，调用推荐系统进行预测
        recommender_system.prediction()
    cvr_prediction_matrix = np.loadtxt(file_path, delimiter=',')
    dispatch_routing = dispatch_routing.DispatchRouting(cvr_prediction_matrix,
                                                        users_no, restaurants_no)
    dispatch_routing.test()
