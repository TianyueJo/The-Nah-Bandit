import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast

''' 
The number of options is not fixed in restaurant recommendation, so we provide an additional version of environment and algorithms to handle this.
'''

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def prediction(theta, action_info):
    """
    action_info is a list that each element is an array with dimension d
    theta is an array with dimension d
    show_prob is used to use probability to substitute the predicted choice
    """
    A = len(action_info)
    utility = np.array([theta.T @ action_info[a] for a in range(A)])
    prob = softmax(utility)
    pred = np.argmax(prob)
    return pred

def loss_function(user_choice, pred):
    return int(user_choice != pred)

class UserData:
    def __init__(self, data):
        self.data = data
        self.num_user = len(data["ID"].unique())
        self.round_list = [len(data[data["ID"]==uid]) for uid in range(self.num_user)]
        self.round_gone = [0]*self.num_user
        self.row_map = np.zeros([self.num_user, max(self.round_list)], dtype=np.int_)
        for uid in range(self.num_user):
            for round in range(self.round_list[uid]):
                self.row_map[uid, round]=sum(self.round_list[:uid])+round

    def row(self, uid, round):
        return self.data.iloc[self.row_map[uid, round]]
    
    def gen_one_user(self):
        return np.random.choice([uid for uid in range(self.num_user) if self.round_gone[uid]!=self.round_list[uid]])
    
    def reset(self):
        self.round_gone = [0]*self.num_user

    def get_num_option(self, uid, round):
        return self.row(uid, round)["num_option"]
    
    def get_action_context(self, uid, round):
        row_content = self.row(uid, round)
        return np.array([
            np.array(
                ast.literal_eval(
                    row_content[f"context_{i}"]
                    )
                , dtype=np.float32) 
                for i in range(self.get_num_option(uid, round))
        ])
    
    def get_user_choice(self, uid, round):
        row_content = self.row(uid, round)
        return row_content["compliance"]
    
    def step(self):
        uid = self.gen_one_user()
        round = int(self.round_gone[uid])
        num_option = self.get_num_option(uid, round)
        action_context = self.get_action_context(uid, round)
        user_choice = self.get_user_choice(uid, round)
        self.round_gone[uid]+=1
        return uid, round, num_option, action_context, user_choice
    
    def observe(self, uid, round):
        num_option = self.get_num_option(uid, round)
        action_context = self.get_action_context(uid, round)
        user_choice = self.get_user_choice(uid, round)
        return uid, round, num_option, action_context, user_choice
    
class RandomAlg:
    def __init__(self):
        return 
    def take_action(self, num_option):
        return np.random.choice(num_option)

class Utility(nn.Module):
    def __init__(self, action_context_dim):
        super(Utility, self).__init__()
        self.action_context_dim = action_context_dim
        self.theta = nn.Parameter(torch.rand(self.action_context_dim))
    def forward(self, context):
        """
        context is an A*d array, theta is d array
        """
        # context = torch.from_numpy(context)
        context = torch.tensor(context, dtype = torch.float32)
        utility = (context @ self.theta).squeeze()
        prob = F.softmax(utility)
        return prob
class GetPreferenceParameter:
    def __init__(self, data, action_context_dim, num_epoch = 10):
        self.data = data
        self.num_epoch = num_epoch
        self.action_context_dim = action_context_dim
        self.num_user = len(data["ID"].unique())
        self.round_list = [len(data[data["ID"]==uid]) for uid in range(self.num_user)]
        self.row_map = np.zeros([self.num_user, max(self.round_list)], dtype=np.int_)
        for uid in range(self.num_user):
            for round in range(self.round_list[uid]):
                self.row_map[uid, round]=sum(self.round_list[:uid])+round

    def row(self, uid, round):
        return self.data.iloc[self.row_map[uid, round]]
    
    def get_num_option(self, uid, round):
        return self.row(uid, round)["num_option"]

    def get_action_context(self, uid, round):
        row_content = self.row(uid, round)
        return np.array([
            np.array(
                ast.literal_eval(
                    row_content[f"context_{i}"]
                    )
                , dtype=np.float32) 
                for i in range(self.get_num_option(uid, round))
        ])
    
    def get_user_choice(self, uid, round):
        row_content = self.row(uid, round)
        return row_content["compliance"]
    
    def train_one_user(self, uid):
        utility_model = Utility(self.action_context_dim)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(utility_model.parameters(), lr=0.5, weight_decay=0.01)
        num_epoch = self.num_epoch
        for epoch in range(num_epoch):
            loss = torch.zeros(1, requires_grad=True)
            optimizer.zero_grad()
            for round in range(self.round_list[uid]):
                context = self.get_action_context(uid, round)
                user_choice = self.get_user_choice(uid, round)
                pred = utility_model(context)
                loss = loss + loss_function(pred, torch.tensor(user_choice, dtype=torch.long))
            loss.backward()
            optimizer.step()
        accuracy_list = []
        for round in range(self.round_list[uid]):
            context = self.get_action_context(uid, round)
            user_choice = self.get_user_choice(uid, round)
            pred = prediction(utility_model.theta.detach().cpu().numpy(), context)
            accuracy_list.append(int(pred==user_choice))
        accuracy = np.mean(np.array(accuracy_list))
        return utility_model.theta, accuracy
    
    def train(self, show_accuracy = False):
        theta_list = []
        accuracy_list = []
        for uid in tqdm(range(self.num_user)):
            theta, accuracy = self.train_one_user(uid)
            accuracy_list.append(accuracy)
            theta_list.append(theta.detach().cpu().numpy())
        if show_accuracy:
            return np.array(theta_list), np.array(accuracy_list)
        return np.array(theta_list)

class Cluster:
    def __init__(self, theta_list, K):
        """
        theta_list is a N*d array
        """
        self.K = K
        self.theta_list = theta_list
        self.num_user, self.theta_dim = theta_list.shape
    def clustering(self):
        kmeans = KMeans(n_clusters=self.K, random_state=1, n_init='auto').fit(self.theta_list)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        return centroids, labels
    def get_distances(self, centroids, env):
        distance = np.zeros([self.K, self.num_user])
        for k in range(self.K):
            for uid in range(self.num_user):
                distance[k, uid] = self.distance(uid, centroids[:, k], env)
        return distance
    def distance(self, uid, centroid, env):
        dist = 0
        num_trials = env.round_list[uid]
        for t in range(num_trials):
            _, _, _, action_context, user_choice = env.observe(uid, t)
            action = prediction(centroid, action_context)
            dist+=int(action!=user_choice)
        return dist
    def clustering_with_loss_function(self, num_iter = 50, epsilon = 1e-3, env=None):
        points=self.theta_list.T 
        centers = points[:, np.random.choice(self.num_user, self.K, replace=False)]
        for iter in tqdm(range(num_iter)):
            prev_centers = centers.copy()
            distances = self.get_distances(centers, env)
            labels = np.argmin(distances, axis=0)
            for j in range(self.K):
                centers[:,j] = np.mean(points[:, labels == j], axis = 1)
            error = np.linalg.norm(prev_centers - centers)
            if error < epsilon:
                break
        return centers.T, labels
    
class EWC:
    def __init__(self, k, lr):
        self.k=k
        self.lr=lr
        self.centroids = None
        self.weight = {}

    def load_centroids(self, path):
        self.centers = np.load(path)

    def init_weight(self, uid):
        self.weight[uid] = np.ones(self.k)/self.k

    def choose_expert(self, uid):
        max_indices = np.where(self.weight[uid] == np.max(self.weight[uid]))[0]
        return np.random.choice(max_indices)
    
    def pred_of_expert(self, expert, action_info):
        # Output: the prediction of this expert
        theta = self.centers[expert]
        return prediction(theta, action_info)
    
    def take_action(self, uid, action_info):
        expert = self.choose_expert(uid)
        pred_of_expert = self.pred_of_expert(expert, action_info)
        return pred_of_expert
    
    def update(self, uid, action_info, user_choosen):
        self.weight[uid] *= np.array(
            [np.exp( 
                -1 * self.lr * loss_function(
                user_choosen, 
                self.pred_of_expert(expert, action_info)
                )
             ) 
            for expert in range(self.k)]
        )
        self.weight[uid] /= sum(self.weight[uid])

class OracleCluster:
    def __init__(self, labels, centroids):
        self.labels = labels
        self.centroids = centroids
    def take_action(self, uid, action_info):
        # Output: the recommended option
        expert = self.labels[uid]
        theta = self.centroids[expert]
        action = prediction(theta, action_info)
        return action, expert

class LinUCB:
    def __init__(self, num_user, action_context_dim, lr):
        self.num_user = num_user
        self.action_context_dim = action_context_dim
        self.lr = lr
        self.A_personal = [np.identity(self.action_context_dim) for _ in range(self.num_user)]
        self.b_personal = [np.zeros((self.action_context_dim, 1)) for _ in range(self.num_user)]
    
    def take_action(self, uid, num_arm, action_context):
        pred = np.zeros(num_arm)
        for action in range(num_arm):
            theta = np.linalg.inv(self.A_personal[uid]) @ self.b_personal[uid]
            pred[action] = theta.T @ action_context[action] + self.lr * np.sqrt(action_context[action].T @ np.linalg.inv(self.A_personal[uid]) @ action_context[action])
        return np.argmax(pred)
    
    def update(self, uid, R_arm, reward, action_context):
        self.A_personal[uid] += np.outer(action_context[R_arm], action_context[R_arm])
        self.b_personal[uid] += reward * action_context[R_arm].reshape(-1, 1)

class DynUCB:
    def __init__(self, num_user, action_context_dim, lr, num_cluster):
        self.num_user = num_user
        self.action_context_dim = action_context_dim
        self.lr = lr
        self.num_cluster = num_cluster
        self.A_personal = [np.identity(self.action_context_dim) for _ in range(self.num_user)]
        self.b_personal = [np.zeros((self.action_context_dim, 1)) for _ in range(self.num_user)]

        self.A_cluster = [np.identity(self.action_context_dim) for _ in range(self.num_cluster)]
        self.b_cluster = [np.zeros((self.action_context_dim, 1)) for _ in range(self.num_cluster)]
        self.theta_cluster = [np.linalg.inv(self.A_cluster[k]) @ self.b_cluster[k] for k in range(self.num_cluster)]
        
        self.labels = np.array([
            np.random.choice(range(self.num_cluster)) for uid in range(self.num_user)
        ])

    def take_action(self, uid, num_arm, action_context):
        pred = np.zeros(num_arm)
        label = self.labels[uid]
        A = self.A_cluster[label]
        b = self.b_cluster[label]
        theta = np.linalg.inv(A) @ b

        for action in range(num_arm):
            pred[action] = theta.T @ action_context[action] + self.lr * np.sqrt(action_context[action].T @ np.linalg.inv(A) @ action_context[action])
        
        return np.argmax(pred)
    
    def update(self, uid, R_arm, reward, action_context):
        self.A_personal[uid] += np.outer(action_context[R_arm], action_context[R_arm])
        self.b_personal[uid] += reward * action_context[R_arm].reshape(-1, 1)
        theta = np.linalg.inv(self.A_personal[uid]) @ self.b_personal[uid]
        self.labels[uid]=np.argmin(
            np.linalg.norm(
                (self.theta_cluster - theta)[:, 1:], axis=1
            )
        )
        self.update_cluster()

    def update_cluster(self):
        for k in range(self.num_cluster):
            self.A_cluster[k] = np.sum(
                np.array(self.A_personal)[self.labels==k, :], axis=0
            ) - np.identity(self.action_context_dim) * (np.sum(self.labels==k) -1)
            self.b_cluster[k] = np.sum(
                np.array(self.b_personal)[self.labels==k, :], axis=0
            )
            self.theta_cluster[k] = np.linalg.inv(self.A_cluster[k]) @ self.b_cluster[k]

class Linear_model:
    def __init__(self, action_context_dim):
        self.utility_model = Utility(action_context_dim+1)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.utility_model.parameters(), lr=0.6, weight_decay=0.001)
    
    def update(self, batch_context, y, rec):
        batch_context = self.transform_context(batch_context, rec)
        # batch_context = torch.tensor(batch_context, dtype = torch.float32)
        self.optimizer.zero_grad()
        pred = self.utility_model(batch_context)
        loss = self.loss_function(pred, torch.from_numpy(np.array(y)).long())
        loss.backward()
        self.optimizer.step()
        return pred

    def take_action(self, context):
        context = self.transform_context(context)
        # context = torch.tensor(context, dtype = torch.float32)
        return torch.argmax(self.utility_model(context)).item()
    
    def transform_context(self, context, rec=None):
        num_arm = context.shape[0]
        added_dim = np.zeros([num_arm, 1])
        if rec!=None:
            for arm in range(num_arm):
                added_dim[arm, 0]=int(arm==rec)
        return np.concatenate([context, added_dim], axis=1)