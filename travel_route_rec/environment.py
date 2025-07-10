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

'''
A mechanism that if an user always choose comply / not comply, recommend the option that the user always choose
'''
USE_ALL_COM_OR_NOT = False

def softmax(x):
    T = 1 # Temperature for softmax
    x/=T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def prediction(theta, action_info, show_prob = False):
    """
    action_info is a list that each element is an array with dimension d
    theta is an array with dimension d
    show_prob is used to use probability to substitute the predicted choice
    """
    A = len(action_info)
    utility = np.array([theta.T @ action_info[a] for a in range(A)])
    prob = softmax(utility)
    pred = np.argmax(utility)
    if show_prob: return prob
    return pred

def loss_function(user_choice, pred):
    return int(user_choice != pred)

def load_data(data, num_user, max_round, user_context_dim, action_context_dim, num_action):
    """
    Get user_context, compliance_record, and action_context from data
    """
    user_context = np.zeros([num_user, user_context_dim])
    action_context = np.zeros([num_user, max_round, num_action, action_context_dim])
    Rec_record = np.zeros([num_user, max_round])
    compliance_record = np.zeros([num_user, max_round])

    for i in range(user_context_dim):
        user_context[:, i] = data.iloc[:, i+1]
    for i in range(max_round):
        column_name = f"compliance_{i}"
        compliance_record[:, i] = data[column_name]
    for i in range(max_round):
        column_name = f"Rec_{i}"
        Rec_record[:, i] = data[column_name]
    for t in range(max_round):
        for a in range(num_action):
            column_name_TT = f"TT_{t}{a}"
            column_name_CO2 = f"CO2_{t}{a}"
            action_context[:, t, a, 0] = data[column_name_TT]
            action_context[:, t, a, 1] = data[column_name_CO2]

    return user_context, action_context, Rec_record, compliance_record

def get_oracle_weight(data, num_user):
    oracle_weight = np.zeros([num_user, 3])
    oracle_weight[:, 0] = data["w_TT"]
    oracle_weight[:, 1] = data["w_CO2"]
    oracle_weight[:, 2] = data["w_Rec"]
    return oracle_weight
    
def transform_action_context(action_info):
    """
    Concate intercept to action context
    """
    action_info = np.concatenate([action_info, np.ones([action_info.shape[0], 1])], axis=1)
    return action_info

def concate_intercept(user_context_dim, user_context):
    """
    Concate intercept to user context
    """
    user_context_dim +=1 # [Age(3), Gender(1), Education(3), Num_cars(1), 1]
    user_context = np.concatenate([np.ones([user_context.shape[0], 1]), user_context], axis = 1)
    return user_context_dim, user_context

class UserData:
    def __init__(self, user_context, compliance_record, action_context, REC_RELAVANT_CHOICE=False, oracle_weight=None):
        """
        meaning of each variable:
            num_user = N
            user_context_dim = D
            action_context_dim = d
            max_round = T
            num_action = A

        input shape:
            user_context is a N*(D-1) array 
            compliance record is a N*T array
            action_context is a N*T*A*(d-1) array
        """
        self.num_user = user_context.shape[0]
        self.user_context_dim = user_context.shape[1]
        self.user_context = user_context
        self.compliance_record = compliance_record.astype(int)
        self.max_round = action_context.shape[1]
        self.num_action = action_context.shape[2]
        self.action_context_dim = action_context.shape[3]
        self.action_context = action_context
        self.rounds = np.zeros(self.num_user).astype(int)
        self.REC_RELAVANT_CHOICE = REC_RELAVANT_CHOICE # Whether the choice is relavant to recommendation
        self.oracle_weight = oracle_weight
    
    def reset(self):
        self.rounds = np.zeros(self.num_user)
    
    def get_action_context(self, uid, round):
        """
        return the action_context for user i at round t, A*d array
        """
        return self.action_context[uid, round]

    def get_user_context(self, uid):
        return self.user_context[uid]
    
    def get_user_choice(self, uid, action_context, rec):
        '''
        action_context: A*d array
        '''
        weight = self.oracle_weight[uid]
        rec_context = np.zeros([self.num_action, 1])
        rec_context[rec, 0] = 1
        new_context = np.concatenate([action_context, rec_context], axis=1)
        user_choice = prediction(weight, new_context)
        return user_choice
    
    def step(self, uid):
        round = int(self.rounds[uid])
        user_context = self.get_user_context(uid)
        action_context = self.get_action_context(uid, round)
        user_choice = None
        if self.REC_RELAVANT_CHOICE:
            user_choice = [self.get_user_choice(uid, action_context, i) for i in range(self.num_action)]
        else:
            user_choice = self.compliance_record[uid][round]
        self.rounds[uid] += 1
        return user_context, action_context, user_choice
    
    def observe(self, uid, round):
        """
        Observe the contexts and user's choice, for training
        """
        user_context = self.get_user_context(uid)
        action_context = self.get_action_context(uid, round)
        user_choice = None
        if self.REC_RELAVANT_CHOICE:
            user_choice = [self.get_user_choice(uid, action_context, i) for i in range(self.num_action)]
        else:
            user_choice = self.compliance_record[uid][round]
        return user_context, action_context, user_choice

class RandomAlg:
    '''
    Random algorithm
    '''
    def __init__(self, num_arm):
        self.num_arm = num_arm
    def take_action(self):
        return np.random.choice(self.num_arm)
      
class LinUCB:
    def __init__(self, num_user, num_arm, user_context_dim_each, action_context_dim_each, lr):
        '''
        Hybrid LinUCB, incorporating both user context and action context.
        A and b are for population term, A_personal and b_personal are for person specific term.
        '''
        self.num_arm = num_arm
        self.num_user = num_user
        self.user_context_dim_each = user_context_dim_each
        self.user_context_dim = user_context_dim_each * self.num_arm
        self.action_context_dim_each = action_context_dim_each
        self.action_context_dim = action_context_dim_each * self.num_arm
        self.alpha = lr
        self.A = np.identity(self.user_context_dim)
        self.b = np.zeros((self.user_context_dim, 1))
        self.A_personal = [np.identity(self.action_context_dim) for _ in range(self.num_user)]
        self.b_personal = [np.zeros((self.action_context_dim, 1)) for _ in range(self.num_user)]

    def transform_context(self, user_context, action_context):
        user_contexts = []
        action_contexts = []
        for i in range(self.num_arm):
            # Use context transform
            front = np.zeros((self.user_context_dim_each * i))
            back = np.zeros((self.user_context_dim_each * (self.num_arm - 1 - i)))
            new_user_context = np.concatenate([front, user_context, back], axis=0)
            user_contexts.append(new_user_context)

            # Action context transform
            front = np.zeros((self.action_context_dim_each * i))
            back = np.zeros((self.action_context_dim_each * (self.num_arm - 1 - i)))
            new_action_context = np.concatenate([front, action_context[i], back], axis=0)
            action_contexts.append(new_action_context)
        return user_contexts, action_contexts

    def take_action(self, round, uid, user_contexts, action_contexts):
        # Set the weight of population term
        gamma = 1 / (round + 1)

        # Get the prediction
        predict = np.zeros(self.num_arm)
        for action in range(self.num_arm):
            theta = np.linalg.inv(self.A) @ self.b
            predict[action] = gamma * (theta.T @ user_contexts[action] + self.alpha * np.sqrt(user_contexts[action].T @ np.linalg.inv(self.A) @ user_contexts[action]))
        
        for action in range(self.num_arm):
            theta = np.linalg.inv(self.A_personal[uid]) @ self.b_personal[uid]
            predict[action] += (1 - gamma) * (theta.T @ action_contexts[action] + self.alpha * np.sqrt(action_contexts[action].T @ np.linalg.inv(self.A_personal[uid]) @ action_contexts[action]))
        return np.argmax(predict)

    def update(self, uid, user_context, reward, action_context):
        # Update A and b
        self.A += np.outer(user_context, user_context)
        self.b += reward * user_context.reshape(-1, 1)

        self.A_personal[uid] += np.outer(action_context, action_context)
        self.b_personal[uid] += reward * action_context.reshape(-1, 1)

class DynUCB:
    def __init__(self, num_user, num_arm, user_context_dim_each, action_context_dim_each, lr, num_cluster):
        self.num_arm = num_arm
        self.num_user = num_user
        self.user_context_dim_each = user_context_dim_each
        self.user_context_dim = user_context_dim_each * self.num_arm
        self.action_context_dim_each = action_context_dim_each
        self.action_context_dim = action_context_dim_each * self.num_arm
        self.alpha = lr
        self.num_cluster = num_cluster

        self.A = np.identity(self.user_context_dim)
        self.b = np.zeros((self.user_context_dim, 1))

        self.A_cluster = [np.identity(self.action_context_dim) for _ in range(self.num_cluster)]
        self.b_cluster = [np.zeros((self.action_context_dim, 1)) for _ in range(self.num_cluster)]
        self.theta_cluster = [np.linalg.inv(self.A_cluster[k]) @ self.b_cluster[k] for k in range(self.num_cluster)]
        
        self.A_personal = [np.identity(self.action_context_dim) for _ in range(self.num_user)]
        self.b_personal = [np.zeros((self.action_context_dim, 1)) for _ in range(self.num_user)]

        self.labels = np.array([
            np.random.choice(range(self.num_cluster)) for uid in range(self.num_user)
        ])
    def transform_context(self, user_context, action_context):
        user_contexts = []
        action_contexts = []
        for i in range(self.num_arm):
            # Use context transform
            front = np.zeros((self.user_context_dim_each * i))
            back = np.zeros((self.user_context_dim_each * (self.num_arm - 1 - i)))
            new_user_context = np.concatenate([front, user_context, back], axis=0)
            user_contexts.append(new_user_context)

            # Action context transform
            front = np.zeros((self.action_context_dim_each * i))
            back = np.zeros((self.action_context_dim_each * (self.num_arm - 1 - i)))
            new_action_context = np.concatenate([front, action_context[i], back], axis=0)
            action_contexts.append(new_action_context)
        return user_contexts, action_contexts
    
    def take_action(self, round, uid, user_contexts, action_contexts):
        # Get the prediction
        predict = np.zeros(self.num_arm)
        label = self.labels[uid]
        A = self.A_cluster[label]
        b = self.b_cluster[label]
        theta = np.linalg.inv(A) @ b

        gamma = 1 / (round + 1)
        for action in range(self.num_arm):
            theta_pop = np.linalg.inv(self.A) @ self.b
            predict[action] = gamma * (theta_pop.T @ user_contexts[action] + self.alpha * np.sqrt(user_contexts[action].T @ np.linalg.inv(self.A) @ user_contexts[action]))
        
        for action in range(self.num_arm):
            predict[action] += (1-gamma) * (theta.T @ action_contexts[action] + self.alpha * np.sqrt(action_contexts[action].T @ np.linalg.inv(A) @ action_contexts[action]))

        return np.argmax(predict)

    def update(self, uid, user_context, reward, action_context):

        self.A += np.outer(user_context, user_context)
        self.b += reward * user_context.reshape(-1, 1)
        
        # Update user weight
        self.A_personal[uid] += np.outer(action_context, action_context)
        self.b_personal[uid] += reward * action_context.reshape(-1, 1)
        theta = np.linalg.inv(self.A_personal[uid]) @ self.b_personal[uid]
        # print((self.theta_cluster - theta)[:, 1:])
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

class Utility(nn.Module):
    def __init__(self, action_context_dim):
        '''
        FC layter + Softmax
        '''
        super(Utility, self).__init__()
        self.action_context_dim = action_context_dim
        self.theta = nn.Parameter(torch.rand(self.action_context_dim))
    def forward(self, context):
        """
        context is an A*d array, theta is d array
        """
        utility = (context @ self.theta).squeeze()
        prob = F.softmax(utility, dim=1 if context.dim()==3 else 0)
        return prob

class GetPreferenceParameter:
    def __init__(self, compliance_record, action_context, Rec_record, USE_NONCOMPLIANCE = True):
        """
        Meanning of the variables:
            num_user = N
            action_context_dim = d
            max_round = T
            num_action = A
        Input shape:
            compliance record is a N*T array
            action_context is a N*T*A*(d-1) array
            Rec_record is a N*T array
        """
        self.compliance_record = torch.tensor(compliance_record).int()
        self.Rec_record = torch.tensor(Rec_record).int()
        self.num_user = action_context.shape[0]
        self.max_round = action_context.shape[1]
        self.num_action = action_context.shape[2]
        self.action_context_dim = action_context.shape[3]
        self.action_context = torch.tensor(action_context, dtype = torch.float32)

        # Add Rec context
        Rec_context = torch.zeros([self.num_user, self.max_round, self.num_action, 1])
        if USE_NONCOMPLIANCE:
            for uid in range(self.num_user):
                for t in range(self.max_round):
                    rec_action = self.Rec_record[uid, t]
                    Rec_context[uid, t, rec_action, 0] = 1
        self.action_context = torch.concat([self.action_context, Rec_context], dim=3)
        self.action_context_dim += 1

    def train_one_user(self, uid):
        utility_model = Utility(self.action_context_dim)
        loss_function = nn.CrossEntropyLoss()
        # Add L-2 norm regularization to the parameters
        optimizer = torch.optim.Adam(utility_model.parameters(), lr=0.5, weight_decay=0.001)
        num_epoch = 130
        for epoch in range(num_epoch):
            batch_context = self.action_context[uid]
            y = F.one_hot(self.compliance_record[uid].to(torch.int64), num_classes=self.num_action).to(torch.float32)
            optimizer.zero_grad()
            pred = utility_model(batch_context)
            loss = loss_function(pred, self.compliance_record[uid].long())# + torch.abs(utility_model.theta[2]) * 0.06
            loss.backward()
            optimizer.step()
        return utility_model.theta
    
    def train(self):
        theta_list = []
        for uid in range(self.num_user):
            theta = self.train_one_user(uid)
            theta_list.append(theta.detach().cpu().numpy())
        return np.array(theta_list)

class Linear_model:
    def __init__(self, action_context_dim, USE_NON_COMPLIANCE = True):
        self.USE_NON_COMPLIANCE = USE_NON_COMPLIANCE
        self.utility_model = Utility(action_context_dim+1)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.utility_model.parameters(), lr=0.6, weight_decay=0.001)
    
    def update(self, batch_context, y, rec):
        batch_context = self.transform_context(batch_context, rec if self.USE_NON_COMPLIANCE else None)
        batch_context = torch.tensor(batch_context, dtype = torch.float32)
        self.optimizer.zero_grad()
        pred = self.utility_model(batch_context)
        loss = self.loss_function(pred, torch.from_numpy(np.array(y)).long())
        loss.backward()
        self.optimizer.step()
        return pred

    def take_action(self, context):
        context = self.transform_context(context)
        context = torch.tensor(context, dtype = torch.float32)
        return torch.argmax(self.utility_model(context)).item()
    
    def transform_context(self, context, rec=None):
        num_arm = context.shape[0]
        added_dim = np.zeros([num_arm, 1])
        if rec!=None:
            for arm in range(num_arm):
                added_dim[arm, 0]=int(arm==rec)
            
        return np.concatenate([context, added_dim], axis=1)

class Cluster:
    def __init__(self, max_round, theta_list, K):
        """
        theta_list is a N*d array
        """
        self.K = K
        self.theta_list = theta_list
        self.num_user, self.theta_dim = theta_list.shape
        self.max_round = max_round
    
    def get_distances(self, centers, env):
        distance = np.zeros([self.K, self.num_user])
        for k in range(self.K):
            for uid in range(self.num_user):
                distance[k, uid] = self.distance(uid, centers[:, k], env)
        return distance
    
    def distance(self, uid, center, env):
        dist = 0
        for t in range(self.max_round):
            _, action_context, user_choice = env.observe(uid, t)
            action = prediction(center, transform_action_context(action_context))
            dist += int(action != user_choice)
        return dist
    
    def clustering_with_loss_function(self, num_iterations = 50, epsilon = 1e-3, env = None):
        """
        K-Means with loss-guided distance metric.
        points is a d*N array
        """
        points=self.theta_list.T 
        centers = points[:, np.random.choice(self.num_user, self.K, replace=False)]
        for iter in tqdm(range(num_iterations)):
            prev_centers = centers.copy()
            distances = self.get_distances(centers, env)
            labels = np.argmin(distances, axis=0)
            for j in range(self.K):
                centers[:,j] = np.mean(points[:, labels == j], axis = 1)
            error = np.linalg.norm(prev_centers - centers)
            if error < epsilon:
                break
        return centers.T, labels

    def clustering(self):
        """
        Normal K-Means algorithm
        """
        kmeans = KMeans(n_clusters=self.K, random_state=1).fit(self.theta_list)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        return centroids, labels
    
class TrainUserContext:
    '''
    Logistic regression model that directly predict user's choice based on user's context. Used in EWC algorithm.
    '''
    def __init__(self):
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    def train_beta(self, contexts, labels):
        self.model.fit(contexts, labels)
    def weight_initialization(self, context):
        return self.model.predict_proba(context.reshape(1, -1))

class EWC:
    """
    Expert with Clustering algorithm
    """
    def __init__(self, k, lr, max_round, USE_USER_CONTEXT = False, user_contexts_train = None, labels_train = None):
        self.k = k
        self.centers = None
        self.lr = lr
        self.max_round = max_round
        self.weight = {}
        self.weight_init = {}
        self.USE_USER_CONTEXT = USE_USER_CONTEXT
        if USE_USER_CONTEXT:
            self.user_context_model = TrainUserContext()
            self.user_context_model.train_beta(user_contexts_train, labels_train)
        if USE_ALL_COM_OR_NOT:
            self.all_com_or_not = {}

    def load_centers(self, path):
        self.centers = np.load(path)

    def init_weight(self, uid, user_context):
        if self.USE_USER_CONTEXT:
            self.weight_init[uid] = self.user_context_model.weight_initialization(user_context)[0] 
            gamma = 1 # weight of context 
            self.weight[uid] = (1-gamma) * np.ones(self.k)/self.k + gamma * self.weight_init[uid] 
        else:
            self.weight[uid] = np.ones(self.k)/self.k
        if USE_ALL_COM_OR_NOT:
            self.all_com_or_not[uid] = -1

    def choose_expert(self, uid, rand):
        '''
        Choose an expert base one the weight
        '''
        if not rand:
            max_indices = np.where(self.weight[uid] == np.max(self.weight[uid]))[0]
            return np.random.choice(max_indices)
        return np.random.choice(range(self.k), p = self.weight[uid])
    
    def pred_of_expert(self, expert, action_info, show_prob = False):
        '''
        Output: the prediction of this expert
        '''
        theta = self.centers[expert]
        return prediction(theta, action_info, show_prob=show_prob)
    
    def take_action(self, uid, action_info, rand):
        '''
        rand means whether to choose the recommendation based on the probability distribution or simply choose the option with highest probability.
        When taking action, we treat 1_{Rec=a}=0 for all actions.
        '''
        action_info = transform_action_context(action_info)
        expert = self.choose_expert(uid, rand)
        pred_of_expert = self.pred_of_expert(expert, action_info)
        if USE_ALL_COM_OR_NOT:
            if self.all_com_or_not[uid] == 0 or self.all_com_or_not[uid] == 1:
                pred_of_expert = self.all_com_or_not[uid]
        return pred_of_expert
    
    def update(self, uid, t, action_info, user_choosen, Rec):
        '''
        Update the weight of expert.
        When updating the weight, we treat transform the action_info, add Rec context
        '''
        Rec_context = np.zeros([action_info.shape[0], 1])
        Rec_context[Rec, 0] = 1
        action_info = np.concatenate([
            action_info,
            Rec_context
        ], axis=1)
        self.weight[uid] *= np.array(
            [np.exp( 
                -1 * self.lr * loss_function(
                    user_choosen, 
                    self.pred_of_expert(expert, action_info)
                    )
                ) 
                for expert in range(self.k)
            ]
        )
        self.weight[uid] /= sum(self.weight[uid])
        if USE_ALL_COM_OR_NOT:
            if self.all_com_or_not[uid] == -1:
                self.all_com_or_not[uid] = user_choosen
            if self.all_com_or_not[uid] != user_choosen:
                self.all_com_or_not[uid] = 2

    def pred_only_user_context(self, uid, action_info):
        '''
        Make recommendation only by user's context
        '''
        action_info = transform_action_context(action_info)
        expert = np.argmax(self.weight_init[uid])
        pred = self.pred_of_expert(expert, action_info)
        return pred

class OracleCluster:
    def __init__(self, labels, centroids):
        """
        Input: the labels and centroids generated generated by clustering
        """
        self.labels = labels
        self.centroids = centroids

    def take_action(self, uid, action_info):
        '''
        Ouput: oracle action and the oracle expert
        '''
        action_info = transform_action_context(action_info)
        expert = self.labels[uid]
        theta = self.centroids[expert]
        action = prediction(theta, action_info)
        return action, expert





