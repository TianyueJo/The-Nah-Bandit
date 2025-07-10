import pandas as pd
import numpy as np
from environment import *
import matplotlib.pyplot as plt
import xgboost as xgb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--NUM_TEST", type=int, help="Number of tests to run", default=10)
parser.add_argument("--EXPERIMENT_NAME", type=str, help="Name of the experiment", default="default")
parser.add_argument("--d", type=int, help="Dimension for PCA", default=6)
args = parser.parse_args()

USE_RANDOM = True
USE_EWC = True
USE_LINUCB = True
USE_DYNUCB = True
USE_LINEAR_MODEL = True
USE_XGBOOST = True

ORACLE_CLUSTER = True
USE_NEW_DIST = True

NUM_TESTS = args.NUM_TEST
PCA = args.EXPERIMENT_NAME == "PCA"
K = 8
DYNUCB_K = 8
LR_EWC = 1
LR_UCB = 0.05
PLOT_X_LIM = -1
num_epoch = 15
num_iter_cluster = 10
action_context_dim = 9

data_path = "data/"
if PCA:
    d=args.d
    action_context_dim = d
    data_path = f"additional_experiment/PCA/d={d}/"

params = {
    'objective': 'reg:squarederror',  # For regression task
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse'
}

random_regret = []
EWC_regret = []
ORA_CLU_regret = []
ORA_THETA_regret = []
LinUCB_regret = []
dynUCB_regret = []
linear_model_regret = []
XGBOOST_regret = []
for num_test in range(NUM_TESTS):
    ############## SPLIT DATA ###################

    # Load data
    user_data = pd.read_csv(data_path+"user_data.csv")

    # Identifying users with only one record
    user_counts = user_data['ID'].value_counts()
    single_record_users = user_counts[user_counts == 1].index

    # Removing users with only one record
    filtered_data = user_data[~user_data['ID'].isin(single_record_users)]

    # Re-grouping the filtered data
    re_grouped = filtered_data.groupby('ID')

    # Resetting the lists to store training and testing data
    train_data_list = []
    test_data_list = []

    # Iterating over each group and splitting
    for _, group in re_grouped:
        if np.random.choice([0, 1], p=[0.4, 0.6]) == 0:
            train_data_list.append(group)
        else:
            test_data_list.append(group)

    # Concatenating all the groups back together
    train_data = pd.concat(train_data_list)
    test_data = pd.concat(test_data_list)
    train_data = train_data.sort_values(by=['ID', 'Trial'])
    test_data = test_data.sort_values(by=['ID', 'Trial'])
    # reassign ID
    def reassign_ID(filtered_data):
        ID_list = filtered_data["ID"].unique()
        index = 0
        for ID in ID_list:
            filtered_data.loc[filtered_data["ID"]==ID, "ID"]= index
            index+=1
    reassign_ID(train_data)
    reassign_ID(test_data)
    # Checking the size of each set again
    train_size = len(train_data)
    test_size = len(test_data)
    print(train_size, test_size)  # Also returning the number of removed users
    train_data.to_csv(data_path+"train.csv", index = False)
    test_data.to_csv(data_path+"test.csv", index = False)


    ########## TRAINING ##########

    data = pd.read_csv(data_path+"train.csv")
    num_user = len(data["ID"].unique())
    print(f"number of users in training is {num_user}")
    
    env = UserData(data)
    theta_gen = GetPreferenceParameter(data, action_context_dim, num_epoch)
    theta = theta_gen.train()
    np.save(data_path+"theta_personal.npy", theta)
    # Clustering
    cluster = Cluster(theta, K)
    centroids, labels = cluster.clustering_with_loss_function(num_iter=num_iter_cluster, env=env) if USE_NEW_DIST else cluster.clustering() 
    np.save(data_path+"centroids.npy", centroids)
    np.save(data_path+"labels.npy", labels)


    ########## TESTING ##########
    data = pd.read_csv(data_path+"test.csv")
    num_user = len(data["ID"].unique())
    max_round = len(data)
    print("########### TEST PROGRAM ###########")
    print("number of users:", num_user)
    print("action context dimension:", action_context_dim)
    print("total rounds:", max_round)

    env = UserData(data)
    print(f"number of rounds is ranging from {min(env.round_list)} to {max(env.round_list)}")

    if USE_RANDOM:
        env.reset()
        RanAlg = RandomAlg()
        regret = [0]
        for _ in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            R_arm = RanAlg.take_action(num_option)
            loss = loss_function(user_choice, R_arm)
            regret += [regret[-1] + loss]
        random_regret.append(regret[:]) 
    
    if USE_LINEAR_MODEL:
        env.reset()
        linear_models = [Linear_model(action_context_dim) for uid in range(num_user)]
        regret = [0]
        for _ in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            R_arm = linear_models[uid].take_action(action_context)
            loss = loss_function(user_choice, R_arm)
            reward = 1-loss
            linear_models[uid].update(action_context, user_choice, R_arm)
            regret += [regret[-1] + loss]
        linear_model_regret.append(regret[:]) 
    
    if USE_LINUCB:
        env.reset()
        linUCB = LinUCB(num_user, action_context_dim, LR_UCB)
        regret = [0]
        for _ in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            R_arm = linUCB.take_action(uid, num_option, action_context)
            loss = loss_function(user_choice, R_arm)
            reward = 1-loss
            linUCB.update(uid, R_arm, reward, action_context)
            regret += [regret[-1] + loss]
        LinUCB_regret.append(regret[:]) 
        
    if USE_DYNUCB:
        env.reset()
        dynUCB = DynUCB(num_user, action_context_dim, LR_UCB, DYNUCB_K)
        regret = [0]
        for _ in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            R_arm = dynUCB.take_action(uid, num_option, action_context)
            loss = loss_function(user_choice, R_arm)
            reward = 1-loss
            dynUCB.update(uid, R_arm, reward, action_context)
            regret += [regret[-1] + loss]
        dynUCB_regret.append(regret[:]) 
    
    
    
    if ORACLE_CLUSTER:
        theta_gen = GetPreferenceParameter(data, action_context_dim, num_epoch)
        theta_oracle = theta_gen.train()
        np.save(data_path+"theta_oracle.npy", theta_oracle)
        cluster = Cluster(theta_oracle, K)
        centroids, labels = cluster.clustering_with_loss_function(num_iter=num_iter_cluster, env=env) if USE_NEW_DIST else cluster.clustering() 
        np.save(data_path+"oracle_centroids.npy", centroids)
        np.save(data_path+"oracle_labels.npt", labels)

        env.reset() 
        regret = [0]
        oracleCluster = OracleCluster(labels, centroids)
        for _ in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            R_arm, expert = oracleCluster.take_action(uid, action_context)
            loss = loss_function(user_choice, R_arm)
            regret +=[regret[-1] + loss]
        ORA_CLU_regret.append(regret[:])
        
        # Oracle theta
        env.reset() 
        regret = [0]
        for _ in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            theta = theta_oracle[uid]
            R_arm = prediction(theta, action_context)
            loss = loss_function(user_choice, R_arm)
            regret += [regret[-1] + loss]
        ORA_THETA_regret.append(regret[:])
    if USE_EWC:
        ewc = EWC(K, LR_EWC)
        ewc.load_centroids(data_path+"centroids.npy")
        env.reset()
        regret = [0]
        for _ in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            if round==0: ewc.init_weight(uid)
            R_arm = ewc.take_action(uid, action_context)
            ewc.update(uid, action_context, user_choice)
            loss = loss_function(user_choice, R_arm)
            regret += [regret[-1] + loss]
        EWC_regret.append(regret[:])
    
    if USE_XGBOOST:
        env.reset()
        XGB_models = [None]*num_user
        regret = [0]
        for t in range(max_round):
            uid, round, num_option, action_context, user_choice = env.step()
            value_list = np.zeros(num_option)
            for option in range(num_option):
                value_list[option]=XGB_models[uid].predict(xgb.DMatrix(action_context[option, np.newaxis])) if XGB_models[uid]!=None else 0
            
            # get loss
            rec = np.argmax(value_list)
            loss = 1 if rec!=user_choice else 0
            regret+=[regret[-1]+loss]
            # update
            y = np.zeros(num_option)
            y[user_choice]=1
            dtrain = xgb.DMatrix(action_context, label=y)
            if XGB_models[uid]!=None:
                XGB_models[uid]=xgb.train(params, dtrain, num_boost_round=1, xgb_model = XGB_models[uid])
            else:
                XGB_models[uid]=xgb.train(params, dtrain, num_boost_round=1)
        XGBOOST_regret.append(regret[:])
        
result_data = pd.DataFrame()
min_len = min([len(random_regret[test]) for test in range(NUM_TESTS)])
for test in range(NUM_TESTS):
    result_data[f"Random_{test}"] = random_regret[test][:min_len]
    result_data[f"LinUCB_{test}"] = LinUCB_regret[test][:min_len]
    result_data[f"DynUCB_{test}"] = dynUCB_regret[test][:min_len]
    result_data[f"linear model_{test}"] = linear_model_regret[test][:min_len]
    result_data[f"EWC_{test}"] = EWC_regret[test][:min_len]
    result_data[f"XGBoost_{test}"] = XGBOOST_regret[test][:min_len]
    result_data[f"ORA_CLU_{test}"] = ORA_CLU_regret[test][:min_len]
    result_data[f"ORA_THETA_{test}"] = ORA_THETA_regret[test][:min_len]

result_data.to_csv(data_path+f"result_data.csv")