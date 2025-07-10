import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import *
import argparse
import threading
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument("--NUM_TEST", type=int, help="Number of tests to run")
parser.add_argument("--EXPERIMENT_NAME", type=str, help="Name of the experiment", default="default")
parser.add_argument("--beta_scaler", type=int, help="Beta scaler value", default=1)
parser.add_argument("--prob_noise", type=float, help="Probability of noise", default=0.1)
args = parser.parse_args()
NUM_TEST = args.NUM_TEST
EXPERIMENT_NAME = args.EXPERIMENT_NAME
beta_scaler = args.beta_scaler

USE_NEW_DISTANCE = True
USE_RANDOM = False
USE_LINUCB = True
USE_EWC = True
USE_XGBOOST = True
USE_LINEAR_MODEL = True
USE_EWC_NO_USER_CONTEXT = True
USE_EWC_NO_NONCOMPLIANCE = True
ORACLE_CLUSTER = True
USE_DYN_UCB = True
USE_ONLY_USER_CONTEXT = True
USE_USER_CONTEXT = True

data_path = f"data/beta={beta_scaler}/"
if EXPERIMENT_NAME=="heterogeneous_users":
    data_path = f"additional_experiment/heterogeneous_users/"
elif EXPERIMENT_NAME=="noised_user_context":
    prob_noise = args.prob_noise
    data_path = f"additional_experiment/noised_user_context/p={prob_noise}/"

random_regret = []
LinUCB_regret = []
EWC_regret = []
ORA_CLU_regret = []
ORA_THETA_regret = []
LM_regret = []
EWC_NO_USER_CONTEXT_regret = []
EWC_NO_NONCOMPLIANCE_regret = []
ONLY_USER_CONTEXT_regret = []
DynUCB_regret = []
XGB_regret = []

for test in range(NUM_TEST):
    ########### SPLIT #############
    data = pd.read_csv(data_path+'synthetic_data.csv')
    data_train, data_test = train_test_split(data, test_size = 0.4)
    data_train.to_csv(data_path+f'train.csv', index = False)
    data_test.to_csv(data_path+f'test.csv', index = False)

    ############# TRAIN ###############
    # Hyperparameters
    K = 6
    NUM_ITER_KMEANS = 10

    # Load data
    data = pd.read_csv(data_path+"train.csv")
    num_user = len(data)
    max_round = 40
    user_context_dim = 8 # [Age(3), Gender(1), Education(3), Num_cars(1)]
    action_context_dim = 2 # [TT, CO2]
    num_action = 2 # [Regular route, Eco-friendly route]


    # Get user_context, action_context, Rec_record, compliance_record from data
    user_context, action_context, Rec_record, compliance_record = load_data(data,
                                                                            num_user,
                                                                            max_round,
                                                                            user_context_dim,
                                                                            action_context_dim,
                                                                            num_action
                                                                            )

    # Concate intercept to the context
    user_context_dim, user_context = concate_intercept(user_context_dim, user_context)
    np.save(data_path+"user_context_train.npy", user_context)

    # Create environment
    env = UserData(user_context, compliance_record, action_context)

    # Calculate theta_i for each user in training part.
    theta_gen = GetPreferenceParameter(compliance_record, action_context, Rec_record)
    theta = theta_gen.train()
    np.save(data_path+"theta_personal.npy", theta)

    # Apply clustering algorithm to the set of theta_i. Get the centroids and labels.
    max_round = action_context.shape[1]
    cluster = Cluster(max_round, theta, K)
    centroids, labels = cluster.clustering_with_loss_function(env=env, num_iterations=NUM_ITER_KMEANS) if USE_NEW_DISTANCE else cluster.clustering()
    np.save(data_path+"centroids.npy", centroids)
    np.save(data_path+"labels.npy", labels)

    env.reset()
    # Calculate theta_i for each user in training part.
    theta_gen = GetPreferenceParameter(compliance_record, action_context, Rec_record, False)
    theta = theta_gen.train()
    # Apply clustering algorithm to the set of theta_i. Get the centroids and labels.
    max_round = action_context.shape[1]
    cluster = Cluster(max_round, theta, K)
    centroids, labels = cluster.clustering_with_loss_function(env=env, num_iterations=NUM_ITER_KMEANS) if USE_NEW_DISTANCE else cluster.clustering()
    np.save(data_path+"centroids_without_non.npy", centroids)
    np.save(data_path+"labels_without_non.npy", labels)

    ############# TEST ###############
    # Hyperparameters
    DYNUCB_K = 12
    LR_UCB = 0.65 # Learning rate for UCB algorithm, default=0.05
    LR_EWC = 10 # Learning rate for EWC algorithm 

    # Experiment setting
    PLOT_ROUND = 40
    data = pd.read_csv(data_path+"test.csv")
    num_user = len(data)
    max_round = PLOT_ROUND
    user_context_dim = 8 # [1, age, gender, num_cars]
    action_context_dim = 2 # [1, TT, CO2]
    num_action = 2


    # Get user_context, action_context, Rec_record, compliance_record from data
    user_context, action_context, Rec_record, compliance_record = load_data(data,
                                                                            num_user,
                                                                            max_round,
                                                                            user_context_dim,
                                                                            action_context_dim,
                                                                            num_action
                                                                            )

    # Extract oracle weight
    oracle_weight = get_oracle_weight(data, num_user)

    # Add an intercept to the context
    user_context_dim, user_context = concate_intercept(user_context_dim, user_context)

    # Set Plot rounds
    print("########### TEST PROGRAM ###########")
    print("number of users:", num_user)
    print("user context dimension:", user_context_dim)
    print("action context dimension:", action_context_dim)
    print("total rounds:", max_round)

    # Set the environment
    env = UserData(user_context, compliance_record, action_context, True, oracle_weight)

    if USE_XGBOOST:
        action_context_reshaped = action_context.reshape(action_context.shape[0], action_context.shape[1], -1)
        def thread_task(uid):
            X = action_context_reshaped[uid, :, :]
            y = compliance_record[uid, :]
            added = np.array([[100, 100, 100, 100]])
            X = np.concatenate([added, X], axis=0)
            y = np.concatenate([[0], y], axis=0)

            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.3,
                n_estimators=100,
                objective='binary:logistic', 
                eval_metric='logloss',      
                use_label_encoder=False
            )
            for t in range(max_round+1):
                model.fit(X[:t, :], y[:t])
                context = X[t, None, :]
                choice = y[t]
                rec = model.predict(context)
                rec = rec[0]
                if t>=1:
                    regret[uid, t-1]=int(rec!=choice)

        regret = np.zeros((num_user, max_round))
        threads=[]
        for uid in range(num_user):
            t = threading.Thread(target=thread_task, args=(uid,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        regret = regret.T.ravel()
        regret = np.cumsum(regret, axis=0)
        XGB_regret.append(regret)

    # Random
    if USE_RANDOM:
        env.reset()
        RanAlg = RandomAlg(num_action)
        regret = [0]
        for i in range(max_round):
            for j in range(num_user):
                context, _, user_choice = env.step(j)
                R_arm = RanAlg.take_action()
                user_choice = user_choice[R_arm]
                loss = loss_function(user_choice, R_arm)
                regret += [regret[-1] + loss]
        random_regret.append(regret[:])

    # Hybrid linear UCB algorithm that focus on action infomation (TT and CO2), only consider them as context for each person
    if USE_LINUCB:
        env.reset()
        linUCB = LinUCB(num_user, num_action, user_context_dim, action_context_dim, LR_UCB)
        regret = [0]
        for i in range(max_round):
            for j in range(num_user):
                user_context_i, action_context_i, user_choice = env.step(j)
                user_contexts, action_contexts = linUCB.transform_context(user_context_i, action_context_i)
                R_arm = linUCB.take_action(i, j, user_contexts, action_contexts) 
                user_choice = user_choice[R_arm]
                loss = loss_function(user_choice, R_arm)
                reward = 1-loss
                linUCB.update(j, user_contexts[R_arm], reward, action_contexts[R_arm])
                regret += [regret[-1] + loss]
        LinUCB_regret.append(regret[:])
    
    if USE_DYN_UCB:
        env.reset()
        dynUCB = DynUCB(num_user, num_action, user_context_dim, action_context_dim, LR_UCB, DYNUCB_K)
        regret = [0]
        for i in range(max_round):
            for j in range(num_user):
                user_context_i, action_context_i, user_choice = env.step(j)
                user_contexts, action_contexts = dynUCB.transform_context(user_context_i, action_context_i)
                R_arm = dynUCB.take_action(i, j, user_contexts, action_contexts) 
                user_choice = user_choice[R_arm]
                loss = loss_function(user_choice, R_arm)
                reward = 1-loss
                dynUCB.update(j, user_contexts[R_arm], reward, action_contexts[R_arm])
                regret += [regret[-1] + loss]
        dynUCB.debug()
        DynUCB_regret.append(regret[:])

    if ORACLE_CLUSTER:
        # Generate oracle theta
        model = GetPreferenceParameter(compliance_record, action_context, Rec_record)
        theta_oracle = model.train()
        np.save(data_path+"theta_oracle.npy", theta_oracle)
        cluster = Cluster(max_round, theta_oracle, K)
        centroids = None
        labels = None
        if USE_NEW_DISTANCE:
            centroids, labels = cluster.clustering_with_loss_function(num_iterations = NUM_ITER_KMEANS, env=UserData(user_context, compliance_record, action_context))
        else:
            centroids, labels = cluster.clustering()
        np.save(data_path+"oracle_centroids.npy", centroids)
        np.save(data_path+"oracle_labels.npt", labels)

        # Cluster
        env.reset() 
        regret = [0]
        oracleCluster = OracleCluster(labels, centroids)
        for i in range(max_round):
            for uid in range(num_user):
                context, action_info, user_choice = env.step(uid)
                R_arm, expert = oracleCluster.take_action(uid, action_info)
                user_choice = user_choice[R_arm]
                loss = loss_function(user_choice, R_arm)
                regret += [regret[-1] + loss]
        ORA_CLU_regret.append(regret[:])

        # Oracle theta
        env.reset() 
        regret = [0]
        for i in range(max_round):
            for uid in range(num_user):
                context, action_info, user_choice = env.step(uid)
                theta = theta_oracle[uid]
                R_arm = prediction(theta, transform_action_context(action_info))
                user_choice = user_choice[R_arm]
                loss = loss_function(user_choice, R_arm)
                regret += [regret[-1] + loss]
        ORA_THETA_regret.append(regret[:])
    
    # Expert with clustering algorithm
    if USE_EWC:
        expert_with_clus = EWC(K, LR_EWC, max_round, USE_USER_CONTEXT=USE_USER_CONTEXT, user_contexts_train=np.load(data_path+"user_context_train.npy"), labels_train=np.load(data_path+"labels.npy")) 
        expert_with_clus.load_centers(data_path+"centroids.npy")
        env.reset()
        regret = [0]
        for i in range(max_round):
            for uid in range(num_user):
                context, action_info, user_choice = env.step(uid)
                if i == 0: expert_with_clus.init_weight(uid, context)
                R_arm = expert_with_clus.take_action(uid, action_info, False) # rand = False
                user_choice = user_choice[R_arm]
                expert_with_clus.update(uid, i, action_info, user_choice, R_arm) ## modified here
                loss = loss_function(user_choice, R_arm)
                regret += [regret[-1] + loss]
        EWC_regret.append(regret[:])
    
    if USE_EWC_NO_NONCOMPLIANCE:
        expert_with_clus = EWC(K, LR_EWC, max_round, USE_USER_CONTEXT=USE_USER_CONTEXT, user_contexts_train=np.load(data_path+"user_context_train.npy"), labels_train=np.load(data_path+"labels_without_non.npy")) 
        expert_with_clus.load_centers(data_path+"centroids_without_non.npy")
        env.reset()
        regret = [0]
        for i in range(max_round):
            for uid in range(num_user):
                context, action_info, user_choice = env.step(uid)
                if i == 0: expert_with_clus.init_weight(uid, context)
                R_arm = expert_with_clus.take_action(uid, action_info, False) # rand = False
                user_choice = user_choice[R_arm]
                expert_with_clus.update(uid, i, action_info, user_choice, R_arm) ## modified here
                loss = loss_function(user_choice, R_arm)
                regret += [regret[-1] + loss]
        EWC_NO_NONCOMPLIANCE_regret.append(regret[:])
    
    if USE_EWC_NO_USER_CONTEXT:
        USE_USER_CONTEXT = False
        expert_with_clus = EWC(K, LR_EWC, max_round, USE_USER_CONTEXT=USE_USER_CONTEXT, user_contexts_train=np.load(data_path+"user_context_train.npy"), labels_train=np.load(data_path+"labels.npy")) 
        expert_with_clus.load_centers(data_path+"centroids.npy")
        env.reset()
        regret = [0]
        for i in range(max_round):
            for uid in range(num_user):
                context, action_info, user_choice = env.step(uid)
                if i == 0: expert_with_clus.init_weight(uid, context)
                R_arm = expert_with_clus.take_action(uid, action_info, False) # rand = False
                user_choice = user_choice[R_arm]
                expert_with_clus.update(uid, i, action_info, user_choice, R_arm) ## modified here
                loss = loss_function(user_choice, R_arm)
                regret += [regret[-1] + loss]
        EWC_NO_USER_CONTEXT_regret.append(regret[:])

    if USE_ONLY_USER_CONTEXT:
        USE_USER_CONTEXT = True
        expert_with_clus = EWC(K, LR_EWC, max_round, USE_USER_CONTEXT=USE_USER_CONTEXT, user_contexts_train=np.load(data_path+"user_context_train.npy"), labels_train=np.load(data_path+"labels.npy")) 
        expert_with_clus.load_centers(data_path+"centroids.npy")
        env.reset()
        regret_2 = [0]
        for i in range(max_round):
            for uid in range(num_user):
                context, action_info, user_choice = env.step(uid)
                if i == 0: expert_with_clus.init_weight(uid, context)
                R_arm = expert_with_clus.pred_only_user_context(uid, action_info)
                user_choice = user_choice[R_arm]
                loss = loss_function(user_choice, R_arm)
                regret_2 += [regret_2[-1] + loss]
        ONLY_USER_CONTEXT_regret.append(regret_2[:])
    
    if USE_LINEAR_MODEL:
        env.reset()
        regret = np.zeros((num_user, max_round))
        for uid in range(num_user):
            linear_model = Linear_model(action_context_dim)
            for i in range(max_round):
                context, action_info, user_choice = env.step(uid)
                R_arm = linear_model.take_action(action_info)
                user_choice = user_choice[R_arm]
                linear_model.update(action_info, user_choice, R_arm)
                loss = loss_function(user_choice, R_arm)
                regret[uid, i]=loss
        regret = regret.T.ravel()
        regret = np.cumsum(regret, axis=0)
        regret = np.concatenate([np.zeros(1), regret], axis=0)
        LM_regret.append(regret[:])

XGB_regret = np.concatenate([np.zeros((NUM_TEST,1)), np.array(XGB_regret)], axis=1)
result_data = pd.DataFrame()
for test in range(NUM_TEST):
    if USE_RANDOM:
        result_data[f"Random_{test}"] = random_regret[test]
    if USE_LINUCB:
        result_data[f"LinUCB_{test}"] = LinUCB_regret[test]
    if USE_EWC:
        result_data[f"EWC_{test}"] = EWC_regret[test]
    if USE_EWC_NO_NONCOMPLIANCE:
        result_data[f"EWC_NO_NONCOMPLIANCE_{test}"] = EWC_NO_NONCOMPLIANCE_regret[test]
    if ORACLE_CLUSTER:
        result_data[f"ORA_CLU_{test}"] = ORA_CLU_regret[test]
        result_data[f"ORA_THETA_{test}"] = ORA_THETA_regret[test]
    if USE_XGBOOST:
        result_data[f"XGBoost_{test}"] = XGB_regret[test, :]
    if USE_LINEAR_MODEL:
        result_data[f"linear model_{test}"] = LM_regret[test]
    if USE_DYN_UCB:
        result_data[f"DynUCB_{test}"] = DynUCB_regret[test]
    if USE_EWC_NO_USER_CONTEXT:
        result_data[f"EWC_NO_USER_CONTEXT_CLU_{test}"] = EWC_NO_USER_CONTEXT_regret[test]
    if USE_ONLY_USER_CONTEXT:
        result_data[f"ONLY_USER_CONTEXT_{test}"] = ONLY_USER_CONTEXT_regret[test]

result_data.to_csv(data_path+"result_data.csv")






        

        








        

        





        

        

