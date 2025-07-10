import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--beta_scaler', type=int, default=0, help='Scaler for beta parameters')
args = parser.parse_args()
beta_scaler = args.beta_scaler

df = pd.read_csv("processed_data.csv")
user_context = pd.read_csv("user_context_processed.csv")
lamda = 4
max_round_gen = 40
max_round_survey = 6
num_sample_each_user = 24
mean = [-0.1, -0.1]
covariance = [[0.01, 0], 
              [0, 0.01]] 
beta_parameters = [0.3, 0.3]
data_path = f"../data/beta={beta_scaler}/"


TT_context = np.array([
    10,
    10,
    10,
    10,
    10.5,
    11,
    11.5])*10
CO2_context = np.array([
    100,
    86.95652174,
    91.30434783,
    95.65217391,
    90,
    90,
    90
    ])

def softmax(x, T=1):
    x/=T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(y_true, y_pred):
    # Ensure the prediction values are in the range (0,1)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Calculate cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred))

    return loss

def gen_weight_for_id(id_data):
    '''
    Assume w_rec ~ beta(alpha, beta)*10
    '''
    synthetic_weight = []
    loss_list = []
    y = id_data["response"]
    action_context = id_data[["TT", "CO2"]]
    num_sample = 0
    while num_sample < num_sample_each_user:
        w = np.random.multivariate_normal(mean, covariance)
        loss = 0
        for t in range(max_round_survey):
            Utility = np.array([
                
                w @ np.array([action_context.iloc[t, 0]*10, action_context.iloc[t, 1]]),
                w @ np.array([100, 100])
                
            ])
            y_hat = softmax(Utility)
            loss += cross_entropy_loss(np.array([y.iloc[t], 1-y.iloc[t]])
                                       , y_hat)
        loss /=max_round_survey
        p_accept = np.exp(-1 * lamda * loss)
        if np.random.random() < p_accept: 
            w_rec = np.random.beta(beta_parameters[0], beta_parameters[1]) * beta_scaler
            synthetic_weight.append(np.append(w, w_rec))
            loss_list.append(loss)
            num_sample+=1
    return synthetic_weight, loss_list

def gen_compliance_for_weight(w, id, weight_id):
    synthetic_data = pd.DataFrame()
    for t in range(max_round_gen):
        TT = np.random.normal(loc=TT_context.mean(), scale=TT_context.std())  # Assuming normal distribution
        CO2 = np.random.normal(loc=CO2_context.mean(), scale=CO2_context.std())  # Assuming normal distribution
        rec = np.random.choice([0,1], p =[0.5, 0.5])
        Utility = np.array([
            w @ np.array([100, 100, int(rec==0)]),
            w @ np.array([TT, CO2, int(rec==1)])
        ])
        y_hat = softmax(Utility)
        compliance = 0 if y_hat[0] > 0.5 else 1
        synthetic_row = pd.DataFrame({
            'ID': str(id) + '_' + str(weight_id),
            'Age_18 - 34': user_context.iloc[id-1, 1],
            'Age_35 - 49': user_context.iloc[id-1, 2],
            'Age_50 - 64': user_context.iloc[id-1, 3],
            'Gender_Male': user_context.iloc[id-1, 4],
            'Education_Bachelor\'s degree': user_context.iloc[id-1, 5],
            'Education_High school diploma or equivalent': user_context.iloc[id-1, 6],
            'Education_Master\'s degree or higher': user_context.iloc[id-1, 7],
            'num_cars_One car': user_context.iloc[id-1, 8],
            'w_TT': w[0],
            'w_CO2': w[1],
            'w_Rec': w[2],
            'TT': TT,
            'CO2': CO2,
            'Rec': rec,
            'compliance': compliance
        }, index=[0])
        synthetic_data = pd.concat([synthetic_data, synthetic_row], ignore_index=True)
    return synthetic_data

def create_new_dataframe(df):
    # Create a new empty dataframe
    new_df = pd.DataFrame()
    print(len(df))

    # Iterate through the original dataframe columns
    for col in df.columns:
        if 'TT_' in col or 'CO2_' in col:
            # Extract index i from the column name
            i = col.split('_')[-1]

            # Check if it's a TT or CO2 column and add new columns accordingly
            if 'TT_' in col:
                new_df[f'TT_{i}0'] = [100] * len(df)
                new_df[f'TT_{i}1'] = df[col]
            elif 'CO2_' in col:
                new_df[f'CO2_{i}0'] = [100] * len(df)
                new_df[f'CO2_{i}1'] = df[col]
        else:
            # Copy other columns as is
            new_df[col] = df[col]

    return new_df

def filter_data(data):
    compliance_columns = [f'compliance_{i}' for i in range(max_round_gen)]
    mean_values = data[compliance_columns].mean(axis = 1)
    return data[mean_values < 0.99][mean_values > 0.01]

if __name__ == "__main__":
    ############ Generate synthetic data ############
    print(f"Mean of TT is {TT_context.mean()}, std is {TT_context.std()}")
    print(f"Mean of CO2 is {CO2_context.mean()}, std is {CO2_context.std()}")
    synthetic_data = pd.DataFrame()
    weights=[]
    losses = []
    for id in tqdm(df["ID"].unique()):
        id_data = df[df["ID"]==id]
        synthetic_weight, loss_list = gen_weight_for_id(id_data)
        weights.append(synthetic_weight)
        losses.append(loss_list)
        for weight_id in range(len(synthetic_weight)):
            weight = synthetic_weight[weight_id]
            synthetic_data_for_id = gen_compliance_for_weight(weight, id, weight_id)
            synthetic_data = pd.concat([synthetic_data, synthetic_data_for_id], ignore_index=True)
    np.save(data_path+"weights.npy", np.array(weights))
    
    ############ Rearrange ############
    data = synthetic_data
    reshaped_data = pd.DataFrame()

    # Define non-time series columns
    non_ts_columns = ['Age_18 - 34', 'Age_35 - 49', 'Age_50 - 64', 'Gender_Male', 
                    'Education_Bachelor\'s degree', 
                    'Education_High school diploma or equivalent', 
                    'Education_Master\'s degree or higher', 
                    'num_cars_One car',
                    'w_TT',
                    'w_CO2',
                    'w_Rec'
                    ]

    # Iterating over each group and reshaping
    for id, group in tqdm(data.groupby('ID')):
        num_rows = len(group) // max_round_gen
        for i in range(num_rows):
            subset = group.iloc[i*max_round_gen:(i+1)*max_round_gen]
            reshaped_series = pd.Series()
            reshaped_series['ID'] = id

            # Interleaving TT, CO2, and compliance columns
            for j in range(max_round_gen):
                reshaped_series[f'TT_{j}'] = subset['TT'].iloc[j]
                reshaped_series[f'CO2_{j}'] = subset['CO2'].iloc[j]
                reshaped_series[f'Rec_{j}'] = subset['Rec'].iloc[j]
                reshaped_series[f'compliance_{j}'] = subset['compliance'].iloc[j]
            reshaped_data = pd.concat([reshaped_data, reshaped_series.to_frame().T], ignore_index=True)
    
    # Adding the non-time series columns
    for col in non_ts_columns:
        reshaped_data[col] = reshaped_data['ID'].map(data.groupby('ID')[col].first())

    # Reordering columns to match requested format
    time_series_columns = [f'{var}_{i}' for i in range(max_round_gen) for var in ['TT', 'CO2', 'Rec', 'compliance']]
    column_order_final = non_ts_columns + time_series_columns
    reshaped_data = reshaped_data[column_order_final]
    reshaped_data = create_new_dataframe(reshaped_data)
    reshaped_data.to_csv(data_path+"synthetic_data.csv")