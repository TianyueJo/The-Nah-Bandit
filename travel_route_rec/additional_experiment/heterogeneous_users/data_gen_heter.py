import numpy as np
import pandas as pd
from tqdm import tqdm
weights=np.load("../../data/beta=0/weights.npy")
weights_reshaped = np.reshape(weights, [-1, weights.shape[2]])
max_weight = np.max(weights_reshaped, axis=0)
min_weight = np.min(weights_reshaped, axis=0)
context_range = [[min_weight[0], max_weight[0]], [min_weight[1], max_weight[1]], [0.3, 0.3]]
N = weights.shape[0] * weights.shape[1]
weights_heter = np.zeros([N,0])
for d in range(3):
    context_d = np.random.uniform(low=context_range[d][0], high=context_range[d][1], size=N)
    if d==2:
        context_d = np.random.beta(context_range[d][0], context_range[d][1], size=N)
    weights_heter = np.concatenate([weights_heter, context_d[:, np.newaxis]], axis=1)
np.save("weights_heter.npy", weights_heter)
df = pd.read_csv("../../data_generation/processed_data.csv")
user_context = pd.read_csv("../../data_generation/user_context_processed.csv")
max_round_survey = 6
max_round_gen = 160 # default is 40
lamda = 4 # Higher lambda means synthetic data is more similar to original data, but generated more slowly
mean = [-0.1, -0.1]
covariance = [[0.01, 0], 
              [0, 0.01]] 
beta_parameters = [0.3, 0.3]
# beta_scaler = 0 # beta measures how user's choice depends on the recommendation
num_sample_each_user = 24 # default is 48

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
    new_columns = []
    column_names = []
    for col in df.columns:
        if 'TT_' in col or 'CO2_' in col:
            # Extract index i from the column name
            i = col.split('_')[-1]

            if 'TT_' in col:
                new_columns.append(pd.Series([100] * len(df)))
                column_names.append(f'TT_{i}0')

                new_columns.append(df[col])
                column_names.append(f'TT_{i}1')

            elif 'CO2_' in col:
                new_columns.append(pd.Series([100] * len(df)))
                column_names.append(f'CO2_{i}0')

                new_columns.append(df[col])
                column_names.append(f'CO2_{i}1')

        else:
            new_columns.append(df[col])
            column_names.append(col)
    new_df = pd.concat(new_columns, axis=1)
    new_df.columns = column_names

    return new_df

############ Generate synthetic data ############
synthetic_data = pd.DataFrame()
weights_heter = np.reshape(weights_heter, weights.shape)
print(weights_heter.shape)
for id in tqdm(range(1, weights.shape[0]+1)):
    id_data = df[df["ID"]==id]
    synthetic_weight = weights_heter[id-1, :, :]
    for weight_id in range(len(synthetic_weight)):
        weight = synthetic_weight[weight_id]
        synthetic_data_for_id = gen_compliance_for_weight(weight, id, weight_id)
        synthetic_data = pd.concat([synthetic_data, synthetic_data_for_id], ignore_index=True)

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
        reshaped_series = pd.Series(dtype='float64')
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

############ Generate multiple options ############
# Function to create a new dataframe with the desired structure
df = reshaped_data
# Create the new dataframe
new_dataframe = create_new_dataframe(df)
new_dataframe.to_csv('synthetic_data.csv', index=False)