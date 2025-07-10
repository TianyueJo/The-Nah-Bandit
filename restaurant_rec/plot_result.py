import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--EXPERIMENT_NAME", type=str, help="Name of the experiment", default="default")
parser.add_argument("--d", type=float, help="Dimension for PCA", default=4)
args = parser.parse_args()

ABLA = args.EXPERIMENT_NAME == "ablation_study"
PCA = args.EXPERIMENT_NAME == "PCA"
file_path = "data/"
YLIM = 860
bound = 1034
K=8
if PCA:
    d = args.d
    file_path = f"additional_experiment/PCA/d={d}/"
df = pd.read_csv(file_path+f"result_data.csv")

# First, let's drop the unnamed index column as it's not needed for plotting
df = df.drop(df.columns[0], axis=1)

# Now let's melt the DataFrame so that we have the 'Round', 'Algorithm', and 'Regret' format
df_melted = df.reset_index().melt(id_vars=['index'], var_name='Algorithm', value_name='Regret')
df_melted.rename(columns={'index': 'Round'}, inplace=True)

# Split the 'Algorithm' column to separate the algorithm names from the test indices
df_melted['Algorithm'] = df_melted['Algorithm'].str.rsplit('_', n=1).str[0]
df_melted['Test'] = df_melted['Algorithm'].str.rsplit('_', n=1).str[1]
# df_melted['Algorithm'], df_melted['Test'] = df_melted['Algorithm'].str.rsplit('_').str

# Now, let's group by 'Round' and 'Algorithm' to calculate the mean and 95% confidence interval
df_grouped = df_melted.groupby(['Round', 'Algorithm'])['Regret'].agg([np.mean, lambda x: np.std(x, ddof=1) * 1.96 / np.sqrt(len(x))])
df_grouped.columns = ['Mean Regret', '95% CI']
df_grouped = df_grouped.reset_index()

color_palette = {
    'Random': 'black',
    'FTL': 'blue',
    'LinUCB': 'limegreen',
    'EWC': 'red',
    'ORA_CLU': 'firebrick',
    'ORA_THETA': 'darkviolet',
    'XGBoost': 'darkturquoise',
    'linear model': 'violet',
    'DynUCB': 'darkgreen',
    "EWC_NO_USER_CONTEXT_CLU": 'orange',
    "ONLY_USER_CONTEXT": 'darkorange',
    "EWC_NO_NONCOMPLIANCE": 'blue',
}

# line_styles = {
#     'Random': '-',
#     'FTL': '-',
#     'LinUCB': '-',
#     'EWC': '-',
#     'ORA_CLU': '-',
#     'ORA_THETA': '-',
#     'XGBoost': '-',
#     'linear model': '-',
#     'DynUCB': '-',
#     "EWC_NO_USER_CONTEXT_CLU": '-',
#     "ONLY_USER_CONTEXT": '-',
#     "EWC_NO_NONCOMPLIANCE": '-',
# }
line_styles = {
    'Random': ':', 
    'FTL': (0, (5, 10)),  
    'LinUCB': '-.',      
    'EWC': '-',                
    'ORA_CLU': (0, (5, 1)),        
    'ORA_THETA': (0, (3, 5, 1, 5)),
    'XGBoost': (0, (1, 1)),   
    'linear model': "--",  
    'DynUCB': (0, (3, 1, 1, 1)),  
    "EWC_NO_USER_CONTEXT_CLU": (0, (1, 2, 2, 2)), 
    "ONLY_USER_CONTEXT": (0, (5, 2, 1, 2)),      
    "EWC_NO_NONCOMPLIANCE": (0, (2, 1)),         
}
labels = {
    'Random': 'Random',
    'FTL': 'FTL',
    'LinUCB': 'LinUCB',
    'EWC': 'EWC (Ours)',
    'ORA_CLU': 'Oracle cluster',
    'ORA_THETA': r'Oracle $\theta_i$',
    'XGBoost': 'XGBoost',
    'linear model': 'Non-compliance',
    'DynUCB': 'DYNUCB',
    "EWC_NO_USER_CONTEXT_CLU": r'Without $u_i$',
    "ONLY_USER_CONTEXT": r'LR on $u_i$',
    "EWC_NO_NONCOMPLIANCE": 'Without non-compliance',
}



# Create the figure and axis
plt.figure(figsize=(12, 8))

legend_handles = []  # List to store legend handles

algs1 = [
    'DynUCB',
    'LinUCB',
    'linear model',
    'XGBoost',
    'EWC'
    ]

# ablation study

algs2 =[
    'FTL',
    'linear model', # without expert and clustering
    "EWC_NO_USER_CONTEXT_CLU", # without user context
    'EWC',
    'ORA_CLU', # oracle cluster identity
    'ORA_THETA', # oracle theta for each user
]

algs = algs2 if ABLA else algs1
# Loop through each algorithm to plot
for algorithm in algs:
    subset = df_grouped[df_grouped['Algorithm'] == algorithm][:bound]
    
    # Get the color and line style for the algorithm
    alg_color = color_palette[algorithm]
    alg_line_style = line_styles[algorithm]

    # Plot the mean regret
    plt.plot(subset['Round'], subset['Mean Regret'], label=labels[algorithm], color=alg_color, linestyle=alg_line_style, lw=3.0)

    # Add the confidence interval as a shaded area
    plt.fill_between(subset['Round'], subset['Mean Regret'] - subset['95% CI'], subset['Mean Regret'] + subset['95% CI'], color=alg_color, alpha=0.3)

    # Create a custom legend handle
    legend_handle = plt.Line2D([], [], color=alg_color, linestyle=alg_line_style, label=labels[algorithm])
    legend_handles.append(legend_handle)
SIZE=22
# Add title, labels, and custom legend
plt.xlabel('Total decision rounds', fontsize=SIZE)
plt.ylabel('Regret', fontsize=SIZE)
plt.xticks(fontsize=SIZE)
plt.yticks(fontsize=SIZE)
plt.ylim(top=YLIM)
legend = plt.legend(handles=legend_handles, fontsize=SIZE)
for legobj in legend.legendHandles:
    legobj.set_linewidth(3)
plt.grid(True)
if PCA:
    plt.savefig(f"result/restaurant_PCA_d={d}.png", format = "png")
else:
    plt.savefig(f"result/restaurant.png", format = "png")
plt.show()