import pandas as pd
import random
import argparse

def randomize_data(csv_file, p, output_file='data_randomized.csv'):
    """
    Reads data from csv_file, randomly replaces age, gender, education, and num_cars
    values in each row with a probability p, and saves the results to output_file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Define column names (make sure these match your CSV columns)
    age_cols = ['Age_18 - 34', 'Age_35 - 49', 'Age_50 - 64']
    gender_col = 'Gender_Male'
    edu_cols = [
        "Education_Bachelor's degree",
        "Education_High school diploma or equivalent",
        "Education_Master's degree or higher"
    ]
    num_cars_col = 'num_cars_One car'
    
    # Process each row
    for idx, row in df.iterrows():
        # 1) Randomly replace age information with a probability p
        if random.random() < p:
            new_age = random.choice(age_cols)       # Randomly choose one age category
            df.loc[idx, age_cols] = 0              # Set all age columns to 0
            df.loc[idx, new_age] = 1               # Set the chosen age column to 1
        
        # 2) Randomly replace gender information with a probability p
        if random.random() < p:
            df.loc[idx, gender_col] = random.choice([0, 1])  # Gender can be 0 or 1
        
        # 3) Randomly replace education level information with a probability p
        if random.random() < p:
            new_edu = random.choice(edu_cols)       # Randomly choose one education level
            df.loc[idx, edu_cols] = 0               # Set all education columns to 0
            df.loc[idx, new_edu] = 1                # Set the chosen education column to 1
        
        # 4) Randomly replace number of cars with a probability p
        if random.random() < p:
            df.loc[idx, num_cars_col] = random.choice([0, 1])  # Choose either 0 or 1
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Randomized data saved to: {output_file}")
parser = argparse.ArgumentParser()
parser.add_argument("--prob_noise", type=float, help="Probability of noise", default=0.1)
args = parser.parse_args()
P=args.prob_noise
randomize_data(csv_file=f'../../data/beta=1/rearranged_synthetic_data.csv', p=P, output_file=f'p={P}/rearranged_synthetic_data.csv')