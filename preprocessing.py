# %%
import pandas as pd
import os

# Define the directory path where the files are located
directory = 'data/5-labeled-sentiment/dataMaud/'

# Initialize an empty dictionary to store dataframes for each year
df_per_year = {}

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Extract year and fuel type from the filename
        year, fuel_type = filename.split('_')[0], filename.split('_')[1].split('.')[0]

        # Load the CSV file into a dataframe
        df = pd.read_csv(os.path.join(directory, filename))

        # Add a new column for fuel type
        df['fuel_type'] = fuel_type
        df['year'] = year

        # Add the dataframe to the dictionary using year as key
        if year not in df_per_year:
            df_per_year[year] = df
        else:
            # If dataframe for the year already exists, merge them
            df_per_year[year] = pd.concat([df_per_year[year], df])

# %%
# Function to check if 'labeler3' values exist in 'labels_edo' or 'labels_marin'
def check_labels(data_dict):
    results = {}
    for key, df in data_dict.items():
        # Check if 'labeler3' values are in 'labels_edo' or 'labels_marin'
        try:
            exists_in_either = df['labeler3'].isin(df['labels_edo']).any() or df['labeler3'].isin(df['labels_marin']).any()
            results[key] = exists_in_either
        except:
            print('an exception occurred')

    return results

results_dict = check_labels(df_per_year)

for key, val in results_dict.items():
    print(key, val)


# %%
df_per_year['1970s'].rename(columns={'labele3': 'labeler3'})
df_per_year['1960s'] = df_per_year['1960s'][['text', 'labels', 'fuel_type', 'year']]
# create merged dataset for labeler3
for key, df in df_per_year.items():
    try:
        df_per_year[key] = df[['text_split', 'labeler3', 'fuel_type', 'year']].rename(columns={'text_split':'text', 'labeler3':'labels'})
    except:
        print('an exception occurred')
#%%
directory = 'data/5-labeled-sentiment/labeled-full/processed/merged'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Extract year and fuel type from the filename
        year, fuel_type, _ = filename.split('_')
        # Load the CSV file into a dataframe
        df = pd.read_csv(os.path.join(directory, filename), usecols=['text', 'labels'])
        # Add a new column for fuel type
        df['fuel_type'] = fuel_type
        df['year'] = year

        # Add the dataframe to the dictionary using year as key
        if year not in df_per_year:
            df_per_year[year] = df
        else:
            merged_df = pd.merge(
                df_per_year[year], df,
                on=['text', 'fuel_type', 'year'],
                how='left', suffixes=('', '_new')
            )
            # Update labels only where values are different
            mask = (merged_df['labels_new'].notna()) & (merged_df['labels'] != merged_df['labels_new'])
            df_per_year[year].loc[mask, 'labels'] = merged_df.loc[mask, 'labels_new']
            # Drop 'labels_new' column if it exists
            if 'labels_new' in df_per_year[year].columns:
                df_per_year[year] = df_per_year[year].drop(columns=['labels_new'])


# %%
concatenated_df = pd.concat(df_per_year, ignore_index=True)
concatenated_df['labels'] = concatenated_df['labels'].astype(int)

#%%
concatenated_df['labels'].value_counts()
#%%
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Split the DataFrame into train_val and test sets
train_df, test_df = train_test_split(concatenated_df, test_size=0.2, random_state=42)

# # Split the train_val set into train and validation sets
# train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

# Convert your splits into dictionaries
train_data = {
    'text': train_df['text'],
    'label': train_df['labels'],
    'fuel_type':train_df['fuel_type'],
    'year':train_df['year'],
}

# val_data = {
#     'text': val_df['text'],
#     'label': val_df['labels'],
#     'fuel_type': val_df['fuel_type'],
#     'year':val_df['year'],
# }

test_data = {
    'text': test_df['text'],
    'label': test_df['labels'],
    'fuel_type': test_df['fuel_type'],
    'year':test_df['year'],
}

# Convert each split into a Dataset
train_dataset = Dataset.from_dict(train_data)
# val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Store the datasets in a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    # 'validation': val_dataset,
    'test': test_dataset
})


#%%
import os
import pickle

dataset_dict.save_to_disk(".")
