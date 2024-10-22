import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_excel("data/final_dataset_with_dtp.xlsx")

dtp3_df = pd.read_excel("data/dtp_final.xlsx")


df = df.rename(columns={'Country_Name_String': 'country', 'Year': 'year'})

merged_df = pd.merge(df, dtp3_df, on=['country', 'year'], how='inner')

rows_with_missing_values = df.isnull().any(axis=1).sum()


#merged_df = merged_df.dropna() 
merged_df = merged_df.rename(columns={'DTP': 'DTP1', 'country': 'Country', 'year':'Year'})
merged_df = merged_df.drop(labels=['vaccine'], axis=1)

print(merged_df.head())

merged_df.to_excel('data/final_dataset_with_dtp.xlsx', index=False)