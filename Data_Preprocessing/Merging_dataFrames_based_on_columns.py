import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_excel("data/final_dataset_with_dtp.xlsx")

dtp3_df = pd.read_excel("data/dtp3_final.xlsx")


dtp3_df = dtp3_df.rename(columns={'country': 'Country', 'year': 'Year'})

merged_df = pd.merge(df, dtp3_df, on=['Country', 'Year'], how='inner')

rows_with_missing_values = df.isnull().any(axis=1).sum()


#merged_df = merged_df.dropna() 

merged_df = merged_df.drop(labels=['vaccine', 'unicef_region', 'iso3'], axis=1)


print(merged_df.head())

merged_df.to_excel('data/final_dataset_with_dtp3.xlsx', index=False)