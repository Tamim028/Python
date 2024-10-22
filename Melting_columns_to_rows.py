#melting Columns into rows...

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

dtp3_df = pd.read_excel("data/DTP3_vaccine_2000_2023.xlsx")

dtp3_melted = dtp3_df.melt(id_vars=["unicef_region", "iso3", "country", "vaccine"], 
                    var_name="year", value_name="DTP3")

# Convert the 'year' column to integer if necessary
dtp3_melted['year'] = dtp3_melted['year'].astype(int)


dtp3_melted.to_excel('data/dtp3_final.xlsx', index=None)

print(dtp3_melted.head())
