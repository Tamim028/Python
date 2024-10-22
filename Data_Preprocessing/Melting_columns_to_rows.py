#melting Columns into rows...

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

dtp3_df = pd.read_excel("data/raw_vaccine_data/POL3_vaccine.xlsx")

dtp3_melted = dtp3_df.melt(id_vars=["unicef_region", "iso3", "country", "vaccine"], 
                    var_name="Year", value_name="POL3")

# Convert the 'year' column to integer if necessary
dtp3_melted['Year'] = dtp3_melted['Year'].astype(int)


dtp3_melted.to_excel('data/vaccine_final/pol3_final.xlsx', index=None)

print('TOTAL DATA: ', len(dtp3_melted))
print(dtp3_melted.head())
