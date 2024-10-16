import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('data/BCG.csv')
bd_df = df[df['country'] == 'Bangladesh']

def isNumber(s):
   for char in s:
    if char >= '0' and char <= '9':
        continue
    else:
        return False
   return True

years = []
v_coverages = []

for column in bd_df.columns:
    if isNumber(column):
        years.append(column)
        number = int(column)
        if number >= 2000 and number <= 3000:
            #print(bd_df[column].values[0])
            v_coverages.append(bd_df[column].values[0])

#print(years, v_coverages)
#print(bd_df.head)
print("Years Count ", len(years))
print("Vaccine Count ", len(v_coverages))


print(bd_df.head)


year_df = pd.DataFrame(data={'Year': years, 'Vaccine_Coverage': v_coverages})
year_df.to_csv('data/BCG_BD_Coverage.csv')

sns.histplot(year_df)
print(year_df.head)