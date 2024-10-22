import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_excel('data/K_means.xlsx')

X = df['X']
Y = df['Y']


sns.regplot(x=X, y=Y, fit_reg=False)

kmeans = KMeans(n_clusters=3, max_iter=300, random_state=0)
model = kmeans.fit(df)
predicted_result = model.predict(df)

plt.scatter(X, Y, c = predicted_result, s = 50, cmap = 'viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', alpha=0.5)
plt.show()