import pandas as pd
import numpy as np


df = pd.read_parquet('closeprice.parquet')

df_recent = df.tail(2000)

df_recent_clean = df_recent.dropna(axis=1)
df_recent_clean = df_recent_clean.iloc[:, :1000]

log_returns = np.log(1 + df_recent_clean.pct_change(1))
# Fill NaN values with forward fill
log_returns = log_returns.fillna(method='bfill')

print(log_returns.isna().sum().sum())

from scipy.stats import zscore

z_scores = log_returns.apply(zscore)

standardized_log_returns = z_scores

#shuffled_standardized_log_returns = standardized_log_returns.apply(np.random.permutation)



H = standardized_log_returns.values
H = np.array(H)

W = (1/2000)* np.transpose(H).dot(H)

eigenvalues = np.linalg.eigvals(W)

print(max(eigenvalues))
import matplotlib.pyplot as plt

plt.hist(eigenvalues, bins=50, range=(324,325), density=True)
plt.show()



# Fill NaN values with forward fill
