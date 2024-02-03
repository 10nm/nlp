

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

# df.to_pickle('scaled.pkl')
df = pd.read_pickle('scaled.pkl')

print(df)

correlation = df['scaled_list'].corr(df['stored_score'])
print(correlation)