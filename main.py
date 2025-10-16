import pandas as pd
dataset = pd.read_csv('games.csv')
dataset

dataset.dtypes

dataset.shape

100 * dataset.isnull().sum() / len(dataset)

# Descriptive Statistical Summary
dataset.describe().round(2)

# Drop columns with >50% missing values
missing_pct = dataset.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 50].index
print(f"Columns to drop (>50% missing): {list(cols_to_drop)}")

data = dataset.drop(columns=cols_to_drop)


data = data.dropna()

data.dtypes

