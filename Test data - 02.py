import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv("test.csv")

# Print the column names to verify the available columns
print("Columns in the DataFrame:", df.columns)

# Select relevant columns (excluding 'SalePrice' as it is not present)
df = df[['LotArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'GarageArea', 'SaleCondition']]

# Replace values in 'SaleCondition'
df['SaleCondition'] = df['SaleCondition'].replace({'Normal': 1, 'Abnorml': 2, 'Partial': 3, 'AdjLand': 4, 'Alloca': 5, 'Family': 6})

# Compute the correlation matrix
mat = df.corr()
print(mat)

# Plot the heatmap
sns.heatmap(mat)
plt.show()

from sklearn.model_selection import train_test_split

# We'll use 'SaleCondition' as the target variable for demonstration purposes
x = df[['LotArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'GarageArea']]
y = df['SaleCondition']

print(x)
print(y)

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.95)

print(x_tr)
print(y_tr)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_tr, y_tr)
pred = model.predict(x_ts)
print('actual out:', list(y_ts))
print('predict out:', pred)
print('accuracy:', model.score(x_tr, y_tr))
