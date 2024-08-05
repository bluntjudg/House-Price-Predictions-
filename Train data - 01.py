import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("train.csv")

print(df.columns)

print(df.head)
df.isnull().sum()
# information about the dataset
df.info()
df.isnull().sum()
df.shape

df=df[['LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea','FullBath','HalfBath','BedroomAbvGr','GarageArea','SaleCondition','SalePrice']]
df['SaleCondition']=df['SaleCondition'].replace({'Normal':1, 'Abnorml':2, 'Partial':3, 'AdjLand':4, 'Alloca':5, 'Family':6})
m=df.corr()
print(m)
mat=df.corr()
sns.heatmap(mat)
plt.show()
from sklearn.model_selection import train_test_split

x=df[['LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea','FullBath','HalfBath','BedroomAbvGr','GarageArea','SaleCondition','SalePrice']]
y=df['SalePrice']


print(x)

print(y)

x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,train_size=0.95)

print(x_tr)
print(y_tr)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_tr, y_tr)
pred = model.predict(x_ts)
print('actual out:', list(y_ts))
print('predict out:', pred)
print('accuracy:', model.score(x_tr, y_tr))
