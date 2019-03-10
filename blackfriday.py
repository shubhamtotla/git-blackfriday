#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#importing black friday data in python
df = pd.read_csv("D:\\train.csv",engine='python')
test_df = pd.read_csv("D:\\test.csv",engine='python')
#Quick data exploration
df.head(10)
df.shape
df.describe()
#DATA CLEANING
#Checking for missing values
df.isnull().sum()
df.isnull().sum().sum()
#Removing missing values
df.Product_Category_2.fillna(df.Product_Category_2.mean(),inplace=True)
df.Product_Category_3.fillna(df.Product_Category_3.mean(),inplace=True)
#Converting into numerical type
X=df
from sklearn.preprocessing import LabelEncoder
a=['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']
LE = LabelEncoder()
for i in a:
    X[i] = LE.fit_transform(X[i])
X.dtypes

#Data Exploration
#Gender vs Purchase
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.bar(X['Gender'],df['Purchase'])
plt.title('Gender vs Purchase')
plt.xlabel('Gender')
plt.ylabel('Purchase')
plt.show()
#Age vs Purchase
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.bar(X['Age'],df['Purchase'])
plt.title('Age vs Purchase')
plt.xlabel('Age')
plt.ylabel('Purchase')
plt.show()
#City vs Purchase
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.bar(X['City_Category'],df['Purchase'])
plt.title('City vs Purchase')
plt.xlabel('City')
plt.ylabel('Purchase')
plt.show()
#Heatmap to check correlation
corrmat = df.corr()
fig,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax=.8, square=True)
#Model prediction
#Importing test dataset and cleaning data
test_df = pd.read_csv("D:\\test.csv",engine='python')
test_df.Product_Category_2.fillna(df.Product_Category_2.mean(),inplace=True)
test_df.Product_Category_3.fillna(df.Product_Category_3.mean(),inplace=True)
Y=test_df
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for i in a:
    Y[i] = LE.fit_transform(Y[i])
#ALGORITHM
#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn.model_selection import cross_val_score
from sklearn import metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    
    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors],(dtrain[target]) , cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((dtrain[target]).values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    
#Define target and ID columns:
target = 'Purchase'
IDcol = ['User_ID','Product_ID']
#Linear Regression Model
from sklearn.linear_model import LinearRegression
LR = LinearRegression(normalize=True)
predictors = X.columns.drop(['Purchase','Product_ID','User_ID'])
modelfit(LR, X, Y, predictors, target, IDcol, 'LR.csv')
coef1 = pd.Series(LR.coef_, predictors)
coef1.plot(kind='bar', title='Model Coefficients')
#Decision Tree 
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
predictors = X.columns.drop(['Purchase','Product_ID','User_ID'])
modelfit(DT, X, Y, predictors, target, IDcol, 'DT.csv')
#Random Forest
from sklearn.tree import DecisionTreeRegressor
RF = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
predictors = X.columns.drop(['Purchase','Product_ID','User_ID'])
modelfit(RF, X, Y, predictors, target, IDcol, 'RF.csv')
