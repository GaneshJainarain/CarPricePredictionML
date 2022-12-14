import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


df = pd.read_csv('env/Code/cardata.csv')
#print("First Five Instances of Dataframe:")
#print(df.head())
#print("Shape of Our Dataframe:")
#print(df.shape)

#print(df['Seller_Type'].unique())
#print(df['Fuel_Type'].unique())
#print(df['Transmission'].unique())
#print(df['Owner'].unique())


##check missing values

#print("Checking for missing values")

#print(df.isnull().sum())


#print(df.describe())

final_dataset = df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]

#print(final_dataset.head())

final_dataset['Current Year'] = 2022

#print(final_dataset.head())

final_dataset['car_age'] = final_dataset['Current Year'] - final_dataset['Year']

#print(final_dataset.head())

final_dataset.drop(['Year'],axis=1,inplace=True)

#print(final_dataset.head())

final_dataset.drop(['Current Year'],axis=1,inplace=True)
#print(final_dataset.head())


final_dataset=pd.get_dummies(final_dataset,drop_first=True)

print(final_dataset.head())



print(final_dataset.corr())


#sns.pairplot(final_dataset)
#get correlations of each features in dataset
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
#print(sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn"))
#plt.show()

#Selling Price is the dependent Feature everything else is an Independent Feature

X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]

#print("Our Independent Features:")
#print(X.head())
#print("Our Dependent Feature:")
#print(y.head())


model = ExtraTreesRegressor()
model.fit(X,y)
#print(model.feature_importances_)

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
#plt.show() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
regressor = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': [True, False]}

pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, 
param_distributions = random_grid,
scoring='neg_mean_squared_error', 
n_iter = 10, cv = 5, verbose=2, 
random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
print("#####")
pprint(rf_random.best_params_)
