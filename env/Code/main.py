import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('env/Code/cardata.csv')
print("First Five Instances of Dataframe:")
print(df.head())
print("Shape of Our Dataframe:")
print(df.shape)

'''
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())

##check missing values
print(df.isnull().sum())

print(df.describe())

final_dataset = df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
print(final_dataset.head())

final_dataset['Current Year'] = 2022
print(final_dataset.head())

final_dataset['car_age'] = final_dataset['Current Year'] - final_dataset['Year']
print(final_dataset.head())

final_dataset.drop(['Year'],axis=1,inplace=True)
print(final_dataset.head())

final_dataset=pd.get_dummies(final_dataset,drop_first=True)
print(final_dataset.head())

final_dataset=final_dataset.drop(['Current Year'],axis=1)
print(final_dataset.head())
print(final_dataset.corr())


#sns.pairplot(final_dataset)
#get correlations of each features in dataset
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
#plt.figure(figsize=(10,10))
#plot heat map
print(sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn"))
plt.show()

#Selling Price is the dependent Feature everything else is an Independent Feature
X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]

print(X.head()) 
print(y.head())
'''