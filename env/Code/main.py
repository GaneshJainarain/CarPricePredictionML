import pandas as pd

df = pd.read_csv('env/Code/cardata.csv')
print(df.shape)

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





