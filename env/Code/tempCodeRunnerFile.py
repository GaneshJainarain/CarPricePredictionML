final_dataset = df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
print(final_dataset.head())

final_dataset['Current Year'] = 2022
print(final_dataset.head())

final_dataset['car_age'] = final_dataset['Current Year'] - final_dataset['Year']
print(final_dataset.head())

final_dataset.drop(['Year'],axis=1,inplace=True)
print(final_dataset.head())