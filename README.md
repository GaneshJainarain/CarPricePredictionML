# Car Price Prediction using Machine Learning



This dataset contains information about used cars.
This data can be used for a lot of purposes such as price prediction to exemplify the use of linear regression in Machine Learning.
The columns in the given dataset are as follows:

name
year
selling_price
km_driven
fuel
seller_type
transmission
Owner

```python
import pandas as pd
df = pd.read_csv('env/Code/cardata.csv')
print(df.head())
```
![Pandas Head and Shape for dataframe](env/Code/TerminalOutput/PandasShape&Head.png)