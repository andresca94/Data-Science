import pandas as pd
import matplotlib.pyplot as plt
pumpkins = pd.read_csv('../Regression/US-pumpkins.csv')
pumpkins.head()


pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

#Check if there is missing data in the current dataframe:

pumpkins.isnull().sum()

###To make your dataframe easier to work with, drop several of its columns, using drop(), keeping only the columns you need:

new_columns = ['Package', 'Month', 'Low Price', 'High Price', 'Date']
pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)

#Determine average price of pumpkin

price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

month = pd.DatetimeIndex(pumpkins['Date']).month

#Converted data into a fresh Pandas dataframe and normalize data

new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})

new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)

#Visualization

price = new_pumpkins.Price
month = new_pumpkins.Month
plt.scatter(price, month)
plt.show()

new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
plt.ylabel("Pumpkin Price")
