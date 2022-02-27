
#DATA CLEANING

#Data loading through the read_csv function

#The code below presents the titanic passenger dataset, load the data, perform some preliminary analysis and clean the columns with missing data. Dummies" variables will be added to the dataset representing categorical data.

import pandas as pd
import os
import numpy as np
import random

mainpath = "/Users/andrescarvajal/Desktop/Portfolio/dataCleaning/"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)

data = pd.read_csv(fullpath)

data.head(10)

data.tail(8)

data.shape

data.columns.values

data.describe()

data.dtypes


# Missing values

pd.isnull(data["body"])

pd.notnull(data["body"])

pd.isnull(data["body"]).values.ravel().sum()

pd.notnull(data["body"]).values.ravel().sum()

# Missing values in a data set can occur for two reasons:

#Data extraction
#Data collection

#Deleting missing values

data.dropna(axis=0, how="all")

data2 = data

data2.dropna(axis=0, how="any")

#Calculation of missing values

data3 = data

data3["body"] = data3["body"].fillna(0) #Change the NaN values to 0
data3["home.dest"] = data3["home.dest"].fillna("Unknown") #Change the NaN values to "Unknown"
data3.head(5)

pd.isnull(data3["age"]).values.ravel().sum()

data3["age"].fillna(data["age"].mean())

data3["age"].fillna(method="ffill")

data3["age"].fillna(method="backfill")

data3

#Dummy variables

data["sex"].head(10)

dummy_sex = pd.get_dummies(data["sex"], prefix="sex")

dummy_sex.head(10)

column_name=data.columns.values.tolist()
column_name

data = data.drop(["sex"], axis = 1)

data = pd.concat([data, dummy_sex], axis = 1)

def createDummies(df, var_name):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis = 1)
    df = pd.concat([df, dummy ], axis = 1)
    return df

createDummies(data3, "sex")


#Data Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data2 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", sep=",")
data3 = data2

data2.head()

data2.columns.values

data_cols = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Columns.csv")
data_col_list = data_cols["Column_Names"].tolist()
data2 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", header = None, names = data_col_list)
data2.columns.values

get_ipython().run_line_magic('matplotlib', 'inline')


#Scatter Plot

data3.plot(kind="scatter", x="Day Mins", y="Day Charge")

data3.plot(kind="scatter", x="Night Mins", y="Night Charge")

figure, axs = plt.subplots(2,2, sharey=True, sharex=True)
data3.plot(kind="scatter", x="Day Mins", y ="Day Charge", ax=axs[0][0])
data3.plot(kind="scatter", x="Night Mins", y="Night Charge", ax=axs[0][1])
data3.plot(kind="scatter", x="Day Calls", y ="Day Charge", ax=axs[1][0])
data3.plot(kind="scatter", x="Night Calls", y="Night Charge", ax=axs[1][1])


#Frequency histogram

k = int(np.ceil(1+np.log2(3333)))
plt.hist(data3["Day Calls"], bins = k) #bins = [0,30,60,...,200]
plt.xlabel("Number of calls per day")
plt.ylabel("Frequency")
plt.title("Histogram of the number of calls per day")


#Boxplot

plt.boxplot(data3["Day Calls"])
plt.ylabel("Number of calls per day")
plt.title("Boxplot of daily calls")

data3["Day Calls"].describe()

IQR=data3["Day Calls"].quantile(0.75)-data3["Day Calls"].quantile(0.25)
IQR

data3["Day Calls"].quantile(0.25) - 1.5*IQR

data3["Day Calls"].quantile(0.75) + 1.5*IQR


#Data loading through the open function

data3 = open(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt",'r')

cols = data3.readline().strip().split(",")
n_cols = len(cols)

counter = 0
main_dict = {}
for col in cols:
    main_dict[col] = []

for line in data3:
    values = line.strip().split(",")
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1

print("El data set tiene %d filas y %d columnas"%(counter-1, n_cols))

df3 = pd.DataFrame(main_dict)
df3.head()


#Data Wrangling
#Create a subset of data
#Subset of column or columns

df4 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", sep=",")
df4

account_length = df4["Account Length"]

account_length.head()

type(account_length)

subset = df4[["Account Length", "Phone", "Eve Charge", "Day Calls"]]

type(subset)

desired_columns = ["Account Length", "Phone", "Eve Charge", "Night Calls"]
subset = df4[desired_columns]
subset.head()

desired_columns = ["Account Length", "VMail Message", "Day Calls"]
desired_columns

all_columns_list = df4.columns.values.tolist()
all_columns_list

np.random.choice(all_columns_list) #Choose column value randomly 

sublist = [x for x in all_columns_list if x not in desired_columns]
sublist

subset = df4[sublist]
subset.head()


#Subset of rows

df3[1:25]


#Subsets of rows with Boolean conditions

#Users with Day Mins > 300
data1 = df4[df4["Day Mins"]>300]
data1.shape

#New York Users (State = "NY")
data2 = df4[df4["State"]=="NY"]
data2.shape

##AND -> &
data3 = df4[(df4["Day Mins"]>300) & (df4["State"]=="NY")]
data3.shape

##OR -> |
data4 = df4[(df4["Day Mins"]>300) | (df4["State"]=="NY")]
data4.shape

data5 = df4[df4["Day Calls"]<df4["Night Calls"]]
data5.shape

data6 = df4[df4["Day Mins"]<df4["Night Mins"]]
data6.shape

#Minutes of day, minutes of night and Count Length of the first 50 individuals
subset_first_50 = df4[["Day Mins", "Night Mins", "Account Length"]][:50]
subset_first_50.head()


subset[:10]


# ### Filtering with loc and iloc

df4.iloc[:,3:6] ##All rows for columns between 3 and 6
df4.iloc[1:10,:] ##All columns for rows 1 to 10

df4.loc[[1,5,8,36], ["Area Code", "VMail Plan", "Day Mins"]]


#Insert new rows into the dataframe

df4["Total Mins"] = df4["Day Mins"] + df4["Night Mins"] + df4["Eve Mins"]

df4["Total Mins"].head()

df4["Total Calls"] = df4["Day Calls"] + df4["Night Calls"] + df4["Eve Calls"]

df4["Total Calls"].head()

data.shape

data.head()


#Reading and writing files
 
# The following code reads a .dbf file with data corresponding to the geology of an area of Bolivia, cleans the data, checking the data types NaN and 0.00000, then writes them to a new .csv file. 

from dbfread import DBF


table = DBF(mainpath + '/' + '/Bolivia/Puntos.dbf', load=True)
frame = pd.DataFrame(iter(table))

#Checking type of missing values for x column
x = [x for x in frame.iloc[:]['X'] if np.isnan(x)]
print(np.isnan(x).any())

#Checking type of missing values for y column
y = [x for x in frame.iloc[:]['Y'] if np.isnan(x)]
print(np.isnan(x).any())

#writting new .csv file
outfile = open("Puntos_modificado.csv", "w")
# output the header row
outfile.write('OBJECTID,X,Y,Profondeur,Z_Tertiary,ProfTertia')
outfile.write('\n')
# output each of the rows:
row_string = str()
for i in range(len(frame)):
    if frame.iloc[i]['X'] == 0 and frame.iloc[i]['Y'] == 0 or np.isnan(frame.iloc[i]['X']) or np.isnan(frame.iloc[i]['Y']):
        continue
    else:
        row_string = '{},{},{},{},{},{}'.format(frame.iloc[i]['OBJECTID'],frame.iloc[i]['X'],frame.iloc[i]['Y'],frame.iloc[i]['Profondeur'],frame.iloc[i]['Z_Tertiary'],frame.iloc[i]['ProfTertia'])
    outfile.write(row_string)
    outfile.write('\n')
outfile.close()

#print dataframe
frame

#printing filtered csv
csv_file = pd.read_csv('Puntos_modificado.csv')
csv_file


#Read data from a URL


medals_url = "http://winterolympicsmedals.com/medals.csv"

medals_data = pd.read_csv(medals_url)

medals_data.head()


#Downloading data with urllib3

# Let's make an example using the urllib3 library to read the data from an external URL, process it and convert it to a python data frame before saving it to a local CSV.

def downloadFromURL(url, filename, sep = ",", delim = "\n", encoding="utf-8", 
                   mainpath = "Users/andrescarvajal/Desktop/Portfolio/dataCleaning/"):
    #import the library and make the connection to the data web site.
    import urllib3
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    print("The status of the response is %d" %(r.status))
    response = r.data 
    
    #The reponse object contains a binary string, so we convert it to a string by decoding it to UTF-8.
    str_data = response.decode(encoding)

    #split the string into an array of rows, separating it by intros
    lines = str_data.split(delim)

    #The first line contains the header, so we extract it
    col_names = lines[0].split(sep)
    n_cols = len(col_names)

    #generate an empty dictionary where the information processed from the external URL will be stored.
    counter = 0
    main_dict = {}
    for col in col_names:
        main_dict[col] = []

    #process row by row the information to fill the dictionary with the data as we did before.
    for line in lines:
        #skip the first line, which contains the header, and we have already processed
        if(counter > 0):
            #divide each string by the commas as separator element
            values = line.strip().split(sep)
            #add each value to its respective dictionary column
            for i in range(len(col_names)):
                main_dict[col_names[i]].append(values[i])
        counter += 1

    print("The data set has %d rows y %d columns"%(counter-1, n_cols))

    #convert the processed dictionary to Data Frame and check that the data is correct.
    df = pd.DataFrame(main_dict)    
    return df

medals_df = downloadFromURL(medals_url, "athletes/downloaded_medals")
medals_df.head()

