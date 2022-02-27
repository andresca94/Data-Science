#!/usr/bin/env python
# coding: utf-8

# # DATA CLEANING

# ## Data loading through the read_csv function
# 
# The code below presents the titanic passenger dataset, load the data, perform some preliminary analysis and clean the columns with missing data. Dummies" variables will be added to the dataset representing categorical data.

# In[1]:


import pandas as pd
import os
import numpy as np
import random

mainpath = "/Users/andrescarvajal/Desktop/Portfolio/dataCleaning/"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)

data = pd.read_csv(fullpath)

data.head(10)


# In[2]:


data.tail(8)


# In[3]:


data.shape


# In[4]:


data.columns.values


# In[5]:


data.describe()


# In[6]:


data.dtypes


# ### Missing values

# In[7]:


pd.isnull(data["body"])


# In[8]:


pd.notnull(data["body"])


# In[9]:


pd.isnull(data["body"]).values.ravel().sum()


# In[10]:


pd.notnull(data["body"]).values.ravel().sum()


# ### Missing values in a data set can occur for two reasons:
# 
# * Data extraction
# * Data collection

# ### Deleting missing values

# In[11]:


data.dropna(axis=0, how="all")


# In[12]:


data2 = data


# In[13]:


data2.dropna(axis=0, how="any")


# ### Calculation of missing values

# In[15]:


data3 = data


# In[16]:


data3["body"] = data3["body"].fillna(0) #Change the NaN values to 0
data3["home.dest"] = data3["home.dest"].fillna("Unknown") #Change the NaN values to "Unknown"
data3.head(5)


# In[17]:


pd.isnull(data3["age"]).values.ravel().sum()


# In[18]:


data3["age"].fillna(data["age"].mean())


# In[19]:


data3["age"].fillna(method="ffill")


# In[20]:


data3["age"].fillna(method="backfill")


# In[21]:


data3


# ### Dummy variables

# In[22]:


data["sex"].head(10)


# In[23]:


dummy_sex = pd.get_dummies(data["sex"], prefix="sex")


# In[24]:


dummy_sex.head(10)


# In[25]:


column_name=data.columns.values.tolist()
column_name


# In[26]:


data = data.drop(["sex"], axis = 1)


# In[27]:


data = pd.concat([data, dummy_sex], axis = 1)


# In[28]:


def createDummies(df, var_name):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis = 1)
    df = pd.concat([df, dummy ], axis = 1)
    return df


# In[29]:


createDummies(data3, "sex")


# ## Data Visualization
# 
# The following code uses telephone customer churn data, it shows the visualization of the data by plotting columns of the data set in function of other columns and also in relation to their frequency. 

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data2 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", sep=",")
data3 = data2


# In[31]:


data2.head()


# In[32]:


data2.columns.values


# In[33]:


data_cols = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Columns.csv")
data_col_list = data_cols["Column_Names"].tolist()
data2 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", header = None, names = data_col_list)
data2.columns.values


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Scatter Plot

# In[35]:


data3.plot(kind="scatter", x="Day Mins", y="Day Charge")


# In[36]:


data3.plot(kind="scatter", x="Night Mins", y="Night Charge")


# In[ ]:


figure, axs = plt.subplots(2,2, sharey=True, sharex=True)
data3.plot(kind="scatter", x="Day Mins", y ="Day Charge", ax=axs[0][0])
data3.plot(kind="scatter", x="Night Mins", y="Night Charge", ax=axs[0][1])
data3.plot(kind="scatter", x="Day Calls", y ="Day Charge", ax=axs[1][0])
data3.plot(kind="scatter", x="Night Calls", y="Night Charge", ax=axs[1][1])


# ### Frequency histogram

# In[38]:


k = int(np.ceil(1+np.log2(3333)))
plt.hist(data3["Day Calls"], bins = k) #bins = [0,30,60,...,200]
plt.xlabel("Number of calls per day")
plt.ylabel("Frequency")
plt.title("Histogram of the number of calls per day")


# ### Boxplot

# In[39]:


plt.boxplot(data3["Day Calls"])
plt.ylabel("Number of calls per day")
plt.title("Boxplot of daily calls")


# In[40]:


data3["Day Calls"].describe()


# In[41]:


IQR=data3["Day Calls"].quantile(0.75)-data3["Day Calls"].quantile(0.25)
IQR


# In[42]:


data3["Day Calls"].quantile(0.25) - 1.5*IQR


# In[43]:


data3["Day Calls"].quantile(0.75) + 1.5*IQR


# ## Data loading through the open function

# In[44]:


data3 = open(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt",'r')


# In[45]:


cols = data3.readline().strip().split(",")
n_cols = len(cols)


# In[46]:


counter = 0
main_dict = {}
for col in cols:
    main_dict[col] = []


# In[47]:


for line in data3:
    values = line.strip().split(",")
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1

print("El data set tiene %d filas y %d columnas"%(counter-1, n_cols))


# In[48]:


df3 = pd.DataFrame(main_dict)
df3.head()


# # Data Wrangling
# ## Create a subset of data
# ### Subset of column or columns

# In[49]:


df4 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", sep=",")
df4


# In[50]:


account_length = df4["Account Length"]


# In[51]:


account_length.head()


# In[52]:


type(account_length)


# In[53]:


subset = df4[["Account Length", "Phone", "Eve Charge", "Day Calls"]]


# In[54]:


type(subset)


# In[55]:


desired_columns = ["Account Length", "Phone", "Eve Charge", "Night Calls"]
subset = df4[desired_columns]
subset.head()


# In[56]:


desired_columns = ["Account Length", "VMail Message", "Day Calls"]
desired_columns


# In[57]:


all_columns_list = df4.columns.values.tolist()
all_columns_list


# In[61]:


np.random.choice(all_columns_list) #Choose column value randomly 


# In[59]:


sublist = [x for x in all_columns_list if x not in desired_columns]
sublist


# In[60]:


subset = df4[sublist]
subset.head()


# ### Subset of rows

# In[62]:


df3[1:25]


# ### Subsets of rows with Boolean conditions

# In[63]:


##Users with Day Mins > 300
data1 = df4[df4["Day Mins"]>300]
data1.shape


# In[64]:


##New York Users (State = "NY")
data2 = df4[df4["State"]=="NY"]
data2.shape


# In[65]:


##AND -> &
data3 = df4[(df4["Day Mins"]>300) & (df4["State"]=="NY")]
data3.shape


# In[66]:


##OR -> |
data4 = df4[(df4["Day Mins"]>300) | (df4["State"]=="NY")]
data4.shape


# In[67]:


data5 = df4[df4["Day Calls"]<df4["Night Calls"]]
data5.shape


# In[68]:


data6 = df4[df4["Day Mins"]<df4["Night Mins"]]
data6.shape


# In[69]:


##Minutes of day, minutes of night and Count Length of the first 50 individuals
subset_first_50 = df4[["Day Mins", "Night Mins", "Account Length"]][:50]
subset_first_50.head()


# In[70]:


subset[:10]


# ### Filtering with loc and iloc

# In[71]:


df4.iloc[:,3:6] ##All rows for columns between 3 and 6
df4.iloc[1:10,:] ##All columns for rows 1 to 10


# In[73]:


df4.loc[[1,5,8,36], ["Area Code", "VMail Plan", "Day Mins"]]


# ### Insert new rows into the dataframe

# In[74]:


df4["Total Mins"] = df4["Day Mins"] + df4["Night Mins"] + df4["Eve Mins"]


# In[75]:


df4["Total Mins"].head()


# In[76]:


df4["Total Calls"] = df4["Day Calls"] + df4["Night Calls"] + df4["Eve Calls"]


# In[77]:


df4["Total Calls"].head()


# In[78]:


data.shape


# In[79]:


data.head()


# # Reading and writing files
# 
# The following code reads a .dbf file with data corresponding to the geology of an area of Bolivia, cleans the data, checking the data types NaN and 0.00000, then writes them to a new .csv file. 

# In[80]:


from dbfread import DBF


table = DBF(mainpath + '/' + '/Bolivia/Puntos.dbf', load=True)
frame = pd.DataFrame(iter(table))

#Checking type of missing values for x column
x = [x for x in frame.iloc[:]['X'] if np.isnan(x)]
print(np.isnan(x).any())

#Checking type of missing values for y column
y = [x for x in frame.iloc[:]['Y'] if np.isnan(x)]
print(np.isnan(x).any())


# In[81]:


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


# In[82]:


#printing filtered csv
csv_file = pd.read_csv('Puntos_modificado.csv')
csv_file


# # Read data from a URL

# In[85]:


medals_url = "http://winterolympicsmedals.com/medals.csv"


# In[86]:


medals_data = pd.read_csv(medals_url)


# In[87]:


medals_data.head()


# # Downloading data with urllib3
# 
# Let's make an example using the urllib3 library to read the data from an external URL, process it and convert it to a python data frame before saving it to a local CSV.

# In[90]:


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


# In[91]:


medals_df = downloadFromURL(medals_url, "athletes/downloaded_medals")
medals_df.head()

