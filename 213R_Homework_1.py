
# coding: utf-8

# In[1]:


# Mitchell Pudil
# Econ 213R: Applied Machine Learning
# Homework 1: Exploring Data

# First Dataset: Homeschooling Data

import pandas as pd
import numpy as np
import matplotlib as plt


# Import Data and look at head
df = pd.read_csv("/Users/mitchellpudil/Desktop/Personal_Research/Homeschool/homeschool2.csv")
df.head(10) #


# In[2]:


df.describe()


# In[7]:


df['lnhhinc'].hist(bins=15)      # Histogram of natural log of income


# In[8]:


# Create column called "income" to graph actual income 

df['income'] = np.exp(df['lnhhinc']) 
df['income'].hist(bins=15)


# In[9]:


# Boxplot of child's age (cage)
df.boxplot(column='cage')


# In[10]:


# boxplot of child's age by whether or not they are homeschooled
df.boxplot(column='cage', by='homeschool')


# In[14]:


# Now using matplotlib to chart probability of being homeschooled by gender

import matplotlib.pyplot as plt

temp1 = df['cmale'].value_counts(ascending=True)
temp2 = df.pivot_table(values='homeschool', index=['cmale'])
print("Frequency Table for Homeschooling:")
print(temp1)

print("\nProbbility of being homeschooled by Gender")
print(temp2)

fig=plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax1.set_xlabel('homeschool')
ax1.set_ylabel('Count of Students')
ax1.set_title('Homeschool by Gender')
temp1.plot(kind='bar')

ax2=fig.add_subplot(122)
temp2.plot(kind='bar')
ax2.set_xlabel('homeschool')
ax2.set_ylabel('Probability of Getting Homeschooled')
ax2.set_title('Probability of Getting Homeschooled by Gender')





# In[16]:


# Let's look at how many homeschoolers/non-homeschoolers are religious (since it's BYU, afterall)

#stacked
temp3 = pd.crosstab(df['homeschool'], df['religious'])
temp3.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)




# In[19]:


# Subsetting df to look at people who are male, not homeschooled, and religious

malenr = df.loc[(df["cmale"]==1) & (df["homeschool"]==0) & (df["religious"]==1), ["cmale","homeschool","religious"]]
malenr.head(10)


# In[21]:


# Let's look at people who are male, whose family makes more than $100k, and are homeschooled

richmalehs = df.loc[(df['cmale']==1) & (df['hhinc'] > 100000) & (df['homeschool']==1)]
richmalehs.head(10)


# In[22]:


# Dataframe 2: Crime Rates

df2 = pd.read_csv("/Users/mitchellpudil/Desktop/Crime_Rate/crime.csv")

# Let's look at how many rows are missing per column
df2.apply(lambda x: sum(x.isnull()), axis=0)


# In[24]:


# Let's replace robbery rates with the mean of robbery rates

df2['rob'].fillna(df2['rob'].mean(), inplace=True)


# In[29]:


# Let's for the same for larceny and Snowfall
df2['larc'].fillna(df2['larc'].mean(), inplace=True)
df2['Snow'].fillna(df2['Snow'].mean(), inplace=True)

# Let's run that lambda function again to make sure that there are no missing values for the robbery rate
df2.apply(lambda x: sum(x.isnull()), axis=0)


# In[32]:


# Let's see if there's any correlation between snowfall and robbery

corr = np.corrcoef(df2['Snow'], df2['rob'])
corr[0][1]

# So there is a little bit of correlation between snowfall and robbery


# In[104]:


# The last data set we will use will come from a .txt file
indicators = pd.read_table('/Users/mitchellpudil/Desktop/indicators.txt', delim_whitespace=True)
indicators.head()
indicators.describe()


# In[98]:


indicators['PriceChange'].hist()


# In[161]:


# Let's look at the distribution of the average number of Loan Payments Overdue by Gender

indicators.boxplot(column='LoanPaymentsOverdue', by='Male')


# In[158]:


# Let's do a simple linear regression to estimate the effect of Price Change on Loan Payments Overdue
import statsmodels.formula.api as sm
result = sm.ols(formula="LoanPaymentsOverdue ~ PriceChange + Male", data=indicators).fit()
price_change_coef = round(result.params[1], 2)
male_coef = round(result.params[2], 2)

print("We expect that for a 100% increase in price change, that the number of overdue loan payments changes by {}.".format(price_change_coef))
print("We also find that the average difference in overdue loan payments between a male and a female (male-female) is {}.".format(male_coef))



# In[188]:


# For our next dataset, we will import data from an html file. The file we will be looking at is NFL statisticst)
r = requests.get("http://www.espn.com/mlb/standings/_/%20season/2017")
print(r.text[0:500])
# Parse html stored in our text into soup
from bs4 import BeautifulSoup
soup = BeautifulSoup(r.text, 'html.parser')


# In[272]:


import json

climate_data_json = json.load(open("/Users/mitchellpudil/Desktop/climate.json", 'r'))


# In[273]:


# Convert the data to a dataframe using pandas

climate_data_rough = pd.DataFrame(climate_data_json)
climate_data_rough.head(5)


# In[274]:


# Clean Data
df = climate_data_rough['data']  # Drops last column of NaN's
df.drop(df.tail(4).index,inplace=True)    # Drops last few rows of dataframe which are not data
df.head(5), df.tail(5)


# In[250]:


climate_clean = pd.DataFrame(df)
climate_clean.head(5)


# In[285]:


plt.plot(climate_clean)


# In[283]:




