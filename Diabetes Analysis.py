#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
from  matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')


# # Loading Data Set for Performing EDA (exploraitory data analysis)

# In[9]:


df= pd.read_csv("D:\Downloads\diabetes 01.csv")
print(df)


# # EDA starts from here

# In[13]:


df.head(11)
## df.head () returns the toprows of the DataFrame


# In[14]:


df.tail(11)
##  df.tail () returns the lastrows of the DataFrame


# In[16]:


df.shape
#    The first value represents the number of rows in the DataFrame.
 #   The second value represents the number of columns in the DataFrame.


# In[19]:


df.describe()
#The df.describe() method provides a quick overview of the central tendency and dispersion of numerical data in your DataFrame, which is helpful for understanding the distribution of your data.


# In[20]:


df.info()


# In[23]:


df.columns


# # Checking and Cleaning Null Values if any

# In[24]:


df.isnull()
df.isnull().sum()


# # Analysis starts by Plotting Graphs on different parameters.

# # Frequency Plot

# In[35]:


plt.figure(figsize=(9,6))
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
for i, col in enumerate(columns,1):
    plt.subplot(2, 4, i)
    
    
    if i% 2==0:
        sns.boxplot(x='Outcome',  y= col , data =df)
        plt.xlabel('Outcome')
    else:
        sns.histplot(data=df , x= col, hue ='Outcome', kde = True, bins = 20, edgecolor ='k')
        plt.xlabel(col)
    
    plt.ylabel('Frequency')
    
plt.tight_layout()


# # Distribution of Pregnancies

# In[36]:


for column in df.select_dtypes(include=['int64', 'float64']):
    plt.figure(figsize=(8,4))
    sns.histplot(df[column], bins =20, kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()


# # Correlation heatmap to visualize relationships between numerical variables

# In[41]:


correlation_matrix =df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot =True , cmap='coolwarm' , linewidths =0.5)
plt.title('correlation Heatmap')
plt.show()


# # feature distribution by diabetes outcome

# In[64]:


plt.figure(figsize=(8,6))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(2,4, i + 1)
    sns.swarmplot(x='Outcome', y=col , data=df)
    plt.xlabel('Outcome')
    plt.ylabel(col)
plt.tight_layout()
plt.show()


# # Pair Plot of Features with Outcome Comparison

# In[61]:


sns.pairplot(df, hue='Outcome', diag_kind = 'kde')
plt.title('Pair Plot')
plt.show()


# # Pair Plot of Glucose , Insulin , BMI , Age, and Outcome"

# In[67]:


columns_to_include = ['Glucose', 'Insulin', 'BMI', 'Age', 'Outcome']


sns.pairplot(df[columns_to_include], hue='Outcome', markers=["o", "s"], palette={0: 'blue', 1: 'orange'})
plt.show()


# # Box Plot of BloodPressure by Number of Pregnancies

# In[69]:


plt.figure(figsize=(8,8))
sns.boxplot(x='Pregnancies', y='BloodPressure', data=df)
plt.xlabel('Pregnancies')
plt.ylabel('BloodPressure')
plt.title('Box plot of BloodPressure by Number of Pregnancies')
plt.xticks(rotation = 820)
plt.show()


# # Conclusion

# In[ ]:


I performed a comprehensive exploratory data analysis (EDA) on a dataset related to diabetes patients. First, 
I imported essential libraries such as Pandas for data manipulation and Seaborn for visualization.
After loading the dataset from a CSV file, I examined its dimensions, finding 768 rows and 9 columns. 
Utilizing df.describe(), I obtained summary statistics for numerical columns, offering insights into data 
characteristics. Moreover, I confirmed the absence of missing values with df.info(). The data cleaning step 
revealed no null values. To understand data distribution and relationships, I created various visualizations, 
including histograms, density plots, a correlation heatmap, swarm plots, and a pair plot. 
I customized a specific pair plot for key columns and generated a box plot to explore blood pressure variations 
based on the number of pregnancies. This comprehensive analysis provides valuable insights into the dataset, aiding
in data-driven decisions and potential modeling for diabetes prediction.

