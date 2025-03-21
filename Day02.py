import numpy as np 
import pandas as pd 
test_df = pd.read_csv('train.csv')

# print(data)
# print(test_df["Sex"].value_counts())


data = test_df["Sex"].isin(["female"]).sum()
print(test_df[(test_df["Survived"] == 0)  & (test_df["Sex"] == 'female')])
print("The Female which are survived : ",len(test_df.query("Survived == 1 and Sex == 'female'")))
print("The Male which are Survived :",len(test_df.query("Survived == 1 and Sex == 'male'")))
print("The Female is which are not Survived:",len(test_df.query("Survived == 0 and Sex == 'female'")))
print("The male is which are not Survived:",len(test_df.query("Survived == 0 and Sex == 'male'")))
print(test_df["Sex"].value_counts())
print(len(test_df))
# print(test_df.head(4))
# print(test_df.agg({
#     "Age":['sum'],
#     "Sex":['mode']
#    }))

print(test_df["Age"].agg(["mean"]))

print(test_df.columns)
print(test_df.shape)
print(test_df["Age"].isnull().sum())
test_df.dropna(subset=["Age"],axis=0,inplace=True)
print(test_df["Age"].isnull().sum())
print(test_df.shape)
categorical = test_df.select_dtypes(include=["object"])

print(test_df.duplicated(keep="last").astype(int))

newtest_df = pd.read_csv('train.csv')
#Outliers Detection here
#Zscore 
outliers = 0 
# def Z_score_calculator(fulldata,outliers):
#   Z_scores = {}
#   for column in fulldata.columns: 
#     column_data = newtest_df[column]
#     mean = np.mean(column_data)
#     std = np.std(column_data)
#     Z_score = (column_data - mean)/ std
#     Z_scores[column] = Z_score
#   return pd.DataFrame(Z_scores)  



numerical_test_df = newtest_df.select_dtypes(include=["int64","int32","float"]) 
# data = Z_score_calculator(numerical_test_df,outliers)
# print(data[(data >= -3) & (data <= 3 )].dropna())

import pandas as pd
import matplotlib.pyplot as plt

def IQRmethod(datathings):
    data = datathings.copy()
    for column in datathings.columns: 
        Q1 = datathings[column].quantile(0.25)
        Q3 = datathings[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Apply filter column-wise, but keep all rows where any column is within bounds
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return data

# Apply IQR filter
# data = IQRmethod(numerical_test_df)

# Boxplot to check for outliers
plt.title("Fare (After IQR)")
plt.hist(numerical_test_df["Fare"],bins=15)
plt.show()

