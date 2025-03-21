import numpy as np 
import pandas as  pd 
df = pd.read_csv('train.csv')
df["Age"] = df["Age"].fillna(df["Age"].median()).astype(int)
#🔹1) How many passengers survived vs. died?
df.info()
print(df.shape)

Survived = df[(df["Survived"] == 1)]
# print(Survived)
passengerDied = df[df["Survived"] ==  0 ]
# print(passengerDied)

#🔹 What was the survival rate by gender?
Males = df[df["Sex"] == "male"]
Females = df[df["Sex"] == "female"]
Male_Survival = df[(df["Survived"] == 1) & (df["Sex"] == "male")]
Female_Survival = df[(df["Survived"] == 1 ) & (df["Sex"] == "female")]
Male_Survival_Rate =  (Male_Survival.shape[0] / Males.shape[0]) * 100
Female_Survival_Rate = (Female_Survival.shape[0]/Females.shape[0]) * 100
print("Male Survival Rate:",Male_Survival_Rate,"Female Survival Rate:",Female_Survival_Rate)



# 🔹 Did class (1st, 2nd, 3rd) affect survival chances?

print("GroupBy Pclass:",df.groupby("Pclass")["Survived"].sum()
)#🔹 What was the average age of survivors vs. non-survivors?
Survivors_Age = df.query('Survived == 1')
print(Survivors_Age["Age"].mean())
NonSurvivors_Age = df.query('Survived != 1')
print(NonSurvivors_Age["Age"].mean())

#🔹 Was there a correlation between fare price and survival?
numerical_methods = df.select_dtypes(include=["int","float"])
df_corrmatrix = numerical_methods.corr( method="kendall")
print("The Correlation between the Fare and the Survived : ",df_corrmatrix["Fare"]["Survived"])

# 4️⃣ Aggregation & Grouping
# 🔹 What is the average fare price per class?

print(df.groupby("Pclass")["Fare"].mean())
# 🔹 What is the most common age group among passengers?


print(df["Age"].value_counts().idxmax())

# 🔹 Which departure port had the highest number of passengers?



print(df["Embarked"].value_counts().idxmax())


# 🔹 How did survival rates vary by family size?

datawithparentandsiblings = df[((df["SibSp"] != 0) & (df["Parch"] !=  0)) & (df["Survived"] == 1 ) ]

newdasta = (datawithparentandsiblings.shape[0]/df.shape[0])*  100  
print(newdasta)





