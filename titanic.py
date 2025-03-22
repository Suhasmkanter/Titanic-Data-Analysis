import numpy as np 
import pandas as  pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Plot Survival rates by gender & class (bar chart).

values = df.groupby(["Sex",'Pclass'])["Survived"].mean().unstack()
print(values)
#Graph against Survivers vs gender and surivers vs class

import matplotlib.pyplot as plt
import numpy as np

# Grouped values (Mean survival rate of Sex vs. Pclass)
values = df.groupby(["Sex", "Pclass"])["Survived"].mean().unstack()

# Plot bars for each Pclass separately
x_labels = values.index  # ['male', 'female']
x_positions = np.arange(len(x_labels))  # [0, 1]
bar_width = 0.2  # Width of each bar

fig, ax = plt.subplots(figsize=(8, 6))

# Loop through each Pclass and plot bars
# for i, pclass in enumerate(values.columns):
#     ax.bar(x_positions + i * bar_width, values[pclass], width=bar_width, label=f"Pclass {pclass}")

# Formatting the chart
# ax.set_xticks(x_positions + bar_width)  # Center x-labels
# ax.set_xticklabels(x_labels)  # Set labels (Sex: Male, Female)
# ax.set_xlabel("Sex")
# ax.set_ylabel("Survival Rate")
# ax.set_title("Survival Rate by Sex and Pclass")
# ax.legend(title="Pclass")


# Show age distribution of passengers (histogram).
plt.hist(df["Age"],bins=30)


#Use a heatmap to see correlations.
plt.figure()
print(numerical_methods)

sns.heatmap(df_corrmatrix)
# plt.show()

#Feature Engineering 
print(df.dtypes)
df["Sex"] = df["Sex"].map({"male":0,"female":1})
print(df["Sex"].head(4))
df = df.drop(labels=["Cabin","Ticket"],axis=1)
print(df.dtypes)


df["Embarked"] = df["Embarked"].map({"S":0,"C":1,"Q":2,np.nan:0})
print(df.dtypes)

df["Total_family"] = df["SibSp"] + df["Parch"]

print(df.dtypes)
df["Age_Group"] = df["Age"].apply(lambda x: 0 if x < 10 else (1 if x < 18 else (2 if x < 45 else 3))) 

print(df.dtypes)


xtrain = df.drop(columns=["Survived","Name","Age","PassengerId"])



print(xtrain.dtypes)
ytrain = df["Survived"]

print(df.isna().sum())

X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)


scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)



from sklearn.model_selection import GridSearchCV  

params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
grid = GridSearchCV(LogisticRegression(), params, cv=5)  # 5-fold cross-validation
grid.fit(X_train, y_train)


best_params = grid.best_params_  
best_model = grid.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score,classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# y_pred = grid.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier


models = RandomForestClassifier(n_estimators=1000,random_state=42)
models.fit(X_train,y_train)
print("Random Forest Accuracy:", models.score(X_test, y_test))



# Load the test dataset
test_df = pd.read_csv('test.csv')

# Preprocess the test data (apply the same transformations as train data)
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median()).astype(int)
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
test_df["Embarked"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test_df["Total_family"] = test_df["SibSp"] + test_df["Parch"]

# Convert "Age" into categories (if used in training)
test_df["Age_Group"] = test_df["Age"].apply(lambda x: 0 if x < 10 else (1 if x < 18 else (2 if x < 45 else 3)))

# Drop unnecessary columns (same as train data)
test_df = test_df.drop(labels=["Cabin", "Ticket", "Name", "Age","PassengerId"], axis=1, errors='ignore')

# Ensure test data has same features as X_train
print("Test Data Features:", test_df.columns)

# Predict survival using trained model
predictions = models.predict(test_df)

# Print the first few predictions
print(predictions[:10])  # Show first 10 predicted survival values

# If needed, save predictions for submission
submission = pd.DataFrame({"PassengerId": pd.read_csv("test.csv")["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")






