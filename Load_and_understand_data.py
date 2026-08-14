import pandas as pd 
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("D:\Projects\Student Success Predictor\student_performance__dataset.csv")

print("Sample rows")
print(df.head())

print("Dataset Shape")
print(f"Rows: {df.shape[0]},columns: {df.shape[1]}")

print("Dataset Information")
print(df.info())

print("Summary Statistics")
print(df.describe(include='all'))

print("Missing Values")
print(df.isnull().sum())

# Preprocessing 

le=LabelEncoder()
# converts categorial data into numerical values
df['Internet']=le.fit_transform(df["Internet"])  # yes=1 and no=0
df['Passed']=le.fit_transform(df["Passed"])

print("After Encoding")
print(df.head())

print("Datatypes after cleaning")
print(df.dtypes)