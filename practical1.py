
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

df = pd.read_csv("autodata.csv")

df.head(5)

df.tail(5)

df.info()

df.describe()

df.isnull()



df.isnull().sum()



df.notnull()

df.notnull().sum()

avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace = True)

avg_hp = df["horsepower"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_hp)
df["horsepower"].replace(np.nan, avg_hp, inplace = True)


avg_rpm = df["peak-rpm"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_rpm)
df["peak-rpm"].replace(np.nan, avg_hp, inplace = True)




df.isnull().sum()


df['city-L/100km'] = 235/df["city-mpg"]
df.head()


df['highway-L/100km'] = 235/df["highway-mpg"]
df.head()

df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max() 
df[["length","width","height"]].head()




df.columns
df['aspiration'].value_counts()

dummy_variable_1 = pd.get_dummies(df["aspiration"])
dummy_variable_1.head()
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("aspiration", axis = 1, inplace=True)
df.head()


