
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
df= pd.read_csv("C:\\Users\\admin\\Downloads\\movies.csv") 
df

df.mean()
df.loc[:,'budget'].mean()
df.mean(axis=1)[0:4]

df.median()
df.loc[:,'budget'].median()
df.median(axis=1)[0:4]

df.mode()
df.loc[:,'budget'].mode()

df.min()
df.loc[:,'budget'].min(skipna = False)

df.max()
df.loc[:,'budget'].max(skipna = False)

df.std()
df.loc[:,'budget'].std()
df.std(axis=1)[0:4]

df['Gross Percent'] = (df['gross'] / df['gross'].sum()) * 100
df
df.groupby(['genre'])['votes'].mean()




Problem statement 2
import pandas as pd 
import numpy as np 
 df=pd.read_csv("C:\\Users\\admin\\Downloads\\iris.csv")
 df

median=df['sepal_length'].median() 
print("median : ",median)

mean=df['sepal_length'].mean() 
print("mean : ",mean)

mode=df['sepal_length'].mode() 
print("mode : ",mode)

sd=df['sepal_length'].std() 
print("Standard Deviation : ",sd)

group=df.groupby('species')
group
group.get_group('setosa')
