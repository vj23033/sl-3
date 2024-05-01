import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/titanic_data.csv')
data


data.shape

data.describe()

data.describe(include = 'object')


data.isnull().sum()


data['Age'] = data['Age'].fillna(np.mean(data['Age']))

data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


data.isnull().sum()


sns.countplot(data['Survived'])


sns.countplot(data['Pclass'])


sns.countplot(data['Sex'])

sns.boxplot(data['Age'])

sns.boxplot(data['Fare'])

sns.boxplot(data['Pclass'])

sns.boxplot(data['SibSp'])

sns.catplot(x= 'Pclass', y = 'Age', data=data, kind = 'box')

sns.catplot(x= 'Pclass', y = 'Fare', data=data, kind = 'strip')

sns.pairplot(data)

sns.pairplot(data)

sns.scatterplot(x = 'Fare', y = 'Pclass', hue = 'Survived', data = data)

sns.scatterplot(x = 'Survived', y = 'Fare', data = data)

sns.distplot(data['Age'])

sns.jointplot(x = "Survived", y = "Fare", kind = "scatter", data = data)


tc = data.corr()
sns.heatmap(tc, cmap="YlGnBu")
plt.title('Correlation')


sns.catplot(x='Pclass', y='Fare', data=data, kind='bar')

