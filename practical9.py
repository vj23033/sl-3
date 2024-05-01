import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/titanic_data.csv')

data.head()
data.describe()

data.info()

data.isnull().sum()

data['Age'] = data['Age'].fillna(np.mean(data['Age']))
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data.isnull().sum()

sns.boxplot(data['Sex'], data["Age"], data["Survived"], palette = 'Set2').set_title('Plot for distribution of age with respect to each gender along with the information about whether they survived or not')
plt.show()




