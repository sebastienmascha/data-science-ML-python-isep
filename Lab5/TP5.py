import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

# We open and describe data
df_train = pd.read_csv('./titanic_train.csv')
df_test = pd.read_csv('./titanic_test.csv')
df_train.head()
df_train.dtypes

# We calculate nan values
df_train.isnull().sum()

# We plot age distribution and ignore nan values
fig, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['Age'].dropna(), color='darkred', bins=30, ax=ax)
plt.show()


# Survived pieplot
fig, ax = plt.subplots(figsize=(18, 8))
df_train['Survived'].value_counts().plot(kind='pie', subplots=True, figsize=(8, 8))
plt.show()

#Pclass pieplot
fig, ax = plt.subplots(figsize=(18, 8))
df_train['Pclass'].value_counts().plot(kind='pie', subplots=True, figsize=(8, 8))
plt.show()

# Number of dead/survived
fig, ax = plt.subplots(figsize=(18, 8))
sns.countplot(df_train['Survived'])
plt.show()


# Percentage of survived
def pourcentageDf(df):
    return df.value_counts()/df.shape[0]*100

pourcentageSurvived = df_train['Survived'].value_counts()/df_train['Survived'].shape[0]*100
print("Pourcentage total de morts : ", pourcentageSurvived[0],"%")
print("Pourcentage total de survivants : ", pourcentageSurvived[1],"%")


#Percentage of survived child/adults
# We firstly create a new dataframe with child and delete nan values
child = np.zeros((len(df_train['Age'])))
child[df_train['Age'] < 18] = 1
child[df_train['Age'] >= 18] = 0
df_child_train = pd.DataFrame(child)
df_child_train.columns = ['Child']
df_trainChild = pd.concat([df_train, df_child_train], axis=1)
df_trainChild = df_trainChild.dropna(subset=['Age'])

pourcentageEnfantSurvivant = pourcentageDf(df_trainChild[df_trainChild['Child'] == 1]['Survived'])
pourcentageAdulteSurvivant = pourcentageDf(df_trainChild[df_trainChild['Child'] == 0]['Survived'])
fig, ax = plt.subplots()
pourcentageEnfantSurvivant.plot(kind='pie', subplots=True, figsize=(8, 8))
plt.show()
fig, ax = plt.subplots()
pourcentageAdulteSurvivant.plot(kind='pie', subplots=True, figsize=(8, 8))
plt.show()
print("Survived child : ", pourcentageEnfantSurvivant[0], "%")
print("Dead child : ", pourcentageEnfantSurvivant[1], "%")
print("Survived adult : ", pourcentageAdulteSurvivant[0], "%")
print("Dead adult : ", pourcentageAdulteSurvivant[1], "%")


# Percentage of survived men/woman
pourcentageSurvieHomme = pourcentageDf(df_train[df_train['Sex'] == 'male']['Survived'])
pourcentageSurvieFemme = pourcentageDf(df_train[df_train['Sex'] == 'female']['Survived'])
result = pd.concat([pourcentageSurvieHomme, pourcentageSurvieFemme], axis=1)
result.columns = ['Male', 'Female']
print(result)

# Percentage survived per class
pourcentageSurvieCl1 = pourcentageDf(df_train[df_train['Pclass'] == 1]['Survived'])
pourcentageSurvieCl2 = pourcentageDf(df_train[df_train['Pclass'] == 2]['Survived'])
pourcentageSurvieCl3 = pourcentageDf(df_train[df_train['Pclass'] == 3]['Survived'])
result = pd.concat([pourcentageSurvieCl1, pourcentageSurvieCl2, pourcentageSurvieCl3], axis=1)
result.columns = ['Class 1', 'Class 2', 'Class 3']
print(result)

#We create variable Sex_cat - transformation of categorical data in int
sex_array = np.zeros((len(df_trainChild['Sex'])))
sex_array[df_trainChild['Sex'] =='male']=1
sex_array[df_trainChild['Sex'] =='female']=0
df_trainChild['Sex_cat'] = sex_array.astype(object)




# Correlation
corr_matrix = df_trainChild[['Survived', 'Pclass', 'Sex_cat', 'Child', 'Fare']].corr()
fig, ax = plt.subplots(figsize=(18, 8))
sns.heatmap(corr_matrix, annot=True)
plt.show()



# Create Fare2 for train dataset
fare2 = np.zeros((len(df_trainChild['Fare'])))
i=4
for f in range(40, 9, -10):
    fare2[df_trainChild['Fare'] <= f] = i
    i-=1
    print (f)
df_trainChild['Fare2'] = fare2.astype(int)


# Create child, Sex_cat and Fare2 for test
child = np.zeros((len(df_test['Age'])))
child[df_test['Age'] < 18] = 1
child[df_test['Age'] >= 18] = 0
df_child_test = pd.DataFrame(child)
df_child_test.columns = ['Child']
df_testChild = pd.concat([df_test, df_child_test], axis=1)
df_testChild = df_trainChild.dropna(subset=['Age'])
fare2 = np.zeros((len(df_testChild['Fare'])))
i=4
for f in range(40, 9, -10):
    fare2[df_testChild['Fare'] <= f] = i
    i-=1
df_testChild['Fare2'] = fare2
sex_array = np.zeros((len(df_testChild['Sex'])))
sex_array[df_testChild['Sex'] =='male']=1
sex_array[df_testChild['Sex'] =='female']=0
df_testChild['Sex_cat'] = sex_array.astype(object)

pourcentageSurvieFare2_list=[]
col_list = []
for f in range(1,5):
    pourcentageSurvieFare2 = pourcentageDf(df_trainChild[df_trainChild['Fare2'] == f]['Survived'])
    pourcentageSurvieFare2_list.append(pourcentageSurvieFare2)
    col_list.append('Fare '+str(f))
result = pd.concat(pourcentageSurvieFare2_list, axis=1)
result.columns = col_list
print(result)

# Exercice B
# Naive Bayes
# needed imports
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

#classifier choice
gnbModel=GaussianNB()
#choice of the training set, considered attributes and variable to predict
gnbModel.fit(df_trainChild[['Pclass', 'Fare2', 'Child', 'Sex_cat']], df_trainChild['Survived'])	#Sex_cat stands for transformed categorical array

#expected results are stored in a separate vector
expected =df_trainChild['Survived']

#predictions on the training set
predicted = gnbModel.predict(df_trainChild[['Pclass', 'Fare2', 'Child', 'Sex_cat']])
#displaying relevant metrics
print(metrics.classification_report(expected, predicted))

#same when applying the model to the test set
expected =df_testChild['Survived']
predicted = gnbModel.predict(df_testChild[['Child', 'Sex_cat']])
print(metrics.classification_report(expected, predicted))


# do the same with different attributes




# Random trees
from IPython.display import Image
import pydotplus
from sklearn import tree

clf =tree.DecisionTreeClassifier()
clf=clf.fit(df_trainChild[['Pclass', 'Fare2', 'Child', 'Sex_cat']].values, df_trainChild['Survived'].values)
#expected results are stored in a separate vector
expected =df_trainChild['Survived']
#predictions on the training set
predicted = clf.predict(df_trainChild[['Pclass', 'Fare2', 'Child', 'Sex_cat']])
#displaying relevant metrics
print(metrics.classification_report(expected, predicted))

#same when applying the model to the test set
expected =df_testChild['Survived']
predicted = clf.predict(df_testChild[['Pclass', 'Fare2', 'Child', 'Sex_cat']])
print(metrics.classification_report(expected, predicted))
data_feature_names = ['Pclass', 'Fare2', 'Child', 'Sex_cat']
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=data_feature_names,class_names=True, filled=True,rounded=True, precision=0)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.write_png('./filename.png'))

