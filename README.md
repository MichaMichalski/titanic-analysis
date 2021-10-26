# Titanic - Class Prediction Project

## 1. (Business) Goal:
- Goal of this project is to be able to build a model which is able to predict the circumstances on which the passengers of the titanic survived or not.

## 2. Get Data
- I decided to get the data from seaborn to make sure other people can reproduce the same results on their device and herefore having access to the exact same dataset. The seaborn dataset looks mostly the same as the dataset given in the kaggle competition


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = sns.load_dataset('titanic')
```


```python
# first taking a look at the first lines to make sure import worked as expected.
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.7+ KB
    

### The columns `deck` and `age` seem to have a lot of missing values. The rest seems to be fine


```python
df.isna().sum()
```




    survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         2
    class            0
    who              0
    adult_male       0
    deck           688
    embark_town      2
    alive            0
    alone            0
    dtype: int64



## 3. Train/Test split
- Before I can split the dataset I want to make sure, that the data I pass makes sense. 
    - I have a lot of missing values in the `deck` column. I will drop this column as in imputation on this one may generate a lot of non-representant values.
    - For the same reason I will drop the `age` column. Also we do have information in the `who` column regarding if adult or child. This may be enough information to derive from an attribute as `age`
    - I will remove the `adult_male` column as it is only a filter for the `who` column returning `True` if it is a man.
    - The column `pclass` will also be removed, as we have the same information in the `class` column.
    - I will also drop the `alive` column as this is only a resemblance of the `survived` column. I will use the `survived` column as the column I want to predict
    - I will drop the `embark_town` column as it is redundant with the `embarked` column
- I decided to drop the 2 rows having `NaN` values on the `embarked` column as we probably wont suffer a lot data loss and keep things "real"


```python
df.drop(labels=['deck', 'age', 'alive', 'embark_town', 'adult_male', 'pclass'], axis=1, inplace=True)
```


```python
df.dropna(inplace=True)
```


```python
y = df['survived']
# Passing the "cleaned" DataFrame with exception of the 'survived' column
X = df.drop('survived', axis=1)
```


```python
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42)
Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape
```




    ((666, 8), (223, 8), (666,), (223,))



## 4. Explore the data
#### First I will assemble the train data together with its target to have a thorough exploration


```python
fulltraindf = pd.concat([Xtrain, ytrain], axis=1)
fulltraindf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>alone</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>376</th>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>458</th>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>10.5000</td>
      <td>S</td>
      <td>Second</td>
      <td>woman</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>732</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>507</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>26.5500</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>830</th>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>14.4542</td>
      <td>C</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(fulltraindf.corr(), annot=True)
```




    <AxesSubplot:>




    
![png](Titanic_project_files/Titanic_project_16_1.png)
    


Most of my columns are not highly correlated which is good. This means, that I don't have a any reason to drop another column due to redundance

#### 4.1 Survived vs. Not Survived


```python
ax = fulltraindf['survived'].value_counts().plot(kind='bar', rot=0.5)
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = 'Not Survived'
labels[1] = 'Survived'
ax.set_xticklabels(labels)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.01))
```


    
![png](Titanic_project_files/Titanic_project_19_0.png)
    


We can see, that 408 people drowned, while 258 survived

#### 4.2 Class


```python
sns.barplot(data=fulltraindf, x='class', y='survived')
```




    <AxesSubplot:xlabel='class', ylabel='survived'>




    
![png](Titanic_project_files/Titanic_project_22_1.png)
    


Here we see clearly, that being passenger of the first class clearly contributes to the chances of being a Titanic survivor

## 5. Feature Engineering

Let's take a look at our columns and differiantiate between categorical columns and numerical ones

Categorical:
- In the `sex` column we have "male" and "female" as categories.
- In the `embarked` column we have the ports in which the passengers boarded "S", "C" and "Q".
- In the `class` column we have "First", "Second" and "Third" class as categories.
- In the `who` column we have "woman", "man" and "child" as categories.
- Finally in the `alone` column  we have "True" and "False" as values describing if the passengers were in compandy of another passenger or not

Numerical:
- We have the `sibsp` column describing the amount of siblings or spouses for each passenger. The numbers range between 0 and 8
- The `parch` column describes how many parents/children the passenger has on board of the ship. These numbers range from 0 to 6
- Finally we have the `fare` column describing how much the passenger paid for their ticket. Thes numbers range between 0 and 512.3292

I decided to hot-encode all the categorical columns and run the MinMaxScaler on all numerical columns


```python
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
```


```python
trans = ColumnTransformer(
    (
        ('cat_preprocessing', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['sex', 'embarked', 'class', 'who', 'alone']),
        ('num_preprocessing', MinMaxScaler(), ['sibsp', 'parch', 'fare'])
    )
)
```


```python
XtrainTrans = trans.fit_transform(Xtrain)
XtestTrans = trans.transform(Xtest)
```

# 6. Train Model
#### 6.1 Have a baseline Model
##### I decided to go with the DummyClassifier for the baseline modell


```python
from sklearn.dummy import DummyClassifier
```


```python
dm = DummyClassifier()
```


```python
dm.fit(XtrainTrans, ytrain)
```




    DummyClassifier()




```python
dm.score(XtrainTrans, ytrain)
```




    0.6126126126126126



#### 6.2 Train other models


```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```

I decided to run several models and use the VotingClassifier in order to select the one model with the best results


```python
mlr = LogisticRegression()
mrf = RandomForestClassifier()
msv = SVC()
tree = DecisionTreeClassifier(max_depth=3)
models = [
    ('logreg', mlr),
    ('forest', mrf),
    ('svm', msv),
    ('tree', tree)
]
```


```python
m = VotingClassifier(models)
```


```python
m.fit(XtrainTrans, ytrain)
```




    VotingClassifier(estimators=[('logreg', LogisticRegression()),
                                 ('forest', RandomForestClassifier()),
                                 ('svm', SVC()),
                                 ('tree', DecisionTreeClassifier(max_depth=3))])




```python
# Accuracy
m.score(XtrainTrans, ytrain), m.score(XtestTrans, ytest)
```




    (0.8543543543543544, 0.8071748878923767)



The accuracy on the train data is 85% while on the test data it's lower - 80%. Let's proceed to optimization.

# 7. Optimization

#### Crossvalidation


```python
from sklearn.model_selection import cross_val_score
import numpy as np
```

I will compare 4 performance metrics: `accuracy`, `precision`, `recall` and `f1`. I will create a DataFrame out of this `cross validation` in order to have a better overview of this metrics.


```python
strings = [
    'accuracy',
    'precision',
    'recall',
    'f1'
]
cols = ['traintest', 'model', 'metric', 'mean', 'std']
traintest= [[XtrainTrans, ytrain], [XtestTrans, ytest]]
metricdf = pd.DataFrame(columns=cols)
for i in range(len(traintest)):
    if i == 0:
        trainortest = 'train'
    else:
        trainortest = 'test'
    for model in models:
        for metric in strings:
            result = cross_val_score(model[1], traintest[i][0], traintest[i][1], cv = 10, scoring=metric)
            row = {'traintest': trainortest, 'model': model[0], 'metric': metric, 'mean': round(np.mean(result), 3), 'std': round(np.std(result),5)}
            metricdf = metricdf.append(row, ignore_index=True)
    for metric in strings:
        result = cross_val_score(m, traintest[i][0], traintest[i][1], cv = 10, scoring=metric)
        row = {'traintest': trainortest, 'model': 'VotingClassifier', 'metric': metric, 'mean': round(np.mean(result), 3), 'std': round(np.std(result),5)}
        metricdf = metricdf.append(row, ignore_index=True)
```


```python
metricdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>traintest</th>
      <th>model</th>
      <th>metric</th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>logreg</td>
      <td>accuracy</td>
      <td>0.805</td>
      <td>0.03339</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train</td>
      <td>logreg</td>
      <td>precision</td>
      <td>0.770</td>
      <td>0.05865</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train</td>
      <td>logreg</td>
      <td>recall</td>
      <td>0.717</td>
      <td>0.07931</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train</td>
      <td>logreg</td>
      <td>f1</td>
      <td>0.739</td>
      <td>0.04852</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train</td>
      <td>forest</td>
      <td>accuracy</td>
      <td>0.821</td>
      <td>0.04102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>train</td>
      <td>forest</td>
      <td>precision</td>
      <td>0.792</td>
      <td>0.09202</td>
    </tr>
    <tr>
      <th>6</th>
      <td>train</td>
      <td>forest</td>
      <td>recall</td>
      <td>0.748</td>
      <td>0.06423</td>
    </tr>
    <tr>
      <th>7</th>
      <td>train</td>
      <td>forest</td>
      <td>f1</td>
      <td>0.760</td>
      <td>0.04740</td>
    </tr>
    <tr>
      <th>8</th>
      <td>train</td>
      <td>svm</td>
      <td>accuracy</td>
      <td>0.815</td>
      <td>0.02562</td>
    </tr>
    <tr>
      <th>9</th>
      <td>train</td>
      <td>svm</td>
      <td>precision</td>
      <td>0.876</td>
      <td>0.07042</td>
    </tr>
    <tr>
      <th>10</th>
      <td>train</td>
      <td>svm</td>
      <td>recall</td>
      <td>0.620</td>
      <td>0.08550</td>
    </tr>
    <tr>
      <th>11</th>
      <td>train</td>
      <td>svm</td>
      <td>f1</td>
      <td>0.719</td>
      <td>0.05303</td>
    </tr>
    <tr>
      <th>12</th>
      <td>train</td>
      <td>tree</td>
      <td>accuracy</td>
      <td>0.824</td>
      <td>0.03009</td>
    </tr>
    <tr>
      <th>13</th>
      <td>train</td>
      <td>tree</td>
      <td>precision</td>
      <td>0.800</td>
      <td>0.04929</td>
    </tr>
    <tr>
      <th>14</th>
      <td>train</td>
      <td>tree</td>
      <td>recall</td>
      <td>0.732</td>
      <td>0.06118</td>
    </tr>
    <tr>
      <th>15</th>
      <td>train</td>
      <td>tree</td>
      <td>f1</td>
      <td>0.763</td>
      <td>0.04271</td>
    </tr>
    <tr>
      <th>16</th>
      <td>train</td>
      <td>VotingClassifier</td>
      <td>accuracy</td>
      <td>0.821</td>
      <td>0.02402</td>
    </tr>
    <tr>
      <th>17</th>
      <td>train</td>
      <td>VotingClassifier</td>
      <td>precision</td>
      <td>0.851</td>
      <td>0.05324</td>
    </tr>
    <tr>
      <th>18</th>
      <td>train</td>
      <td>VotingClassifier</td>
      <td>recall</td>
      <td>0.659</td>
      <td>0.08959</td>
    </tr>
    <tr>
      <th>19</th>
      <td>train</td>
      <td>VotingClassifier</td>
      <td>f1</td>
      <td>0.736</td>
      <td>0.05068</td>
    </tr>
    <tr>
      <th>20</th>
      <td>test</td>
      <td>logreg</td>
      <td>accuracy</td>
      <td>0.793</td>
      <td>0.09079</td>
    </tr>
    <tr>
      <th>21</th>
      <td>test</td>
      <td>logreg</td>
      <td>precision</td>
      <td>0.761</td>
      <td>0.15184</td>
    </tr>
    <tr>
      <th>22</th>
      <td>test</td>
      <td>logreg</td>
      <td>recall</td>
      <td>0.697</td>
      <td>0.19462</td>
    </tr>
    <tr>
      <th>23</th>
      <td>test</td>
      <td>logreg</td>
      <td>f1</td>
      <td>0.705</td>
      <td>0.13741</td>
    </tr>
    <tr>
      <th>24</th>
      <td>test</td>
      <td>forest</td>
      <td>accuracy</td>
      <td>0.780</td>
      <td>0.06245</td>
    </tr>
    <tr>
      <th>25</th>
      <td>test</td>
      <td>forest</td>
      <td>precision</td>
      <td>0.715</td>
      <td>0.14019</td>
    </tr>
    <tr>
      <th>26</th>
      <td>test</td>
      <td>forest</td>
      <td>recall</td>
      <td>0.646</td>
      <td>0.12824</td>
    </tr>
    <tr>
      <th>27</th>
      <td>test</td>
      <td>forest</td>
      <td>f1</td>
      <td>0.670</td>
      <td>0.06677</td>
    </tr>
    <tr>
      <th>28</th>
      <td>test</td>
      <td>svm</td>
      <td>accuracy</td>
      <td>0.798</td>
      <td>0.07757</td>
    </tr>
    <tr>
      <th>29</th>
      <td>test</td>
      <td>svm</td>
      <td>precision</td>
      <td>0.828</td>
      <td>0.14795</td>
    </tr>
    <tr>
      <th>30</th>
      <td>test</td>
      <td>svm</td>
      <td>recall</td>
      <td>0.597</td>
      <td>0.18993</td>
    </tr>
    <tr>
      <th>31</th>
      <td>test</td>
      <td>svm</td>
      <td>f1</td>
      <td>0.671</td>
      <td>0.14818</td>
    </tr>
    <tr>
      <th>32</th>
      <td>test</td>
      <td>tree</td>
      <td>accuracy</td>
      <td>0.785</td>
      <td>0.08271</td>
    </tr>
    <tr>
      <th>33</th>
      <td>test</td>
      <td>tree</td>
      <td>precision</td>
      <td>0.756</td>
      <td>0.11654</td>
    </tr>
    <tr>
      <th>34</th>
      <td>test</td>
      <td>tree</td>
      <td>recall</td>
      <td>0.635</td>
      <td>0.21952</td>
    </tr>
    <tr>
      <th>35</th>
      <td>test</td>
      <td>tree</td>
      <td>f1</td>
      <td>0.667</td>
      <td>0.15760</td>
    </tr>
    <tr>
      <th>36</th>
      <td>test</td>
      <td>VotingClassifier</td>
      <td>accuracy</td>
      <td>0.807</td>
      <td>0.07037</td>
    </tr>
    <tr>
      <th>37</th>
      <td>test</td>
      <td>VotingClassifier</td>
      <td>precision</td>
      <td>0.818</td>
      <td>0.11946</td>
    </tr>
    <tr>
      <th>38</th>
      <td>test</td>
      <td>VotingClassifier</td>
      <td>recall</td>
      <td>0.635</td>
      <td>0.17906</td>
    </tr>
    <tr>
      <th>39</th>
      <td>test</td>
      <td>VotingClassifier</td>
      <td>f1</td>
      <td>0.697</td>
      <td>0.12151</td>
    </tr>
  </tbody>
</table>
</div>



Some people prefer to use the `classification_report` function for this overview, so I took the opportunity to display the results with this function too.


```python
from sklearn.metrics import classification_report
```


```python
allmodels = [
    mlr,
    mrf,
    msv,
    tree,
    m
]
for each in allmodels:
    for dataset in range(len(traintest)):
        if dataset == 0:
            trainortest = 'train'
        else:
            trainortest = 'test'
        each.fit(traintest[dataset][0], traintest[dataset][1])
        pred = each.predict(traintest[dataset][0])
        print(trainortest, str(type(each)).split('.')[-1].replace("'>", ""))
        print(classification_report(traintest[dataset][1], pred))
```

    train LogisticRegression
                  precision    recall  f1-score   support
    
               0       0.84      0.87      0.85       408
               1       0.77      0.73      0.75       258
    
        accuracy                           0.81       666
       macro avg       0.81      0.80      0.80       666
    weighted avg       0.81      0.81      0.81       666
    
    test LogisticRegression
                  precision    recall  f1-score   support
    
               0       0.85      0.86      0.85       141
               1       0.75      0.73      0.74        82
    
        accuracy                           0.81       223
       macro avg       0.80      0.79      0.80       223
    weighted avg       0.81      0.81      0.81       223
    
    train RandomForestClassifier
                  precision    recall  f1-score   support
    
               0       0.94      0.96      0.95       408
               1       0.94      0.90      0.92       258
    
        accuracy                           0.94       666
       macro avg       0.94      0.93      0.93       666
    weighted avg       0.94      0.94      0.94       666
    
    test RandomForestClassifier
                  precision    recall  f1-score   support
    
               0       0.96      0.96      0.96       141
               1       0.94      0.94      0.94        82
    
        accuracy                           0.96       223
       macro avg       0.95      0.95      0.95       223
    weighted avg       0.96      0.96      0.96       223
    
    train SVC
                  precision    recall  f1-score   support
    
               0       0.80      0.96      0.87       408
               1       0.90      0.62      0.74       258
    
        accuracy                           0.83       666
       macro avg       0.85      0.79      0.80       666
    weighted avg       0.84      0.83      0.82       666
    
    test SVC
                  precision    recall  f1-score   support
    
               0       0.81      0.96      0.88       141
               1       0.91      0.61      0.73        82
    
        accuracy                           0.83       223
       macro avg       0.86      0.79      0.81       223
    weighted avg       0.85      0.83      0.82       223
    
    train DecisionTreeClassifier
                  precision    recall  f1-score   support
    
               0       0.84      0.90      0.87       408
               1       0.83      0.74      0.78       258
    
        accuracy                           0.84       666
       macro avg       0.84      0.82      0.83       666
    weighted avg       0.84      0.84      0.84       666
    
    test DecisionTreeClassifier
                  precision    recall  f1-score   support
    
               0       0.83      0.94      0.88       141
               1       0.87      0.66      0.75        82
    
        accuracy                           0.84       223
       macro avg       0.85      0.80      0.82       223
    weighted avg       0.84      0.84      0.83       223
    
    train VotingClassifier
                  precision    recall  f1-score   support
    
               0       0.84      0.95      0.89       408
               1       0.90      0.71      0.79       258
    
        accuracy                           0.85       666
       macro avg       0.87      0.83      0.84       666
    weighted avg       0.86      0.85      0.85       666
    
    test VotingClassifier
                  precision    recall  f1-score   support
    
               0       0.82      0.96      0.89       141
               1       0.91      0.63      0.75        82
    
        accuracy                           0.84       223
       macro avg       0.87      0.80      0.82       223
    weighted avg       0.85      0.84      0.84       223
    
    

#### Using the classification report vs. cross_val_score
- With the cross_val_score it's easier to take a look at the model running with different sampling and several iterations.
    - Using some statistical methods on the result helps me understand the level of robustness of each model
        - It seems as if the level of robustness drops sigificantly with test/new data - This may be to the fact, that the test data set is smaller
- Using the classification report helps me understand at which scoring method the models seem to perform better than other
    - In train and test data the models seem to perform equally good


```python


```
