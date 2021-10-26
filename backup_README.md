# titanic analysis
A class prediction project based on the titanic dataset. The approach of using machine learning based on the results of data analysis to predict titanic survivors.

## 1. (Business) Goal

#### Goal of this project is to be able to build a model which is able to predict the circumstances on which the passengers of the titanic survived or not.

## 2. Get Data

#### I decided to get the data from seaborn to make sure other people can reproduce the same results on their device and herefore having access to the exact same dataset. The seaborn dataset looks mostly the same as the dataset given in the kaggle competition.

```markdown
# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = sns.load_dataset('titanic')

# first taking a look at the first lines to make sure import worked as expected.
df.head()
```
![First five rows](https://github.com/MichaMichalski/titanic-analysis/blob/main/pics/df_head.JPG)

df

