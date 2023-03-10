<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

# Titanic Survival Prediciton
</div>

# Introduction
This project's goal was to predict using Machine Learning, if a certain person would survive the titanic.

This was my first ever Machine Learning project. I used the kaggle titanic dataset to do it. This project's goal was purely to get me started in Data Science and Machine Learning.

# Data
Even though kaggle provides 2 csv files:
- One for model training;
- One for model validation;

The difference between them is the target column, "Survived", the training dataset had the column while the validation dataset didn't.

Because I was using scikit-learn I could only use 1 single dataset with the target column so I used the training datase(merging the two datasets would be impossible since *test.csv* has no target column).

![alt text](https://raw.githubusercontent.com/lipeeeee/ML_Portfolio/main/titanic_survival_pred/data/default_table_5.png)

### Preparing data
After analyzing the data I figured some features were useless for the Machine Learning model:
- PassengerId 
- Name
- SibSp
- Ticket
- Embarked

### Cabin
Other Fields were completely unusable, such as "Cabin" which out of 800 records only had 200 non-null values, so I had to remove the column completely before training the model.

### Age
I figured "Age" would be good for the model's predictions but, some fields had null values, to not waste the model's potential by deleting the entire column all together i filled the null values with the mean of Ages, this is a common practice in machine learning to deal with null values. 

```python
def clean_age(df):
    df['Age'] = df['Age'].fillna(value=df['Age'].mean())
    return df
```

### Sex
The problem I had with this column was that it wasn't a numeric value, so It was hard for the model to "eat" its values. To solve this I used binary encoding.
Turning "male" values into 1 and "female" into 0.

```python
def clean_sex(df):
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df
```

## Finishing up cleaning data
After encoding and deleting the columns we didn't want, I finished cleaning the data by specifying which columns I wanted using bracket notation.

```python
# Features we want
wanted_features = ['Fare', 'Parch', 'Pclass', 'Age', 'Sex']

def remove_unwanted_features(df, train=True):
    df = df[wanted_features]
    return df

x = remove_unwanted_features(x)
```

# Model
For the actual machine learning model i chose logistic regression, since it is widely used to predict binary outcomes.

## Logistic Regression
Logistic regression is a type of regression analysis used to predict the probability of a binary outcome (i.e. one that can take on only two values, such as true/false or yes/no). It is commonly used in machine learning for classification problems, where the goal is to predict which of several categories an input belongs to. Which in this case, we predict if either a person will survive or not.

Logistic Function:
```
f(x) = 1 / (1 + e^(-x))
```

## Model using scikit-learn

```python
# Choosing Logistic Regression
logisticReg = LogisticRegression()

# Splitting Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Training
logisticReg.fit(x_train, y_train)

# Calculating Score
score = logisticReg.score(x_test, y_test)
score # 80% accuracy
```

# Conclusion
This project gave me a great starting point for classification models and machine learning in general, ***the predictions had a 80% accuracy so It could predict If 4/5 would survive in the titanic***.

*a project by lipeeeee.*
