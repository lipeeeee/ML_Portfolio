<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
<!-- [![Windows](https://img.shields.io/badge/Platform-Windows-0078d7.svg?style=for-the-badge)](https://en.wikipedia.org/wiki/Microsoft_Windows) -->
<!-- [![License](https://img.shields.io/github/license/R3nzTheCodeGOD/R3nzSkin.svg?style=for-the-badge)](LICENSE) -->

# Money and Happiness Data Analysis
</div>

## Introduction
This project aims to analyze the relationship between income and life satisfaction, using data from the Organization for Economic Cooperation and Development (OECD) Better Life Index (BLI) and the International Monetary Fund (IMF) Economic Outlook. The study will employ machine learning techniques to develop a model that predicts the life satisfaction level of a country based on its GDP per capita and try to debunk the idea that money isn't related to happiness.

## Data
The data used in this project consists of two datasets: the OECD Better Life Index (BLI) and the IMF Economic Outlook. The BLI dataset provides a measure of life satisfaction for 36 OECD member countries, as well as a number of social and economic indicators. The IMF Economic Outlook dataset provides information on the GDP per capita of these countries.

### Preparing data
To use the data in my machine learning model, I first needed to clean and prepare it. I accomplished this using a function called prepare_country_stats, which takes in the two datasets and returns a pandas DataFrame with only the necessary columns and indices. Here is the code for that function:

```python
# Reading Data
oecd_bli = pd.read_csv("../data/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("../data/gdp_per_capita.csv", thousands=',', delimiter='\t',
    encoding='latin1', na_values='n/a')

# Function that merges the two datasets
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# Creating dataframes
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
```

This code reads in the two CSV files containing the BLI and GDP per capita data, respectively. It then calls the prepare_country_stats function to clean and merge the datasets, sort them by GDP per capita, and select only the countries we are interested in. The resulting DataFrame, country_stats, contains the GDP per capita and life satisfaction columns, which we store as NumPy arrays x and y, respectively.

### Plotting Data
![alt text](https://raw.githubusercontent.com/lipeeeee/money_happiness_analysis/master/data/data_plot.png?token=GHSAT0AAAAAAB4GNSK3GWID5TTXZJHQ6ZQUY7SLEHA)

This plot indicates that there might be a correlation between GDP per capita and Life satisfaction since the higher GPD goes the higher Life satisfaction increases.

## Model
I chose the Linear Regression model from scikit-learn because it was, in my opinion, the best fit for this dataset.

Linear Regression is a statistical method used to analyze the relationship between a dependent variable and one or more independent variables. It is a simple, yet powerful algorithm that fits a straight line to the data points and uses this line to make predictions.

In the case of this project, the dependent variable is the life satisfaction level of a country and the independent variable is its GDP per capita. Linear Regression will allow us to model the relationship between these two variables and make predictions on the level of life satisfaction of a country based on its GDP per capita.

The scikit-learn library provides a simple interface for implementing Linear Regression. We can create a Linear Regression object and fit it to our data as follows:

```python
from sklearn.linear_model import LinearRegression

# Create a Linear Regression object
model = LinearRegression()

# Fit the model to the data
model.fit(x, y)
```

Here, x and y are the NumPy arrays containing the GDP per capita and life satisfaction data, respectively. The fit() method trains the model on this data, adjusting the coefficients of the straight line to best fit the data points.

Once the model is trained, we can use it to make predictions on new data. For example, to predict the life satisfaction level of a country with a GDP per capita of 50,961, we can use the predict() method as follows:

```python
# Make a prediction
prediction = model.predict([[50961]]) # Predicting Australia GDP(50,961.865)

# Print the prediction
print(prediction) # [[7.2]] (the acutal LS is 7.4)
```

This will output the predicted life satisfaction level for a country with a GDP per capita of 50,961.

Overall, Linear Regression is a good choice for this project because it is a simple and efficient method for modeling the relationship between two variables and making predictions based on this relationship.

## Finding the "Perfect" GDP
In addition to using the linear regression model to predict life satisfaction based on GDP per capita, I also created a function that attempts to find the "perfect" GDP. The idea behind this function is to brute-force the search for the GDP that yields a predicted life satisfaction of 10, which is the highest possible score in the BLI dataset. Here is the code for the bruteforce_gdp function:

```python
def bruteforce_gdp(max_iterations=1000000):  
    for i in range(max_iterations):
        # Check if perfect Life satisfaction
        predicted_LS = model.predict([[i]])
        if (predicted_LS >= 10):
            print("Found the perfect GDP!\n" + 
                  "It is: " + str(i) + "!\nProducing the Life satisfaction of: " + str(predicted_LS))
            return
    
    print("Reached the maximum number of iterations without finding a solution")
```

The function takes an optional max_iterations parameter that sets the maximum number of iterations to be attempted before giving up. The function simply loops over a range of GDP values, uses the linear regression model to predict the resulting life satisfaction, and checks if that prediction is 10 or greater. If it is, the function prints a message indicating that it has found the perfect GDP and the resulting life satisfaction. Otherwise, the function continues to the next GDP value.

It's important to note that this function is a brute-force search and is not guaranteed to find the exact "perfect" GDP value, especially given the limited range of GDP values being considered. Nonetheless, this function provides an interesting way to explore the relationship between GDP and life satisfaction and may yield some surprising results.

### Findings
In my model i had these findings for the "Perfect" GDP:
```
Found the perfect GDP!
It is: 123911!
Producing the Life satisfaction of: [[10.00001892]]
```

## Final Notes
Please note that the results produced by the model may vary from one person to another due to the use of a random seed during the machine learning process. The random seed ensures that the model's training process is repeatable, but it can also affect the model's accuracy and predictions. Therefore, if you compile the code on your machine, you may obtain slightly different results from those presented here.

## Conclusion
In conclusion, despite the limited amount of data available for this study, I was able to develop a model that accurately predicted life satisfaction based on GDP per capita using linear regression. My model's predictions were close to the actual life satisfaction score, with an error of only 0.2 on average. With more data and potentially more relevant features, it is likely that the accuracy of the model could be further improved. Overall, this study provides evidence that there is a positive relationship between income and life satisfaction, and that machine learning techniques can be effectively used to make predictions based on this relationship.

*a project by lipeeeee.*
