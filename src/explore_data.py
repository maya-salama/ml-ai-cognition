import pandas as pd

# load the ucues 2024 dataset from the excel file
data = pd.read_excel("data/ucues_2024.xlsx")

# display the number of rows & columns in the original dataset
print("Original dataset shape:")
print(data.shape)

# remove rows that don't contain a valid campus
data = data.dropna(subset=["Campus"])

# remove observations missing data needed for modeling
data = data.dropna()

# display the size of dataset after cleaning
print("\nCleaned dataset shape:")
print(data.shape)

# check that there are no missing values remaining 
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# display summary stats such as mean, sd, min, max, 
# and quartiles for the numerical variables
print("\nSummary statistics:")
print(data.describe())

import matplotlib.pyplot as plt

# create a histogram showing how frequently students use AI
plt.hist(data["AI Use Frequency"], bins=10)

plt.xlabel("AI Use Frequency")
plt.ylabel("Number of Observations")
plt.title("Distribution of AI Use Frequency")

# display histogram
plt.show()

# create a histogram showing the distribution of academic 
# development scores
plt.hist(data["Academic Development"], bins=10)

plt.xlabel("Academic Development")
plt.ylabel("Number of Observations")
plt.title("Distribution of Academic Development")

# display histogram
plt.show()

# create a scatter plot to visualize the relationship beyween
# ai use frequency and academic development
plt.scatter(data["AI Use Frequency"], data["Academic Development"])

plt.xlabel("AI Use Frequency")
plt.ylabel("Academic Development")
plt.title("AI Use Frequency vs. Academic Development")

# display scatterplot
plt.show()

# calculate correlation between ai use frequency 
# and academic development.
# correlation ranges from -1 to 1:
# - 1 means a strong positive relationship
# - 0 means little/no linear relationship
# - -1 means a strong negative relationship
correlation = data["AI Use Frequency"].corr(data["Academic Development"])

print("\nCorrelation between AI Use Frequency and Academic Development:")
print(correlation)

# calculate correlations between academic development
# and other variables we're interested in
print("\nCorrelations with Academic Development:")
print(data[[
    "AI Use Frequency",
    "Pell Grant Recipient",
    "First gen student",
    "Academic Development"
]].corr())

import seaborn as sns
import matplotlib.pyplot as plt

# select the variables we want to compare in the correlation matrix
correlation_matrix = data[[
    "AI Use Frequency",
    "Academic Development",
    "Pell Grant Recipient",
    "First gen student"
]].corr()

# create a heatmap showing the correlations between the variables.
# annot=True displays the numerical correlation values.
# vmin and vmax set the scale from -1 to 1.
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Matrix")

# display heatmap
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# define target variable (y) - the variable we want the model to predict
y = data["Academic Development"]

# define the predictor variables (X) - the variables 
# the model will use to make predictions about academic development
X = data[[
    "AI Use Frequency",
    "Pell Grant Recipient",
    "First gen student"
]]

# display the first five rows of predictor variables
print(X.head())

# display the first five values of the tatrget variable
print(y.head())



# split the dataset into training and testing sets
# 80% of observations are used to train model
# 20% are kept separate so we can test the model on data
# it did not see during training

# random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# display the number of observations in each set
print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# create a linear regression model
model = LinearRegression()

# train the model using the training data.
# the model learns how the predictor variables (X_train)
# relate to the target variable (y_train)
model.fit(X_train, y_train)

# use the trained model to predict academic development
# for the observations in the testing set
predictions = model.predict(X_test)

# calculate the mean absolute error (MAE)
# this tells, on average, how far the model's predictions
# are from the actual academic development values 

mae = mean_absolute_error(y_test, predictions)

# calculate the mean squared error (MSE).
# this measures prediction error by squaring the differences
# between the actual and predicted values.

mse = mean_squared_error(y_test, predictions)

# calculate the Root Mean Squared Error (RMSE).
# RMSE is the square root of MSE and is expressed
# in the same units as Academic Development.
rmse = np.sqrt(mse)


# calculate R-squared (R²).
# R² measures how much of the variation in Academic Development
# is explained by the variables in our linear regression model.
r2 = r2_score(y_test, predictions)


# print the model's evaluation results.
print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# display the model's intercept.
# this is the predicted academcic development value
# when all predictor variables are equal to zero

print("\nModel Intercept:")
print(model.intercept_)

# display the coefficient for each predictor.
# each coefficient represents the predicted change in
# academic development associated with a one-unit
# increase in that predictor, while holding the other
# predictors constant. 
print("\nModel Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)


# create a table comparing the actual Academic Development
# values with the values predicted by our model.
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

# display the first 10 actual and predicted values.
print("\nActual vs. Predicted:")
print(results.head(10))


# plot actual values against the model's predicted values.
plt.scatter(y_test, predictions)

# label the axes.
plt.xlabel("Actual Academic Development")
plt.ylabel("Predicted Academic Development")

# add a title.
plt.title("Actual vs. Predicted Academic Development")

# display the plot.
plt.show()