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

from sklearn.model_selection import RepeatedKFold, KFold, cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

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

# display the first five values of the target variable
print(y.head())


# set up repeated 5-fold cross-validation.
# the data is split into 5 folds (about 63 training and 16 testing
# observations each). every observation is used for testing exactly
# once per round, and the whole process is repeated 10 times with
# different shuffles, giving 5 x 10 = 50 model fits in total.
# this is more reliable than a single 80/20 split, where the results
# depend heavily on which 16 observations end up in the test set.

# random_state=42 makes the splits reproducible
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# train and test a new linear regression model on each of the 50 splits
# and record its score on the testing fold for each metric:
# - MAE: on average, how far predictions are from the actual values
# - MSE: average of the squared differences between actual and predicted
# - RMSE: square root of MSE, in the same units as academic development
# - R²: how much of the variation in academic development the model explains

# sklearn's scoring convention is "higher is better", so the
# error metrics (MAE, MSE, RMSE) are returned as negative numbers
cv_results = cross_validate(
    LinearRegression(),
    X,
    y,
    cv=cv,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "r2"
    ]
)

# flip the error metrics back to positive values
mae_scores = -cv_results["test_neg_mean_absolute_error"]
mse_scores = -cv_results["test_neg_mean_squared_error"]
rmse_scores = -cv_results["test_neg_root_mean_squared_error"]
r2_scores = cv_results["test_r2"]

# print the mean and standard deviation of each metric across the 50 fits.
# the mean tells us the model's typical performance, and the standard
# deviation tells us how much it changes depending on the split
print("\nLinear Regression Cross-Validation Results (5 folds x 10 repeats = 50 fits):")
print(f"MAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
print(f"MSE:  {mse_scores.mean():.6f} ± {mse_scores.std():.6f}")
print(f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
print(f"R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")


# create a baseline model to compare against.
# the dummy regressor does not use the predictor variables. in each
# cross-validation split, it predicts the same value for every test
# observation: the mean academic development of that split's training data.
# its scores provide a reference point for the linear regression scores
baseline = DummyRegressor(strategy="mean")

# evaluate the baseline with the same cross-validation splits
# and the same metrics as the linear regression model
baseline_results = cross_validate(
    baseline,
    X,
    y,
    cv=cv,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "r2"
    ]
)

# flip the error metrics back to positive values
baseline_mae = -baseline_results["test_neg_mean_absolute_error"]
baseline_mse = -baseline_results["test_neg_mean_squared_error"]
baseline_rmse = -baseline_results["test_neg_root_mean_squared_error"]
baseline_r2 = baseline_results["test_r2"]

# print the baseline results
print("\nBaseline Model (predicts the mean) Cross-Validation Results:")
print(f"MAE:  {baseline_mae.mean():.4f} ± {baseline_mae.std():.4f}")
print(f"MSE:  {baseline_mse.mean():.6f} ± {baseline_mse.std():.6f}")
print(f"RMSE: {baseline_rmse.mean():.4f} ± {baseline_rmse.std():.4f}")
print(f"R²:   {baseline_r2.mean():.4f} ± {baseline_r2.std():.4f}")


# compare the two models split by split.
# both models were evaluated with the same cv object, so position i in
# each score array comes from the same train/test split. comparing the
# arrays element by element gives True/False for each split, and .sum()
# counts the number of True values
n_splits = len(mae_scores)

lr_lower_mae = (mae_scores < baseline_mae).sum()
lr_lower_rmse = (rmse_scores < baseline_rmse).sum()
lr_higher_r2 = (r2_scores > baseline_r2).sum()

print(f"\nModel Comparison ({n_splits} matched CV splits):")
print(f"Linear Regression lower MAE:  {lr_lower_mae}/{n_splits} splits")
print(f"Linear Regression lower RMSE: {lr_lower_rmse}/{n_splits} splits")
print(f"Linear Regression higher R²:  {lr_higher_r2}/{n_splits} splits")


# cross-validation measures how well the model performs, but it fits
# 50 different models. to interpret the relationships, we fit one
# final model on all of the cleaned observations
model = LinearRegression()
model.fit(X, y)

# display the model's intercept.
# this is the predicted academic development value
# when all predictor variables are equal to zero
print("\nModel Intercept (fitted on full dataset):")
print(model.intercept_)

# display the coefficient for each predictor.
# each coefficient represents the predicted change in
# academic development associated with a one-unit
# increase in that predictor, while holding the other
# predictors constant.
print("\nModel Coefficients (fitted on full dataset):")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)


# get an out-of-fold prediction for every observation.
# the data is split into 5 folds, and each observation is predicted
# by a model that was trained on the other 4 folds (so it never saw
# that observation during training)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = cross_val_predict(LinearRegression(), X, y, cv=kfold)

# create a table comparing the actual Academic Development
# values with the out-of-fold predicted values.
results = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

# display the first 10 actual and predicted values.
print("\nActual vs. Predicted (out-of-fold):")
print(results.head(10))


# plot actual values against the out-of-fold predicted values.
plt.scatter(y, predictions)

# label the axes.
plt.xlabel("Actual Academic Development")
plt.ylabel("Predicted Academic Development")

# add a title.
plt.title("Actual vs. Predicted Academic Development (Out-of-Fold)")

# display the plot.
plt.show()
