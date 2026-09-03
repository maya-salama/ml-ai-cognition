import pandas as pd

data = pd.read_excel("data/ucues_2024.xlsx")

print("Original dataset shape:")
print(data.shape)

# Remove rows that don't contain a valid campus
data = data.dropna(subset=["Campus"])

# Remove observations missing data needed for modeling
data = data.dropna()

print("\nCleaned dataset shape:")
print(data.shape)

print("\nMissing values after cleaning:")
print(data.isnull().sum())