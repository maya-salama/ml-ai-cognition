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

print("\nSummary statistics:")
print(data.describe())

import matplotlib.pyplot as plt

plt.hist(data["AI Use Frequency"], bins=10)

plt.xlabel("AI Use Frequency")
plt.ylabel("Number of Observations")
plt.title("Distribution of AI Use Frequency")

plt.show()

plt.hist(data["Academic Development"], bins=10)

plt.xlabel("Academic Development")
plt.ylabel("Number of Observations")
plt.title("Distribution of Academic Development")

plt.show()

plt.scatter(data["AI Use Frequency"], data["Academic Development"])

plt.xlabel("AI Use Frequency")
plt.ylabel("Academic Development")
plt.title("AI Use Frequency vs. Academic Development")

plt.show()

correlation = data["AI Use Frequency"].corr(data["Academic Development"])

print("\nCorrelation between AI Use Frequency and Academic Development:")
print(correlation)

print("\nCorrelations with Academic Development:")
print(data[[
    "AI Use Frequency",
    "Pell Grant Recipient",
    "First gen student",
    "Academic Development"
]].corr())

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = data[[
    "AI Use Frequency",
    "Academic Development",
    "Pell Grant Recipient",
    "First gen student"
]].corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

plt.title("Correlation Matrix")
plt.show()