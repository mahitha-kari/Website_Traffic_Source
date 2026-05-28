import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv(r"C:\Users\JC\OneDrive\Desktop\Summer internship\Project-2\website_traffic.csv")
print(df.head())

print("\nColumns:")
print(df.columns)

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove null values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Convert Date column
df['Date'] = pd.to_datetime(df['date'])

print("\nCleaned Data:")
print(df.head())

# -----------------------------
# ANALYSIS
# -----------------------------

# Monthly visitors
monthly_visitors = df.groupby(df['Date'].dt.month)['Visitors'].sum()

print("\nMonthly Visitors:")
print(monthly_visitors)

# Top traffic source
top_source = df.groupby('Traffic Sources')['Visitors'].sum()

print("\nTraffic Sources:")
print(top_source)

# -----------------------------
# VISUALIZATION
# -----------------------------

# Line Chart
monthly_visitors.plot(kind='line', marker='o')

plt.title("Monthly Website Traffic")
plt.xlabel("Month")
plt.ylabel("Visitors")

plt.show()

# Bar Chart
top_source.plot(kind='bar')

plt.title("Traffic Source Comparison")
plt.xlabel("Traffic Source")
plt.ylabel("Visitors")

plt.show()

# Pie Chart
top_source.plot(kind='pie', autopct='%1.1f%%')

plt.title("Traffic Source Distribution")

plt.ylabel("")

plt.show()

# Histogram
df['Bounce Rate'].plot(kind='hist', bins=10)

plt.title("Bounce Rate Distribution")
plt.xlabel("Bounce Rate")

plt.show()

# Scatter Plot
plt.scatter(df['Session Duration'], df['Visitors'])

plt.title("Session Duration vs Visitors")
plt.xlabel("Session Duration")
plt.ylabel("Visitors")

plt.show()

# Heatmap
numeric_df = df.select_dtypes(include='number')

sns.heatmap(numeric_df.corr(), annot=True)

plt.title("Correlation Heatmap")

plt.show()

# -----------------------------
# PREDICTION MODEL
# -----------------------------

X = np.array(monthly_visitors.index).reshape(-1, 1)

y = monthly_visitors.values

model = LinearRegression()

model.fit(X, y)

future_month = np.array([[6]])

prediction = model.predict(future_month)

print("\nPredicted Next Month Visitors:")
print(prediction[0])

# Prediction Graph
plt.plot(monthly_visitors.index, y, marker='o')

plt.plot(6, prediction[0], marker='o')

plt.title("Traffic Prediction")
plt.xlabel("Month")
plt.ylabel("Visitors")

plt.show()

# -----------------------------
# DASHBOARD SUMMARY
# -----------------------------

print("\n------ DASHBOARD ------")

print("Total Visitors:", df['Visitors'].sum())

print("Average Bounce Rate:", df['Bounce Rate'].mean())

print("Highest Visitors:", df['Visitors'].max())

print("Top Traffic Source:")

print(df.groupby('Traffic Sources')['Visitors'].sum().idxmax())