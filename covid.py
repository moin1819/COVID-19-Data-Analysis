import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("covid_data.csv")

# Basic Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Remove duplicates
df.drop_duplicates(inplace=True)

# ----------------------------
# Total Cases by Country
# ----------------------------
if 'Country/Region' in df.columns:
    top_cases = df.groupby('Country/Region')['Confirmed'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    top_cases.plot(kind='bar')
    plt.title('Top 10 Countries by Confirmed Cases')
    plt.ylabel('Confirmed Cases')
    plt.show()

# ----------------------------
# Top Deaths by Country
# ----------------------------
if 'Country/Region' in df.columns:
    top_deaths = df.groupby('Country/Region')['Deaths'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    top_deaths.plot(kind='bar')
    plt.title('Top 10 Countries by Deaths')
    plt.ylabel('Deaths')
    plt.show()

# ----------------------------
# Correlation Heatmap
# ----------------------------
numeric_cols = df.select_dtypes(include='number')

plt.figure(figsize=(8,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ----------------------------
# Distribution of Confirmed Cases
# ----------------------------
if 'Confirmed' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df['Confirmed'], bins=30, kde=True)
    plt.title('Distribution of Confirmed Cases')
    plt.show()

# ----------------------------
# Cases Trend Over Time
# ----------------------------
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

    daily_cases = df.groupby('Date')['Confirmed'].sum()

    plt.figure(figsize=(12,5))
    plt.plot(daily_cases)
    plt.title('COVID-19 Confirmed Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.show()

# ----------------------------
# Top Countries Summary
# ----------------------------
if 'Country/Region' in df.columns:
    print("\nTop 10 Countries by Confirmed Cases:")
    print(df.groupby('Country/Region')['Confirmed']
          .sum()
          .sort_values(ascending=False)
          .head(10))