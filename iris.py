#imports of the libraries that are needed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#settings for the pandas
# This removes the limit on the number of displayed columns
pd.set_option('display.max_columns', None)

# If the text inside the cells is getting cut off, use this:
pd.set_option('display.max_colwidth', None)

# Load the iris dataset from seaborn
data = pd.read_csv("iris.csv")

#Looking at the first 10 rows of the dataset
print(data.head(10))

#Summary statistics of the dataset
print("The summary statistics of the dataset are:")
print(data.describe())

#class counts
print("The class counts of the dataset are:")
print(data['species'].value_counts())

#Missing Value checks
print("Missing values in each column:")
print(data.isnull().sum())

# Visualizing the distribution of each feature
data.hist(figsize=(10, 6), color='teal', edgecolor='grey', bins=20)
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()

plt.figure(figsize=(10, 8))

# Calculate the correlation matrix (only for numeric columns)
correlation_matrix = data.select_dtypes(include=['number']).corr()

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#Class Distribution
plt.figure(figsize=(6, 4))
# Replace 'Species' with your actual target column name
sns.countplot(x='species', data=data)
plt.title("Class Distribution")
plt.show()


#Training of Naive Bayes and Decision Tree Classifier