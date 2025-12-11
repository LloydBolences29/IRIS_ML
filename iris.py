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

#Looking at the first 5 rows of the dataset
print(data.head(10))