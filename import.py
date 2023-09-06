import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

# Set desired width for printing dataframe
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 18)

# Read car_data csv file
df = pd.read_csv('car_data.csv')

# Print first two and last three rows of dataframe
print(df.head(2))
