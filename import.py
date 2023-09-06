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
print(df.tail(3))

# Check type of dataframe
print(type(df))

# Print shape of dataframe
print(df.shape)

# Check dataframe info
print(df.info())

# Print summary statistics of dataframe
print(df.describe())

# Replace 'Not available' values in Mileage column with NaN and convert column to float datatype
df['Mileage'] = df['Mileage'].replace('Not available', np.nan).str.replace(',', '').str.replace(' mi.', '').astype(float)

# Fill missing values in Mileage column with mean of column
df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())

# Check for missing values in other columns
print(df.isna().sum())

# Check for duplicate rows
print(df.duplicated().sum())

# Print unique values in Status column
print(df['Status'].unique())

# Remove rows with missing values in any column
df = df.dropna()

# Load the DataFrame from car_data.csv
car_data = pd.read_csv('car_data.csv')

# Convert the 'New_Price' column values to numeric data types
car_data['New_Price'] = pd.to_numeric(car_data['New_Price'], errors='coerce')

# Filter the DataFrame based on the specified condition
df_filtered = car_data[(car_data['New_Price'] >= 1500000) & (car_data['Year'] > 2022)]

# Plot the boxplot of the 'New_Price' column of the filtered DataFrame
if not df_filtered.empty:
    sn.boxplot(df_filtered['New_Price'])
else:
    print("No data to plot")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# create data frame
car_df = pd.DataFrame({'#': [1, 2, 3],
                       'Model': ['Honda', 'Toyota', 'Ford'],
                       'Year': [2010, 2015, 2018],
                       'Status': ['new', 'used', 'new'],
                       'Mileage': [50000, 80000, 20000],
                       'New_Price': [10000, 12000, 15000],
                       'MSRP': [15000, 18000, 20000]})

# convert New_Price column to numeric data type
car_df['New_Price'] = pd.to_numeric(car_df['New_Price'], errors='coerce')

# create column chart
plt.bar(car_df['Model'], car_df['New_Price'])
plt.title('New Car Prices by Model')
plt.xlabel('Model')
plt.ylabel('New Price')
plt.show()

# create pie chart
plt.pie(car_df['Mileage'], labels=car_df['Model'], autopct='%1.1f%%')
plt.title('Mileage by Model')
plt.show()

# create boxplot
sns.boxplot(x='Status', y='New_Price', data=car_df)
plt.title('New Car Prices by Status')
plt.show()

# create line chart
plt.plot(car_df['Year'], car_df['MSRP'], label='MSRP')
plt.plot(car_df['Year'], car_df['New_Price'], label='New Price')
plt.title('Car Prices Over Time')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.show()

import numpy as np
import pandas as pd

# create array with data type specification
car_array = np.array([(1, 'Honda', 2010, 'new', 50000, 10000, 15000),
                      (2, 'Toyota', 2015, 'used', 80000, 12000, 18000),
                      (3, 'Ford', 2018, 'new', 20000, 15000, 20000)],
                     dtype=[('id', 'i4'), ('model', 'U10'), ('year', 'i4'), ('status', 'U10'),
                            ('mileage', 'i4'), ('price', 'i4'), ('msrp', 'i4')])

# create matrix with data type specification
car_matrix = np.array([[1, 2010, 50000, 10000],
                       [2, 2015, 80000, 12000],
                       [3, 2018, 20000, 15000]], dtype='i4')

# create data frame
car_df = pd.DataFrame({'#': [1, 2, 3],
                       'Model': ['Honda', 'Toyota', 'Ford'],
                       'Year': [2010, 2015, 2018],
                       'Status': ['new', 'used', 'new'],
                       'Mileage': [50000, 80000, 20000],
                       'Price': [10000, 12000, 15000],
                       'MSRP': [15000, 18000, 20000]})

# calculate aggregate functions for array
print("Array Mean: ", np.mean(car_array['price']))
print("Array Sum: ", np.sum(car_array['price']))
print("Array Standard Deviation: ", np.std(car_array['price']))
print("Array Variance: ", np.var(car_array['price']))

# calculate aggregate functions for matrix
print("Matrix Mean: ", np.mean(car_matrix[:, 1]))
print("Matrix Sum: ", np.sum(car_matrix[:, 1]))
print("Matrix Standard Deviation: ", np.std(car_matrix[:, 1]))
print("Matrix Variance: ", np.var(car_matrix[:, 1]))

# calculate aggregate functions for data frame
print("Data Frame Mean: ", car_df['Price'].mean())
print("Data Frame Sum: ", car_df['Price'].sum())
print("Data Frame Standard Deviation: ", car_df['Price'].std())
print("Data Frame Variance: ", car_df['Price'].var())

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create data frame
car_df = pd.DataFrame({'#': [1, 2, 3],
                       'Model': ['Honda', 'Toyota', 'Ford'],
                       'Year': [2010, 2015, 2018],
                       'Status': ['new', 'used', 'new'],
                       'Mileage': [50000, 80000, 20000],
                       'New_Price': [10000, 12000, 15000],
                       'MSRP': [15000, 18000, 20000]})

# convert New_Price column to numeric data type
car_df['New_Price'] = pd.to_numeric(car_df['New_Price'], errors='coerce')

# define X and y
X = car_df[['#', 'Model', 'Year', 'Mileage', 'New_Price', 'MSRP']]
y = car_df['Status']

# convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Model'])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create decision tree classifier and fit to the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# make predictions on the test data and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# create data frame
car_df = pd.DataFrame({'#': [1, 2, 3],
                       'Model': ['Honda', 'Toyota', 'Ford'],
                       'Year': [2010, 2015, 2018],
                       'Status': ['new', 'used', 'new'],
                       'Mileage': [50000, 80000, 20000],
                       'New_Price': [10000, 12000, 15000],
                       'MSRP': [15000, 18000, 20000]})

# convert categorical variables to numerical variables
le = LabelEncoder()
car_df['Model'] = le.fit_transform(car_df['Model'])
car_df['Status'] = le.fit_transform(car_df['Status'])

# scale the data
scaler = StandardScaler()
car_df_scaled = scaler.fit_transform(car_df)

# perform cluster analysis using k-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(car_df_scaled)
labels = kmeans.labels_

# add cluster labels to data frame
car_df['Cluster'] = labels

# visualize the clusters using scatter plot
plt.scatter(car_df['New_Price'], car_df['MSRP'], c=labels)
plt.xlabel('New Price')
plt.ylabel('MSRP')
plt.title('Car Clusters')
plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# create data frame
car_df = pd.DataFrame({'#': [1, 2, 3],
                       'Model': ['Honda', 'Toyota', 'Ford'],
                       'Year': [2010, 2015, 2018],
                       'Status': ['new', 'used', 'new'],
                       'Mileage': [50000, 80000, 20000],
                       'New_Price': [10000, 12000, 15000],
                       'MSRP': [15000, 18000, 20000]})

# convert categorical variable Status to numerical
car_df['Status'] = pd.get_dummies(car_df['Status'], drop_first=True)

# create X and y variables
X = car_df[['Year', 'Mileage', 'Status']]
y = car_df['New_Price']

# fit linear regression model
reg = LinearRegression().fit(X, y)

# predict New Price based on new car data
new_car = [[2022, 0, 1]] # status = new, mileage = 0
predicted_price = reg.predict(new_car)
print("Predicted New Price: ${:,.2f}".format(predicted_price[0]))


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# read the dataset
car_df = pd.read_csv('car_data.csv')

# create list of transactions
transactions = []
for i in range(len(car_df)):
    transactions.append([str(car_df.values[i, j]) for j in range(1, len(car_df.columns))])

# perform one-hot encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# find frequent items
frequent_items = apriori(df, min_support=0.2, use_colnames=True)

# generate association rules
rules = association_rules(frequent_items, metric="lift", min_threshold=1)

# display the rules
print(rules)

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# create data frame
car_df = pd.DataFrame({'#': [1, 2, 3],
                       'Model': ['Honda', 'Toyota', 'Ford'],
                       'Year': [2010, 2015, 2018],
                       'Status': ['new', 'used', 'new'],
                       'Mileage': [50000, 80000, 20000],
                       'New_Price': [10000, 12000, 15000],
                       'MSRP': [15000, 18000, 20000]})

# compute correlation matrix
corr_matrix = car_df.corr()

# plot correlation matrix using heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# create data frame
car_df = pd.DataFrame({'#': [1, 2, 3],
                       'Model': ['Honda', 'Toyota', 'Ford'],
                       'Year': [2010, 2015, 2018],
                       'Status': ['new', 'used', 'new'],
                       'Mileage': [50000, 80000, 20000],
                       'New_Price': [10000, 12000, 15000],
                       'MSRP': [15000, 18000, 20000]})

# convert New_Price column to numeric data type
car_df['New_Price'] = pd.to_numeric(car_df['New_Price'], errors='coerce')

# select columns to perform PCA on
cols = ['Year', 'Mileage', 'New_Price', 'MSRP']
X = car_df[cols]

# instantiate PCA object
pca = PCA()

# fit and transform data
X_pca = pca.fit_transform(X)

# print explained variance ratio
print(pca.explained_variance_ratio_)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# load the dataset
car_df = pd.read_csv('car_data.csv')

# preprocess the text data
car_df['Model'] = car_df['Model'].str.lower()
car_df['Model'] = car_df['Model'].str.replace('[^\w\s]', '')
stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'of'}
car_df['Model'] = car_df['Model'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

# vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(car_df['Model'])
y = car_df['Status']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a naive Bayes classifier on the text data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# evaluate the performance of the model on the testing set
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion matrix:', confusion_matrix(y_test, y_pred))

# use the trained model to predict the status of new car models
new_cars = pd.DataFrame({'Model': ['honda civic', 'toyota corolla', 'ford mustang']})
new_cars['Model'] = new_cars['Model'].str.lower()
new_cars['Model'] = new_cars['Model'].str.replace('[^\w\s]', '')
new_cars['Model'] = new_cars['Model'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
X_new = vectorizer.transform(new_cars['Model'])
y_new = clf.predict(X_new)
print('Predicted status of new cars:', y_new)

