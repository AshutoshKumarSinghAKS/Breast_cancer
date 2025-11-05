import numpy as np
import pandas as pd

breast=pd.read_csv(r"C:\Users\lenovo\OneDrive\Documents\Flipkart\Breast Cancer Model\archive (1)\breast-cancer-data.csv")
breast.head()

breast['menopause'].value_counts()

breast.shape

breast.isnull().sum()

breast.duplicated().sum()

breast.info()

breast = breast.drop_duplicates()

breast.duplicated().sum()

# Drop rows with any null values
breast = breast.dropna()

# Drop columns with any null values
breast = breast.dropna(axis=1)

breast.isnull().sum()

breast.describe

# Remove trailing single quotes from all string columns
breast = breast.map(lambda x: x.strip("'") if isinstance(x, str) else x)

breast.head()

print(breast['tumer-size'].dtype)
print(breast['tumer-size'].value_counts())

print(breast['tumer-size'])

import pandas as pd

# Assuming your DataFrame is named 'breast'
# Function to calculate the mean of a range
def calculate_range_mean(range_str):
    # Check if the value is a string and contains '-'
    if isinstance(range_str, str) and '-' in range_str:
        start, end = map(int, range_str.split('-'))  # Split the range into start and end
        return (start + end) / 2  # Return the mean of the range
    return range_str  # If it's not a valid range, return the value as is (e.g., NaN or number)

# Apply the function to the 'tumer-size' column and replace the values with their mean
breast['tumer-size'] = breast['tumer-size'].apply(calculate_range_mean)

# Display the updated DataFrame
print(breast[['tumer-size']].head())  # Show the first few rows to check the result


breast['menopause'] = breast['menopause'].map({
    "premeno": -1,
    "ge40": 0,
    "lt40": 1,
    "unknown": -2  # Add mapping for unknown if needed
})

breast.head()





# Specify the column to be converted
column_to_convert = 'node-caps'

# Convert "yes" to 1 and "no" to 0
breast[column_to_convert] = breast[column_to_convert].map({'yes': 1, 'no': 0})

# Display the updated DataFrame
print(breast)

breast['menopause'].value_counts()

breast.isnull().sum()

breast.shape

breast.head

import pandas as pd

# Example of dataset loading
# breast = pd.read_csv('your_dataset.csv')

# Checking for missing values
print(breast.isnull().sum())

# Checking unique values in each column
for column in breast.columns:
    print(f"Unique values in {column}: {breast[column].unique()}")

# Checking for duplicates
print("Number of duplicate rows:", breast.duplicated().sum())

# Describing the dataset to spot irregularities
print(breast.describe(include='all'))


# Assuming `breast` is your cleaned DataFrame and `class` is your target column
x = breast.drop('menopause', axis=1)  # Features
y = breast['menopause']  # Target/Labels



x.shape

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train.shape

x_test.shape

x_train

x_test.shape

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
x_train['age'] = encoder.fit_transform(x_train['age'])

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# General function to encode categorical variables
categorical_cols =  breast.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    breast[col] = encoder.fit_transform( breast[col])

# Scale numerical columns
numerical_cols =  breast.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
breast[numerical_cols] = scaler.fit_transform( breast[numerical_cols])

# Display the preprocessed DataFrame
print( breast)

from sklearn .preprocessing import StandardScaler
sc=StandardScaler()

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Step 1: Check for missing values and handle them (if any)
if breast.isnull().sum().sum() > 0:
    print("Handling missing values...")
    breast.fillna(breast.mean(), inplace=True)  # Replace missing values with the column mean

# Step 2: Scale numerical columns
scaler = StandardScaler()
scaled_columns = ['age', 'menopause', 'tumer-size', 'inv-nodes', 
                  'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiate']
breast[scaled_columns] = scaler.fit_transform(breast[scaled_columns])

# Step 3: Confirm preprocessing
print("Processed dataset:")
print(breast.head())

from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Separate features (X) and target (y)
X = breast.drop('class', axis=1)
y = breast['class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Logistic Regression model
lg = LogisticRegression()
lg.fit(x_train, y_train)

# Predict the target for the test set
y_pred = lg.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

from sklearn.feature_selection import RFE

rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
x_train = rfe.fit_transform(x_train, y_train)
x_test = rfe.transform(x_test)

# from xgboost import XGBClassifier

# xgb = XGBClassifier(random_state=42)
# xgb.fit(x_train, y_train)
# y_pred_xgb = xgb.predict(x_test)

# accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
# print("XGBoost Accuracy:", accuracy_xgb)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
# Separate features (X) and target (y)
X = breast.drop('class', axis=1)
y = breast['class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Logistic Regression model
lg = LogisticRegression()
lg.fit(x_train, y_train)

# Predict the target for the test set
y_pred = lg.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

input=([ -0.61898587, -0.8905879 , -0.96613595, -0.02282475, -0.55223595,
        1.23482103,  1.03315813,  1.01035732,  1.78376517])
np_df=np.asarray(input)
prediction=lg.predict(np_df.reshape(1,-1))

if prediction[0]==1:
    print("cancerous")
else:
    print("not cancerous")

x_train[7]

import pickle
print("\n=== Saving Model ===")
with open('model.pkl', 'wb') as f:
    pickle.dump(lg, f)
print("Model saved as 'model.pkl'")

# Save the scaler as well (important for deployment)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'scaler.pkl'")







