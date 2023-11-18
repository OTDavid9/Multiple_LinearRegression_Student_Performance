Student Academic Performance Prediction
This Python code aims to predict student academic performance based on various factors such as hours studied, previous scores, sleep hours, sample question papers practiced, and extracurricular activities. The predictive model is implemented using the Linear Regression algorithm from the scikit-learn library.

Prerequisites
Before running the code, ensure that you have the required Python packages installed. You can install them using the following command:


pip install pandas scikit-learn matplotlib

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
path = "Student_Performance.csv"
df = pd.read_csv(path)

# Display the first few rows of the dataset
print(df.head())

# Display information about the dataset
print(df.info())

# Convert the "Extracurricular Activities" column to numerical values (0 or 1)
Extracurricular_Activities = []
for i in df["Extracurricular Activities"]:
    if i == "Yes":
        Extracurricular_Activities.append(1)
    else:
        Extracurricular_Activities.append(0)

df['Extracurricular_Activities'] = Extracurricular_Activities

# Drop the original "Extracurricular Activities" column
df.drop(columns=['Extracurricular Activities'], inplace=True)

# Display information about the modified dataset
print(df.info())

# Visualize the correlation between each feature and the target variable
# Scatter plots with correlation coefficients are generated for each feature
# The significance level is determined by the p-value
# The plots help understand the linear relationship between features and the target variable
# Adjust the target variable name if necessary
%matplotlib inline
for column in df.columns:
    if column != "Performance Index":
        x = df[column]
        y = df["Performance Index"]
        plt.scatter(x, y)
        plt.title(f"Scatter Plot of {column} vs Performance Index")
        plt.xlabel(column)
        plt.ylabel("Performance Index")
        correlation_coefficient, p_value = pearsonr(x, y)
        if p_value < 0.05:
            if correlation_coefficient > 0.7:
                plt.text(0.5, 0.9, f"Strong positive correlation\nCorrelation Coefficient: {correlation_coefficient:.2f}\nP-Value: {p_value:.4f} (significant)", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            elif correlation_coefficient < -0.7:
                plt.text(0.5, 0.9, f"Strong negative correlation\nCorrelation Coefficient: {correlation_coefficient:.2f}\nP-Value: {p_value:.4f} (significant)", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            else:
                plt.text(0.5, 0.9, f"Weak or no correlation\nCorrelation Coefficient: {correlation_coefficient:.2f}\nP-Value: {p_value:.4f} (significant)", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.9, f"Weak or no correlation\nCorrelation Coefficient: {correlation_coefficient:.2f}\nP-Value: {p_value:.4f} (not significant)", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.show()

# Prepare the features (X) and target variable (y)
columns = df.columns.tolist()
X = df[columns[:-1]]
y = df[columns[-1]]

# Create a linear regression model
reg = linear_model.LinearRegression()

# Fit the model on the entire dataset
reg.fit(X, y)

# Display the intercept and coefficients of the linear regression model
print("Intercept:", reg.intercept_)
print("Coefficients:", reg.coef_)

# Make a prediction for a sample input (adjust values if necessary)
prediction = reg.predict([[7, 99, 9, 1, 1]])
print("Predicted Performance Index:", prediction[0][0])

# Split the data into training and testing sets
features = X
target = y
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a linear regression model
reg = LinearRegression()

# Fit the model on the training data
reg.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = reg.predict(X_test)

# Evaluate the model's performance on the test set using the R-squared score
r_squared = reg.score(X_test, y_test).round(5)

# Print or use the R-squared score
print(f'R-squared Score: {r_squared}')
Output Interpretation
The initial data exploration includes loading the dataset, displaying its structure, and converting categorical data into numerical format.
Scatter plots with correlation coefficients are generated to visualize the relationship between each feature and the target variable (Performance Index).
The Linear Regression model is trained on the entire dataset, and the intercept and coefficients are displayed.
A sample prediction is made using the trained model.
The dataset is split into training and testing sets, and the R-squared score is calculated to evaluate the model's performance on the test set.