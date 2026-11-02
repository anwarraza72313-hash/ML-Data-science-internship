import numpy as np
from sklearn.linear_model import LinearRegression

# Training Data
# Study hours
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

# Student marks
y = np.array([
    20, 25, 35, 40, 50,
    60, 65, 75, 85, 95
])

# Create Machine Learning Model
model = LinearRegression()

# Train the model
model.fit(X, y)

print("Student Marks Prediction System")

# Input from user
hours = float(input("Enter study hours: "))

# Predict marks
prediction = model.predict([[hours]])

print("Predicted Marks:", round(prediction[0], 2))