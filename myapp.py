import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Define file paths
csv_path = 'diabetes.csv'
img_path = 'img.jpeg'

# Check if the CSV file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} not found. Please ensure the file is in the correct directory.")

# Check if the image file exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f"File {img_path} not found. Please ensure the file is in the correct directory.")

# Load the diabetes dataset
diabetes_df = pd.read_csv(csv_path)

# Group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# Split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
Y = diabetes_df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Create an SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# Train the model on the training set
model.fit(X_train, Y_train)

# Make predictions on the training and testing sets
train_Y_pred = model.predict(X_train)
test_Y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_Y_pred, Y_train)
test_acc = accuracy_score(test_Y_pred, Y_test)

# Create the Streamlit app
def app():
    img = Image.open(img_path)
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)
    st.title('Diabetes Disease Prediction')
    st.sidebar.title('Input Features')

if __name__ == '__main__':
    app()
