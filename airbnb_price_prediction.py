import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("AB_NYC_2019.csv")

# Data cleaning
df = df[df['price'] > 0]

df = df[['price', 'neighbourhood_group', 'room_type',
         'minimum_nights', 'number_of_reviews',
         'reviews_per_month', 'availability_365']]

df['reviews_per_month'].fillna(0, inplace=True)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

categorical_features = ['neighbourhood_group', 'room_type']
numerical_features = ['minimum_nights', 'number_of_reviews',
                       'reviews_per_month', 'availability_365']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Sample prediction
sample = pd.DataFrame({
    'neighbourhood_group': ['Manhattan'],
    'room_type': ['Entire home/apt'],
    'minimum_nights': [3],
    'number_of_reviews': [20],
    'reviews_per_month': [1.5],
    'availability_365': [200]
})

print("Predicted Airbnb price:", model.predict(sample)[0])
