import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('House_Price.csv')

# Handle missing values
df.dropna(inplace=True)

# Split the data into features (X) and target variable (y)
X = df[['living area', 'number of bedrooms', 'number of bathrooms']]
y = df['Price']

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse_value = mse(y_test, y_pred)
print(f'Mean Squared Error: {mse_value}')

# Visualize the results
plt.scatter(y_test, y_pred, color='gray')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()
