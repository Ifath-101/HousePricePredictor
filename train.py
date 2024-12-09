import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('houseprice.csv')
data.head()
data.info()
data.describe()
le = LabelEncoder()
data['furnishingstatus'] = le.fit_transform(data['furnishingstatus'])
data['furnishingstatus'].head()
data = pd.get_dummies(data,columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'],drop_first=True) 
data.head()
scaler = StandardScaler()
data[['area','bedrooms','bathrooms','stories','parking']] = scaler.fit_transform(data[['area','bedrooms','bathrooms','stories','parking']])
from sklearn.model_selection import train_test_split
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")
