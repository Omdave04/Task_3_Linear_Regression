import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Housing.csv")
df = pd.get_dummies(df, drop_first=True)

X_simple = df[['area']]
y = df['price']

X_multi = df.drop('price', axis=1)
y_multi = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

y_pred_s = model_simple.predict(X_test_s)
y_pred_m = model_multi.predict(X_test_m)

print("Simple Linear Regression")
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R²:", r2_score(y_test_s, y_pred_s))

print("\nMultiple Linear Regression")
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R²:", r2_score(y_test_m, y_pred_m))

plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', label='Prediction Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Area vs Price')
plt.legend()
plt.grid(True)
plt.show()

print("\nSimple Linear Coefficient:", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)

print("\nMultiple Linear Coefficients:")
for feature, coef in zip(X_multi.columns, model_multi.coef_):
    print(f"{feature}: {coef:.2f}")
print("Intercept:", model_multi.intercept_)
