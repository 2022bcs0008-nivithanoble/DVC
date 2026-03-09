import os, joblib, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = "data/data_partial.csv"
MODEL_PATH = "outputs/model/model.pkl"
METRICS_PATH = "outputs/metrics/results.json"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

X = df.drop(columns="median_house_value")
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(model, MODEL_PATH)

metrics = {
    "MSE": mse,
    "R2": r2
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Training completed")
print("MSE:", mse)
print("R2:", r2)
