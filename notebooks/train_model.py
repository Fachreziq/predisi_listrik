import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error

# ======================
# LOAD DATASET
# ======================

df = pd.read_csv(
    'data/raw/household_power_consumption.csv',
    low_memory=False
)

print(df.columns)

# ======================
# PREPROCESSING
# ======================

columns = [
    'Global_active_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]

df = df[columns]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# ======================
# FITUR & TARGET
# ======================

X = df[[
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]]

y = df['Global_active_power']

# ======================
# SPLIT DATA
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================
# FOLDER MODELS
# ======================

os.makedirs('models', exist_ok=True)

# ======================
# MODEL LIST
# ======================

models = {
    "linear_regression.pkl": LinearRegression(),
    "decision_tree.pkl": DecisionTreeRegressor(),
    "random_forest.pkl": RandomForestRegressor(),
    "knn.pkl": KNeighborsRegressor()
}

# ======================
# TRAIN & SAVE
# ======================

for filename, model in models.items():

    print(f"\nTraining {filename}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    print("MAE:", mae)

    joblib.dump(model, f"models/{filename}")

    print(f"{filename} berhasil disimpan!")

print("\nSemua model berhasil dibuat!")