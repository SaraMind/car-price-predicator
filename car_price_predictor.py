import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import os

# Base path (same location as this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Images folder
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header=None)

# Set column names
columns = [
    "symboling","normalized_losses","make","fuel_type","aspiration",
    "num_doors","body_style","drive_wheels","engine_location","wheel_base",
    "length","width","height","curb_weight","engine_type","num_cylinders",
    "engine_size","fuel_system","bore","stroke","compression_ratio",
    "horsepower","peak_rpm","city_mpg","highway_mpg","price"
]
df.columns = columns

# Clean data
df.replace("?", np.nan, inplace=True)
numeric_cols = ["price", "horsepower", "engine_size", "curb_weight", "city_mpg", "highway_mpg"]
df[numeric_cols] = df[numeric_cols].astype(float)
df.dropna(subset=numeric_cols, inplace=True)

# Select relevant columns
df_clean = df[["engine_size", "horsepower", "curb_weight", "city_mpg", "highway_mpg", "price"]]

# Features and target
X = df_clean[["engine_size", "horsepower", "curb_weight", "city_mpg", "highway_mpg"]]
y = df_clean["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Visualization
# Price vs Engine Size
plt.figure(figsize=(6,4))
plt.scatter(df_clean["engine_size"], df_clean["price"])
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.title("Engine Size vs Price")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "engine_size_vs_price.png"))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(7,5))
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "correlation_heatmap.png"))
plt.show()

# Actual vs Predicted Prices
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "actual_vs_predicted.png"))
plt.show()

# GUI prediction function
def predict_price():
    try:
        data = [float(engine_size_entry.get()),
                float(horsepower_entry.get()),
                float(curb_weight_entry.get()),
                float(city_mpg_entry.get()),
                float(highway_mpg_entry.get())]
        user_data = pd.DataFrame([data], columns=X.columns)
        price = model.predict(user_data)[0]
        messagebox.showinfo("Predicted Price", f"Price: ${price:,.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Enter valid numbers")

# Create window
root = tk.Tk()
root.title("Car Price Predictor")

# Labels and entries
labels = ["Engine Size (cc)", "Horsepower", "Curb Weight (kg)", "City MPG", "Highway MPG"]
entries = []
for i, text in enumerate(labels):
    tk.Label(root, text=text).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

engine_size_entry, horsepower_entry, curb_weight_entry, city_mpg_entry, highway_mpg_entry = entries

# Predict button
tk.Button(root, text="Predict Price", command=predict_price).grid(row=len(labels), column=0, columnspan=2, pady=20)

# Run GUI
root.mainloop()