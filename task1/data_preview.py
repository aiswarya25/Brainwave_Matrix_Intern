
import pandas as pd

# Load the CSV files into pandas DataFrames
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Show some sample rows to verify
print("Fake News Sample:")
print(fake_df.head())

print("\nTrue News Sample:")
print(true_df.head())
