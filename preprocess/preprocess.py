import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--csv-name",type=str, required=True)
parser.add_argument("--train-dir", type=str, required=True)
parser.add_argument("--val-dir", type=str, required=True)
parser.add_argument("--test-dir", type=str, required=True)
args = parser.parse_args()


# Input/Output directories provided by SageMaker
input_path = os.path.join(args.input_dir, args.csv_name)

output_train = args.train_dir
output_val = args.val_dir
output_test = args.test_dir

output_scaler = "/opt/ml/processing/scaler"

# Load dataset
df = pd.read_csv(input_path)

# Encode binary yes/no columns
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].apply(lambda col: col.str.lower())
df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0}).astype(int)

# One-hot encode furnishingstatus
df = pd.get_dummies(df, columns=['furnishingstatus'])
df = df.astype({col: int for col in df.select_dtypes('bool').columns})

# split dataset into train, validation and test dataset
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Scale only the "area" column
scaler = StandardScaler()
train["area"] = scaler.fit_transform(train[["area"]])
val["area"] = scaler.transform(val[["area"]])
test["area"] = scaler.transform(test[["area"]])

# Ensure output dirs exist
os.makedirs(os.path.dirname(output_train), exist_ok=True)
os.makedirs(os.path.dirname(output_val), exist_ok=True)
os.makedirs(os.path.dirname(output_test), exist_ok=True)
os.makedirs(os.path.dirname(output_scaler), exist_ok=True)



# Save inside the provided directories
train.to_csv(os.path.join(output_train, "train.csv"), index=False)
val.to_csv(os.path.join(output_val, "val.csv"), index=False)
test.to_csv(os.path.join(output_test, "test.csv"), index=False)

joblib.dump(scaler, os.path.join(output_scaler,"scaler.pkl"))


