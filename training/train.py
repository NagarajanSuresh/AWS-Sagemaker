import os
import pandas as pd
# from sklearn.linear_model import LinearRegression
import joblib
import xgboost as xgb
import argparse
from sklearn.metrics import root_mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
parser.add_argument("--val", type=str, default="/opt/ml/input/data/val")
args, _ = parser.parse_known_args()


train_dir = args.train
val_dir   = args.val
model_dir = "/opt/ml/model"


train = pd.read_csv(os.path.join(train_dir, "train.csv"))
val   = pd.read_csv(os.path.join(val_dir, "val.csv"))


X_train = train.drop("price", axis=1)
y_train = train["price"]
X_val = val.drop("price", axis=1)
y_val = val["price"]

print(args.learning_rate)

model = xgb.XGBRegressor(
    learning_rate=args.learning_rate,
    max_depth=args.max_depth,
    n_estimators=args.n_estimators
)
model.fit(X_train, y_train)

preds = model.predict(X_val)
rmse = root_mean_squared_error(y_val, preds)
print(f"validation:rmse={rmse}")  # Metric for SageMaker to capture

os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.joblib"))


