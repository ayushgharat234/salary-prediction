# train.py
import mlflow
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from preprocess import load_data, preprocess_data
import os

def train_models():
    df = load_data("data/salary_data.csv")
    X_train, X_test, y_train, y_test, scaler, lambda_bc = preprocess_data(df)

    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor(n_estimators=100)
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("src", exist_ok=True)

    best_model_name = None
    best_rmse = float("inf")

    with mlflow.start_run():
        # Save preprocessing objects
        joblib.dump(scaler, "src/scaler.pkl")
        joblib.dump(lambda_bc, "src/lambda_bc.pkl")

        mlflow.log_artifact("src/scaler.pkl", artifact_path="preprocessing")
        mlflow.log_artifact("src/lambda_bc.pkl", artifact_path="preprocessing")

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)

            mlflow.log_metric(f"{name}_rmse", rmse)

            model_path = f"models/{name}.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="models")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name

        mlflow.set_tag("best_model", best_model_name)
        print(f"Training complete. Best model: {best_model_name} with RMSE: {best_rmse}")

if __name__ == "__main__":
    train_models()