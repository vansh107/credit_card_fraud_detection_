# train_model.py
import json
import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/train_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def under_sampling(df, n_sample):
    """
    Apply undersampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input dataframe
        n_sample (int): Number of samples to keep from majority class

    Returns:
        pd.DataFrame: Balanced dataframe
    """
    logger.info(f"Performing undersampling with {n_sample} samples from majority class")

    # Get indices of class 0 samples (non-fraud)
    indices_class_0 = df[df["isFraud_encoded"] == 0].index

    # Randomly sample indices from class 0 to match the specified sample size
    # Note: we're not setting a random_state here to ensure different samples each time
    sampled_indices_class_0 = pd.Index(pd.Series(indices_class_0).sample(n=n_sample))

    # Combine the sampled indices of class 0 with indices of class 1
    balanced_indices = sampled_indices_class_0.append(df[df["isFraud_encoded"] == 1].index)
    balanced_df = df.loc[balanced_indices]

    logger.info(f"Created balanced dataset with {len(balanced_df)} samples")
    return balanced_df


def evaluate_model(model, X_test, y_test, model_name, output_dir="reports/figures"):
    """
    Evaluate model performance with various metrics.

    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for file naming
        output_dir: Directory to save plots

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info(f"Model: {model_name}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"AUC: {auc_score:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

    # Plot PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(recall_curve, precision_curve, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.grid(True)
    plt.savefig(f"{output_dir}/precision_recall_curve_{model_name}.png")
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker=".")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name} (AUC = {auc_score:.4f})")
    plt.grid(True)
    plt.savefig(f"{output_dir}/roc_curve_{model_name}.png")
    plt.close()

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_{model_name}.png")
    plt.close()

    # Return metrics as a dictionary
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(auc_score),
        "confusion_matrix": conf_matrix.tolist(),
        "model_name": model_name,
    }


def under_sampling_and_model(df, model_type, n_sample, n_iterations=10):
    """
    Perform undersampling and model training multiple times.

    Args:
        df (pd.DataFrame): Input dataframe
        model_type (str): Type of model to train ('RandomForest', 'GradientBoosting', 'XGBoost')
        n_sample (int): Number of samples for undersampling
        n_iterations (int): Number of iterations to run

    Returns:
        list: List of best models from each iteration
        dict: Average metrics across iterations
    """
    logger.info(f"Starting undersampling and {model_type} model training with {n_iterations} iterations")

    # Initialize metrics
    avg_accuracy = 0
    avg_fraud_precision = 0
    avg_fraud_recall = 0
    avg_fraud_f1 = 0
    avg_auc = 0
    f1_values = []
    best_models = []

    for i in range(n_iterations):
        logger.info(f"Iteration {i+1}/{n_iterations}")

        # Create balanced dataset
        balanced_df = under_sampling(df, n_sample)
        b_X = balanced_df.drop("isFraud_encoded", axis=1)
        b_y = balanced_df["isFraud_encoded"]

        # Split data - use stratify to maintain class balance in train/test sets
        X_train, X_test, y_train, y_test = train_test_split(b_X, b_y, test_size=0.2, stratify=b_y)

        # Initialize model and parameter grid based on model type - using notebook parameters
        if model_type == "RandomForest":
            model = RandomForestClassifier()
            param_grid = {
                "max_depth": list(range(2, 10)),
                "criterion": ["gini", "entropy"],
                "n_estimators": list(range(20, 200, 30)),
            }
            search_method = GridSearchCV

        elif model_type == "GradientBoosting":
            model = GradientBoostingClassifier()
            param_grid = {
                "n_estimators": list(range(60, 220, 40)),
                "learning_rate": [0.1, 0.01, 0.05],
                "max_depth": list(range(2, 10)),
            }
            search_method = GridSearchCV

        elif model_type == "XGBoost":
            model = xgb.XGBClassifier()
            param_grid = {
                "max_depth": list(range(2, 10)),
                "learning_rate": [0.1, 0.01, 0.05],
                "n_estimators": list(range(60, 220, 40)),
                "reg_lambda": [i / 10 for i in range(10, 51)],
                "reg_alpha": [i / 10 for i in range(0, 51)],
                "gamma": list(range(0, 11, 5)),
                "colsample_bytree": [i / 10 for i in range(6, 10)],
            }
            search_method = RandomizedSearchCV

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model with cross-validation - exactly like notebook implementation
        if model_type == "XGBoost":
            method_search = search_method(model, param_grid, cv=5, scoring="f1", n_iter=15)
        else:
            method_search = search_method(model, param_grid, cv=5, scoring="f1")

        method_search.fit(X_train, y_train)
        best_model = method_search.best_estimator_
        best_models.append(best_model)

        # Evaluate model
        y_pred = best_model.predict(X_test)

        # Calculate metrics - match notebook approach
        avg_auc += roc_auc_score(y_test, y_pred)  # Using y_pred like in notebook
        avg_accuracy += accuracy_score(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"], output_dict=True)

        avg_fraud_precision += class_rep["Fraud"]["precision"]
        avg_fraud_recall += class_rep["Fraud"]["recall"]
        avg_fraud_f1 += class_rep["Fraud"]["f1-score"]
        f1_values.append(round(class_rep["Fraud"]["f1-score"], 4))

        logger.info(f"Iteration {i+1} completed - F1 Score: {f1_values[-1]}")

    # Calculate average metrics
    avg_auc /= n_iterations
    avg_accuracy /= n_iterations
    avg_fraud_precision /= n_iterations
    avg_fraud_recall /= n_iterations
    avg_fraud_f1 /= n_iterations

    # Log results
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Fraud Precision: {avg_fraud_precision:.4f}")
    logger.info(f"Average Fraud Recall: {avg_fraud_recall:.4f}")
    logger.info(f"Average AUC: {avg_auc:.4f}")
    logger.info(f"Average Fraud F1-score: {avg_fraud_f1:.4f}")
    logger.info(f"F1 Scores: {f1_values}")
    logger.info(f"Max F1 Score: {max(f1_values)}")

    # Prepare metrics dictionary
    metrics = {
        "model_type": model_type,
        "avg_accuracy": float(avg_accuracy),
        "avg_fraud_precision": float(avg_fraud_precision),
        "avg_fraud_recall": float(avg_fraud_recall),
        "avg_fraud_f1": float(avg_fraud_f1),
        "avg_auc": float(avg_auc),
        "f1_values": f1_values,
        "max_f1": float(max(f1_values)),
        "best_params": method_search.best_params_,
    }

    return best_models, metrics


def find_best_model(models_list, X, y):
    """
    Find the best model from a list based on F1 score.

    Args:
        models_list (list): List of trained models
        X: Features to evaluate on
        y: Labels to evaluate against

    Returns:
        tuple: Best model and its F1 score
    """
    best_f1 = 0
    best_model = None

    for model in models_list:
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    return best_model, best_f1


def main():
    """Main function to run the model training pipeline"""
    try:
        logger.info("Starting model training pipeline")

        # Load processed data
        logger.info("Loading processed data")
        df = pd.read_hdf("data/processed/processed_transactions.h5")

        # Check if target variable exists
        if "isFraud_encoded" not in df.columns:
            logger.info("Renaming target variable")
            df["isFraud_encoded"] = df["isFraud"]

        # Determine sample size for undersampling (use the count of minority class)
        n_sample = df["isFraud_encoded"].value_counts()[1]
        logger.info(f"Using {n_sample} samples for undersampling (equal to fraud class count)")

        # Create a balanced dataset for final evaluation
        logger.info("Creating a balanced evaluation dataset")
        # First create a balanced dataset
        balanced_df_final = under_sampling(df, n_sample)

        # Split this balanced data for final evaluation
        X_balanced = balanced_df_final.drop("isFraud_encoded", axis=1)
        y_balanced = balanced_df_final["isFraud_encoded"]
        X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
            X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
        )

        # Save this balanced test set for later evaluation
        balanced_test_df = pd.concat([X_test_bal, y_test_bal], axis=1)
        os.makedirs("data/evaluation", exist_ok=True)
        balanced_test_df.to_csv("data/evaluation/balanced_test_set.csv", index=False)
        logger.info(f"Saved balanced test set with {len(balanced_test_df)} samples")

        # Train models using undersampling approach with the training portion
        train_df = df.copy()  # Use the entire dataset for training with undersampling

        # Train models using undersampling approach
        rf_models, rf_metrics = under_sampling_and_model(train_df, "RandomForest", n_sample, n_iterations=10)
        gb_models, gb_metrics = under_sampling_and_model(train_df, "GradientBoosting", n_sample, n_iterations=10)
        xgb_models, xgb_metrics = under_sampling_and_model(train_df, "XGBoost", n_sample, n_iterations=10)

        # Find the best model of each type using the balanced test set
        best_rf_model, best_rf_f1 = find_best_model(rf_models, X_test_bal, y_test_bal)
        best_gb_model, best_gb_f1 = find_best_model(gb_models, X_test_bal, y_test_bal)
        best_xgb_model, best_xgb_f1 = find_best_model(xgb_models, X_test_bal, y_test_bal)

        logger.info(f"Best RandomForest F1 on balanced test set: {best_rf_f1:.4f}")
        logger.info(f"Best GradientBoosting F1 on balanced test set: {best_gb_f1:.4f}")
        logger.info(f"Best XGBoost F1 on balanced test set: {best_xgb_f1:.4f}")

        # Save the best models
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_rf_model, "models/best_rf_model.pkl")
        joblib.dump(best_gb_model, "models/best_gb_model.pkl")
        joblib.dump(best_xgb_model, "models/best_xgb_model.pkl")
        logger.info("Best models saved to models/ directory")

        # Evaluate best models on balanced test set
        rf_eval = evaluate_model(best_rf_model, X_test_bal, y_test_bal, "RandomForest")
        gb_eval = evaluate_model(best_gb_model, X_test_bal, y_test_bal, "GradientBoosting")
        xgb_eval = evaluate_model(best_xgb_model, X_test_bal, y_test_bal, "XGBoost")

        # Determine overall best model based on F1 score on balanced test set
        models_eval = [
            ("RandomForest", best_rf_f1, rf_eval),
            ("GradientBoosting", best_gb_f1, gb_eval),
            ("XGBoost", best_xgb_f1, xgb_eval),
        ]

        best_model_name, best_model_f1, best_model_eval = max(models_eval, key=lambda x: x[1])
        logger.info(f"Overall best model: {best_model_name} with F1 Score: {best_model_f1:.4f}")

        # Save evaluation metrics for all models
        metrics = {
            "RandomForest": {"training_metrics": rf_metrics, "balanced_test_metrics": rf_eval},
            "GradientBoosting": {"training_metrics": gb_metrics, "balanced_test_metrics": gb_eval},
            "XGBoost": {"training_metrics": xgb_metrics, "balanced_test_metrics": xgb_eval},
            "best_model": {
                "name": best_model_name,
                "f1_score": float(best_model_f1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        # Save metrics to file
        with open("reports/model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Model metrics saved to reports/model_metrics.json")

        # Save a summary of model performance
        with open("reports/model_summary.json", "w") as f:
            summary = {
                "best_model": best_model_name,
                "best_model_f1": float(best_model_f1),
                "model_comparison": {
                    "RandomForest": {
                        "f1": float(best_rf_f1),
                        "precision": float(rf_eval["precision"]),
                        "recall": float(rf_eval["recall"]),
                    },
                    "GradientBoosting": {
                        "f1": float(best_gb_f1),
                        "precision": float(gb_eval["precision"]),
                        "recall": float(gb_eval["recall"]),
                    },
                    "XGBoost": {
                        "f1": float(best_xgb_f1),
                        "precision": float(xgb_eval["precision"]),
                        "recall": float(xgb_eval["recall"]),
                    },
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            json.dump(summary, f, indent=4)
        logger.info("Model summary saved to reports/model_summary.json")

        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
