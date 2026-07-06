# predict_model.py
import json
import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/predict_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_best_model():
    try:
        with open("reports/model_summary.json", "r") as f:
            summary = json.load(f)

        best_model_name = summary.get("best_model", "XGBoost")
        logger.info(f"Best model from summary: {best_model_name}")

        model_paths = {
            "RandomForest": "models/best_rf_model.pkl",
            "GradientBoosting": "models/best_gb_model.pkl",
            "XGBoost": "models/best_xgb_model.pkl",
        }

        model_path = model_paths.get(best_model_name, "models/best_xgb_model.pkl")

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        return model, best_model_name

    except Exception as e:
        logger.error(f"Error loading best model: {str(e)}", exc_info=True)
        logger.info("Falling back to XGBoost model")
        return joblib.load("models/best_xgb_model.pkl"), "XGBoost"


def make_predictions(model, X):
    try:
        logger.info(f"Making predictions on {len(X)} samples")

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        return y_pred, y_proba

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}", exc_info=True)
        raise


def evaluate_predictions(y_true, y_pred, y_proba, model_name, output_dir="reports/figures/prediction"):
    try:
        logger.info("Evaluating predictions")
        os.makedirs(output_dir, exist_ok=True)

        class_counts = np.bincount(y_true)
        total_samples = len(y_true)
        class_weights = total_samples / (len(class_counts) * class_counts)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        auc_score = roc_auc_score(y_true, y_proba)

        cm = confusion_matrix(y_true, y_pred)

        cr = classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"], output_dict=True)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"AUC: {auc_score:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name} Predictions")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_confusion_matrix.png")
        plt.close()

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker=".")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name} Predictions (AUC = {auc_score:.4f})")
        plt.grid(True)
        plt.savefig(f"{output_dir}/prediction_roc_curve.png")
        plt.close()

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name} Predictions")
        plt.grid(True)
        plt.savefig(f"{output_dir}/prediction_pr_curve.png")
        plt.close()

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc_score),
            "confusion_matrix": cm.tolist(),
            "classification_report": {
                k: (
                    {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in cr.items()
            },
            "class_weights": class_weights.tolist(),
            "class_distribution": {"not_fraud": int(class_counts[0]), "fraud": int(class_counts[1])},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating predictions: {str(e)}", exc_info=True)
        raise


def save_predictions(X, y_pred, y_proba, output_file="data/predictions/fraud_predictions.csv"):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        pred_df = X.copy()
        pred_df["predicted_fraud"] = y_pred
        pred_df["fraud_probability"] = y_proba

        pred_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

        return output_file

    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}", exc_info=True)
        raise


def under_sampling(df, n_sample):
    logger.info(f"Performing undersampling with {n_sample} samples from majority class")

    indices_class_0 = df[df["isFraud_encoded"] == 0].index

    sampled_indices_class_0 = pd.Index(pd.Series(indices_class_0).sample(n=n_sample))

    balanced_indices = sampled_indices_class_0.append(df[df["isFraud_encoded"] == 1].index)
    balanced_df = df.loc[balanced_indices]

    logger.info(f"Created balanced dataset with {len(balanced_df)} samples")
    return balanced_df


def evaluate_on_balanced_data(model, model_name):
    try:
        balanced_test_path = "data/evaluation/balanced_test_set.csv"

        if os.path.exists(balanced_test_path):
            logger.info(f"Loading saved balanced test set from {balanced_test_path}")
            balanced_test_df = pd.read_csv(balanced_test_path)

            if "isFraud_encoded" in balanced_test_df.columns:
                y_true = balanced_test_df["isFraud_encoded"]
                X_test = balanced_test_df.drop("isFraud_encoded", axis=1)

                logger.info(f"Loaded balanced test set with shape: {balanced_test_df.shape}")
                logger.info(f"Class distribution in balanced test set: {y_true.value_counts().to_dict()}")

                y_pred, y_proba = make_predictions(model, X_test)

                metrics = evaluate_predictions(
                    y_true, y_pred, y_proba, model_name, output_dir="reports/figures/balanced_prediction"
                )

                logger.info("Completed evaluation on saved balanced test set")
                return metrics, X_test, y_true, y_pred, y_proba
            else:
                logger.warning("Saved balanced test set doesn't have target column")
        else:
            logger.info("No saved balanced test set found, creating a new one")

        df = pd.read_hdf("data/processed/processed_transactions.h5")

        if "isFraud_encoded" not in df.columns:
            logger.info("Renaming target variable")
            df["isFraud_encoded"] = df["isFraud"]

        n_sample = df["isFraud_encoded"].value_counts()[1]
        logger.info(f"Using {n_sample} samples for undersampling (equal to fraud class count)")

        balanced_df = under_sampling(df, n_sample)

        X = balanced_df.drop("isFraud_encoded", axis=1)
        y = balanced_df["isFraud_encoded"]

        logger.info(f"Created balanced dataset with shape: {balanced_df.shape}")
        logger.info(f"Class distribution in balanced dataset: {y.value_counts().to_dict()}")

        y_pred, y_proba = make_predictions(model, X)

        metrics = evaluate_predictions(y, y_pred, y_proba, model_name, output_dir="reports/figures/balanced_prediction")

        logger.info("Completed evaluation on newly created balanced dataset")
        return metrics, X, y, y_pred, y_proba

    except Exception as e:
        logger.error(f"Error evaluating on balanced data: {str(e)}", exc_info=True)
        raise


def main(test_file=None):
    try:
        logger.info("Starting prediction pipeline")

        model, model_name = load_best_model()
        logger.info(f"Loaded {model_name} model")

        balanced_metrics, _, _, _, _ = evaluate_on_balanced_data(model, model_name)

        with open("reports/balanced_prediction_metrics.json", "w") as f:
            json.dump(balanced_metrics, f, indent=4)
        logger.info("Balanced prediction metrics saved to reports/balanced_prediction_metrics.json")

        if test_file:
            logger.info(f"Loading test data from {test_file}")
            if test_file.endswith(".h5"):
                df = pd.read_hdf(test_file)
            else:
                df = pd.read_csv(test_file)

            if "isFraud_encoded" in df.columns or "isFraud" in df.columns:
                target_col = "isFraud_encoded" if "isFraud_encoded" in df.columns else "isFraud"
                y_true = df[target_col]
                X = df.drop(target_col, axis=1)

                logger.info(f"Test data loaded with shape {X.shape}")
                logger.info(f"Target distribution: {y_true.value_counts().to_dict()}")

                y_pred, y_proba = make_predictions(model, X)

                metrics = evaluate_predictions(y_true, y_pred, y_proba, model_name)

                save_predictions(X, y_pred, y_proba)

                with open("reports/prediction_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info("Prediction metrics saved to reports/prediction_metrics.json")

                return metrics
            else:
                logger.info("No target column found, making predictions only")
                X = df

                y_pred, y_proba = make_predictions(model, X)

                save_predictions(X, y_pred, y_proba)

                logger.info("Predictions made without evaluation (no target column)")
                return {"status": "predictions_made_without_evaluation"}
        else:
            logger.info("No test file provided, using processed data for demonstration")

            df = pd.read_hdf("data/processed/processed_transactions.h5")

            if "isFraud_encoded" not in df.columns and "isFraud" in df.columns:
                df["isFraud_encoded"] = df["isFraud"]

            sample_df = df.sample(frac=0.1, random_state=42)
            y_true = sample_df["isFraud_encoded"]
            X = sample_df.drop("isFraud_encoded", axis=1)

            y_pred, y_proba = make_predictions(model, X)

            metrics = evaluate_predictions(y_true, y_pred, y_proba, model_name)

            save_predictions(X, y_pred, y_proba)

            with open("reports/prediction_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info("Prediction metrics saved to reports/prediction_metrics.json")

            return metrics

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        main(test_file)
    else:
        main()
