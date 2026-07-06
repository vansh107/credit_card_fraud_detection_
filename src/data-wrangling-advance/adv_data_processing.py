# adv_data_processing.py
import json
import logging
import os
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from src.logger import CustomLogger, create_log_path
from tabulate import tabulate

warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)

# Setup logging
log_file_path = create_log_path("advanced_wrangling")
logger = CustomLogger(logger_name="advanced_wrangling", log_filename=log_file_path)
logger.set_log_level(level=logging.INFO)


class TransactionAnalyzer:
    """Class to handle advanced transaction analysis"""

    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}
        self.exploration_text = []
        self.transaction_stats = []

    def save_transaction_analysis(self):
        """Save transaction analysis to file"""
        try:
            os.makedirs("reports", exist_ok=True)
            with open("reports/advanced_data_exploration.txt", "w") as f:
                f.write("ADVANCED TRANSACTION ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write("Timestamp: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("-" * 50 + "\n\n")
                f.write("\n".join(self.transaction_stats))
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("\n".join(self.exploration_text))

            logger.save_logs("Transaction analysis saved successfully", log_level="info")
        except Exception as e:
            logger.save_logs(f"Error saving transaction analysis: {str(e)}", log_level="error")
            raise

    def identify_reversed_transactions(self):
        """Identify and analyze reversed transactions"""
        try:
            # Add analysis header to transaction stats
            self.transaction_stats.append("1. REVERSED TRANSACTIONS ANALYSIS")
            self.transaction_stats.append("-" * 30 + "\n")

            selected_columns = [
                "accountNumber",
                "customerId",
                "creditLimit",
                "transactionAmount",
                "merchantName",
                "cardLast4Digits",
                "accountOpenDate",
                "cardCVV",
                "enteredCVV",
                "expirationDateKeyInMatch",
                "isFraud",
            ]

            # Extract subsets
            df_PURCHASE = self.df[selected_columns][self.df["transactionType"] == "PURCHASE"]
            df_REVERSAL = self.df[selected_columns][self.df["transactionType"] == "REVERSAL"]

            # Log initial counts
            self.transaction_stats.append(f"Total PURCHASE transactions: {len(df_PURCHASE)}")
            self.transaction_stats.append(f"Total REVERSAL transactions: {len(df_REVERSAL)}\n")

            # Find reversed transactions
            purchased_set = set(map(tuple, df_PURCHASE.values))
            reversal_set = set(map(tuple, df_REVERSAL.values))
            reversed_transactions = purchased_set.intersection(reversal_set)

            # Convert to DataFrame
            reversed_df = pd.DataFrame(list(reversed_transactions), columns=selected_columns)

            # Calculate metrics
            not_linked_count = len(self.df[self.df["transactionType"] == "REVERSAL"]) - len(reversed_df)
            not_linked_sum = (
                self.df[self.df["transactionType"] == "REVERSAL"]["transactionAmount"].sum()
                - reversed_df["transactionAmount"].sum()
            )

            # Add detailed statistics
            self.transaction_stats.append("Reversed Transactions Statistics:")
            self.transaction_stats.append(f"- Total Reversed Transactions: {len(reversed_df)}")
            self.transaction_stats.append(f"- Total Reversed Amount: ${reversed_df['transactionAmount'].sum():,.2f}")
            self.transaction_stats.append(f"- Average Reversed Amount: ${reversed_df['transactionAmount'].mean():,.2f}")
            self.transaction_stats.append(
                f"- Median Reversed Amount: ${reversed_df['transactionAmount'].median():,.2f}"
            )
            self.transaction_stats.append(f"\nUnlinked Reversal Statistics:")
            self.transaction_stats.append(f"- Unlinked Reversal Count: {not_linked_count}")
            self.transaction_stats.append(f"- Unlinked Reversal Amount: ${not_linked_sum:,.2f}\n")

            # Add fraud analysis for reversed transactions
            fraud_stats = reversed_df.groupby("isFraud")["transactionAmount"].agg(["count", "sum", "mean"])
            self.transaction_stats.append("Fraud Analysis in Reversed Transactions:")
            self.transaction_stats.append(tabulate(fraud_stats, headers="keys", tablefmt="grid"))
            self.transaction_stats.append("")

            # Record metrics
            self.metrics["reversed_transactions"] = {
                "count": len(reversed_df),
                "total_amount": float(reversed_df["transactionAmount"].sum()),
                "not_linked_count": not_linked_count,
                "not_linked_amount": float(not_linked_sum),
                "fraud_distribution": fraud_stats.to_dict(),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.save_logs("Reversed transactions analysis completed", log_level="info")
            return reversed_df

        except Exception as e:
            logger.save_logs(f"Error in reversed transactions analysis: {str(e)}", log_level="error")
            raise

    def identify_multi_swipe(self, time_window=5):
        """Identify and analyze multi-swipe transactions"""
        try:
            # Add analysis header to transaction stats
            self.transaction_stats.append("2. MULTI-SWIPE TRANSACTIONS ANALYSIS")
            self.transaction_stats.append("-" * 30 + "\n")

            # Filter purchase transactions
            df_purchase = self.df[self.df.transactionType == "PURCHASE"].copy()  # Create a copy
            self.transaction_stats.append(f"Total PURCHASE transactions analyzed: {len(df_purchase)}")

            # Find duplicated purchases
            duplicated_cols = ["accountNumber", "customerId", "transactionAmount", "merchantName", "cardLast4Digits"]
            df_duplicated = df_purchase[df_purchase.duplicated(duplicated_cols, keep=False)].copy()  # Create a copy

            # Calculate time differences
            df_duplicated["timeDifference"] = df_duplicated.groupby(["accountNumber", "customerId", "merchantName"])[
                "transactionDateTime"
            ].diff()
            # Identify multi-swipe transactions
            multi_swipe_df = df_duplicated[df_duplicated["timeDifference"] < pd.Timedelta(minutes=time_window)]

            # Add detailed statistics
            self.transaction_stats.append(f"\nMulti-Swipe Statistics (Time Window: {time_window} minutes):")
            self.transaction_stats.append(f"- Total Multi-Swipe Transactions: {len(multi_swipe_df)}")
            self.transaction_stats.append(
                f"- Total Multi-Swipe Amount: ${multi_swipe_df['transactionAmount'].sum():,.2f}"
            )
            self.transaction_stats.append(
                f"- Average Multi-Swipe Amount: ${multi_swipe_df['transactionAmount'].mean():,.2f}"
            )
            self.transaction_stats.append(
                f"- Median Multi-Swipe Amount: ${multi_swipe_df['transactionAmount'].median():,.2f}"
            )

            # Time difference analysis
            time_diff_stats = multi_swipe_df["timeDifference"].describe()
            self.transaction_stats.append("\nTime Difference Analysis:")
            self.transaction_stats.append(
                tabulate(time_diff_stats.to_frame(), headers=["Time Difference Statistics"], tablefmt="grid")
            )

            # Merchant analysis
            merchant_stats = multi_swipe_df.groupby("merchantName").size().sort_values(ascending=False).head()
            self.transaction_stats.append("\nTop Merchants with Multi-Swipe Transactions:")
            self.transaction_stats.append(tabulate(merchant_stats.to_frame(), headers=["Count"], tablefmt="grid"))

            # Convert timedelta statistics to strings for JSON serialization
            time_diff_stats_dict = {}
            for key, value in time_diff_stats.to_dict().items():
                if isinstance(value, pd.Timedelta):
                    time_diff_stats_dict[key] = str(value)
                else:
                    time_diff_stats_dict[key] = value

            # Record metrics with serializable values
            self.metrics["multi_swipe_transactions"] = {
                "count": len(multi_swipe_df),
                "total_amount": float(multi_swipe_df["transactionAmount"].sum()),
                "time_window_minutes": time_window,
                "time_difference_stats": time_diff_stats_dict,  # Use the converted dictionary
                "top_merchants": merchant_stats.to_dict(),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Save analysis
            self.save_transaction_analysis()

            logger.save_logs("Multi-swipe transactions analysis completed", log_level="info")
            return multi_swipe_df

        except Exception as e:
            logger.save_logs(f"Error in multi-swipe analysis: {str(e)}", log_level="error")
            raise


class FeatureProcessor:
    """Class to handle feature processing and engineering"""

    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}
        self.processing_text = []

    def remove_special_transactions(self, reversed_df, multi_swipe_df):
        """Remove reversed and multi-swipe transactions"""
        try:
            initial_rows = len(self.df)

            # Remove reversed transactions
            merged_df = self.df.merge(reversed_df, on=reversed_df.columns.to_list(), how="left", indicator=True)
            self.df = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])

            # Remove multi-swipe transactions - Fix the columns to merge on
            # Use only the common columns between self.df and multi_swipe_df
            common_columns = list(set(self.df.columns) & set(multi_swipe_df.columns))
            merged_df_ms = self.df.merge(multi_swipe_df[common_columns], on=common_columns, how="left", indicator=True)
            self.df = merged_df_ms[merged_df_ms["_merge"] == "left_only"].drop(columns=["_merge"])

            # Reset index
            self.df.reset_index(drop=True, inplace=True)

            # Record metrics
            removed_rows = initial_rows - len(self.df)
            self.metrics["removed_transactions"] = {
                "initial_count": initial_rows,
                "final_count": len(self.df),
                "removed_count": removed_rows,
                "removed_percentage": (removed_rows / initial_rows) * 100,
            }

            # Log the changes
            self.processing_text.append("\n1. SPECIAL TRANSACTIONS REMOVAL")
            self.processing_text.append("-" * 30)
            self.processing_text.append(f"Initial row count: {initial_rows:,}")
            self.processing_text.append(f"Final row count: {len(self.df):,}")
            self.processing_text.append(f"Removed rows: {removed_rows:,}")
            self.processing_text.append(f"Removed percentage: {(removed_rows / initial_rows) * 100:.2f}%\n")

            logger.save_logs("Special transactions removed successfully", log_level="info")
            return self.df

        except Exception as e:
            logger.save_logs(f"Error removing special transactions: {str(e)}", log_level="error")
            raise

    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        try:
            # Get initial missing value counts
            initial_missing = self.df.isnull().sum()
            initial_missing = initial_missing[initial_missing > 0]

            self.processing_text.append("2. MISSING VALUES HANDLING")
            self.processing_text.append("-" * 30)

            if not initial_missing.empty:
                # Log initial missing values
                missing_stats = []
                for col in initial_missing.index:
                    count = initial_missing[col]
                    percentage = (count / len(self.df)) * 100
                    missing_stats.append([col, count, f"{percentage:.2f}%"])

                self.processing_text.append("\nInitial Missing Values:")
                self.processing_text.append(
                    tabulate(missing_stats, headers=["Column", "Count", "Percentage"], tablefmt="grid")
                )

                # Handle missing values using mode within fraud groups
                for col in initial_missing.index:
                    # Create a copy of the column to avoid SettingWithCopyWarning
                    self.df = self.df.copy()

                    # Get mode for each fraud group
                    group_modes = {}
                    for fraud_val in self.df["isFraud"].unique():
                        group_data = self.df[self.df["isFraud"] == fraud_val][col]
                        if not group_data.empty and not group_data.mode().empty:
                            group_modes[fraud_val] = group_data.mode().iloc[0]

                    # Fill missing values based on group
                    for fraud_val, mode_val in group_modes.items():
                        mask = (self.df["isFraud"] == fraud_val) & (self.df[col].isnull())
                        if mode_val is not None:
                            self.df.loc[mask, col] = mode_val

                    # If any nulls remain, fill with overall mode
                    if self.df[col].isnull().any():
                        overall_mode = self.df[col].mode()
                        if not overall_mode.empty:
                            self.df[col] = self.df[col].fillna(overall_mode.iloc[0])
                        else:
                            # If no mode exists, use a default value based on column type
                            dtype = self.df[col].dtype
                            if pd.api.types.is_numeric_dtype(dtype):
                                self.df[col] = self.df[col].fillna(0)
                            elif pd.api.types.is_string_dtype(dtype):
                                self.df[col] = self.df[col].fillna("UNKNOWN")
                            else:
                                self.df[col] = self.df[col].fillna(self.df[col].iloc[0])

                # Verify no missing values remain
                final_missing = self.df.isnull().sum().sum()
                self.processing_text.append(f"\nRemaining missing values: {final_missing}")

                # Record metrics
                self.metrics["missing_values"] = {
                    "initial_missing": initial_missing.to_dict(),
                    "final_missing": final_missing,
                    "imputation_method": "mode_by_fraud_group_with_fallback",
                }
            else:
                self.processing_text.append("No missing values found in the dataset.")

            logger.save_logs("Missing values handled successfully", log_level="info")
            return self.df

        except Exception as e:
            logger.save_logs(f"Error handling missing values: {str(e)}", log_level="error")
            raise

    def derive_features(self):
        """Create derived features"""
        try:
            self.processing_text.append("\n3. FEATURE DERIVATION")
            self.processing_text.append("-" * 30)

            # Create CVV match feature
            self.df["CVV_matched"] = (self.df["cardCVV"] == self.df["enteredCVV"]).astype(int)
            self.processing_text.append("Created new feature: CVV_matched")

            # Drop unnecessary columns
            columns_to_drop = [
                "accountNumber",
                "customerId",
                "cardCVV",
                "enteredCVV",
                "cardLast4Digits",
                "transactionDateTime",
                "currentExpDate",
                "accountOpenDate",
                "dateOfLastAddressChange",
                "availableMoney",
            ]

            self.df = self.df.drop(columns_to_drop, axis=1)

            # Record metrics
            self.metrics["feature_engineering"] = {
                "derived_features": ["CVV_matched"],
                "dropped_columns": columns_to_drop,
                "final_column_count": len(self.df.columns),
            }

            # Log changes
            self.processing_text.append("\nDerived Features:")
            self.processing_text.append("- CVV_matched: Binary indicator of CVV match")
            self.processing_text.append(f"\nDropped {len(columns_to_drop)} columns:")
            self.processing_text.append("- " + "\n- ".join(columns_to_drop))

            logger.save_logs("Feature derivation completed successfully", log_level="info")
            return self.df

        except Exception as e:
            logger.save_logs(f"Error in feature derivation: {str(e)}", log_level="error")
            raise

    def save_processing_text(self):
        """Save processing text to file"""
        try:
            os.makedirs("reports", exist_ok=True)
            with open("reports/advanced_feature_analysis.txt", "w") as f:
                f.write("ADVANCED FEATURE PROCESSING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write("Timestamp: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("-" * 50 + "\n\n")
                f.write("\n".join(self.processing_text))

            logger.save_logs("Feature processing report saved successfully", log_level="info")
        except Exception as e:
            logger.save_logs(f"Error saving feature processing report: {str(e)}", log_level="error")
            raise


class FeatureEncoder:
    """Class to handle feature encoding operations"""

    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}
        self.encoding_text = []
        self.encoders = {}

    def encode_features(self):
        """Encode categorical features using different methods"""
        try:
            self.encoding_text.append("4. FEATURE ENCODING")
            self.encoding_text.append("-" * 30 + "\n")

            # Define encoding methods for different features
            encoding_methods = {
                "acqCountry": "mean",
                "merchantCountryCode": "mean",
                "transactionType": "mean",
                "posEntryMode": "mean",
                "posConditionCode": "mean",
                "cardPresent": "binary",
                "expirationDateKeyInMatch": "binary",
                "isFraud": "binary",
                "merchantName": "frequency",
                "merchantCategoryCode": "frequency",
            }

            # Log encoding strategy
            self.encoding_text.append("Encoding Strategy:")
            for feature, method in encoding_methods.items():
                self.encoding_text.append(f"- {feature}: {method} encoding")

            encoded_features = []
            encoding_stats = []

            # Apply encodings
            for col, encoding_method in encoding_methods.items():
                if encoding_method == "mean":
                    # Mean encoding based on fraud rate
                    mean_encoded_values = self.df.groupby(col)["isFraud"].mean()
                    self.df[col + "_encoded"] = self.df[col].map(mean_encoded_values)
                    self.encoders[col] = {"method": "mean", "mapping": mean_encoded_values.to_dict()}
                    encoding_stats.append([col, "mean", len(mean_encoded_values)])

                elif encoding_method == "binary":
                    # Binary encoding
                    binary_encoder = LabelEncoder()
                    self.df[col + "_encoded"] = binary_encoder.fit_transform(self.df[col])
                    self.encoders[col] = {"method": "binary", "encoder": binary_encoder}
                    encoding_stats.append([col, "binary", 2])

                elif encoding_method == "frequency":
                    # Frequency encoding
                    frequency_encoded_values = self.df[col].value_counts(normalize=True)
                    self.df[col + "_encoded"] = self.df[col].map(frequency_encoded_values)
                    self.encoders[col] = {"method": "frequency", "mapping": frequency_encoded_values.to_dict()}
                    encoding_stats.append([col, "frequency", len(frequency_encoded_values)])

                encoded_features.append(col + "_encoded")

            # Log encoding statistics
            self.encoding_text.append("\nEncoding Statistics:")
            self.encoding_text.append(
                tabulate(encoding_stats, headers=["Feature", "Method", "Categories"], tablefmt="grid")
            )

            # Drop original categorical columns
            self.df.drop(list(encoding_methods.keys()), axis=1, inplace=True)

            # Record metrics
            self.metrics["encoding"] = {
                "methods_used": encoding_methods,
                "encoded_features": encoded_features,
                "original_features_dropped": list(encoding_methods.keys()),
            }

            logger.save_logs("Feature encoding completed successfully", log_level="info")
            return self.df

        except Exception as e:
            logger.save_logs(f"Error in feature encoding: {str(e)}", log_level="error")
            raise


class FeatureScaler:
    """Class to handle feature scaling operations"""

    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}
        self.scaling_text = []
        self.scalers = {}

    def scale_features(self):
        """Scale numerical features"""
        try:
            self.scaling_text.append("5. FEATURE SCALING")
            self.scaling_text.append("-" * 30 + "\n")

            # Define columns to scale
            columns_to_normalize = ["creditLimit", "transactionAmount", "currentBalance"]

            # Initialize MinMaxScaler
            minmax_scaler = MinMaxScaler()
            scaling_stats = []

            # Scale each column
            for col in columns_to_normalize:
                # Get original statistics
                orig_stats = self.df[col].describe()

                # Scale the column
                col_data = self.df[col].values.reshape(-1, 1)
                self.df[col] = minmax_scaler.fit_transform(col_data)

                # Get scaled statistics
                scaled_stats = self.df[col].describe()

                # Store scaler
                self.scalers[col] = minmax_scaler

                # Record statistics
                scaling_stats.append([col, f"{orig_stats['min']:.2f} to {orig_stats['max']:.2f}", "0.0 to 1.0"])

            # Add datetime scaling
            datetime_columns = ["transactionDateTime", "currentExpDate", "accountOpenDate"]
            for col in datetime_columns:
                if col in self.df.columns:
                    col_data = self.df[col].values.reshape(-1, 1)
                    self.df[col] = self.minmax_scaler.fit_transform(col_data)

            # # Add monetary columns scaling
            # monetary_columns = ['creditLimit', 'transactionAmount', 'currentBalance']
            # for col in monetary_columns:
            #     if col in self.df.columns:
            #         col_data = self.df[col].values.reshape(-1, 1)
            #         self.df[col] = self.minmax_scaler.fit_transform(col_data)

            # Log scaling statistics
            self.scaling_text.append("Scaling Results:")
            self.scaling_text.append(
                tabulate(scaling_stats, headers=["Feature", "Original Range", "Scaled Range"], tablefmt="grid")
            )

            # Record metrics
            self.metrics["scaling"] = {
                "scaled_features": columns_to_normalize,
                "scaling_method": "minmax",
                "scale_range": [0, 1],
            }

            logger.save_logs("Feature scaling completed successfully", log_level="info")
            return self.df

        except Exception as e:
            logger.save_logs(f"Error in feature scaling: {str(e)}", log_level="error")
            raise

    def save_scaling_text(self):
        """Save scaling information to file"""
        try:
            os.makedirs("reports", exist_ok=True)
            with open("reports/feature_scaling.txt", "w") as f:
                f.write("FEATURE SCALING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write("Timestamp: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("-" * 50 + "\n\n")
                f.write("\n".join(self.scaling_text))

            logger.save_logs("Feature scaling report saved successfully", log_level="info")
        except Exception as e:
            logger.save_logs(f"Error saving feature scaling report: {str(e)}", log_level="error")
            raise


class DataVisualizer:
    """Class to handle data visualization"""

    def __init__(self, dataframe):
        self.df = dataframe

    def plot_transaction_distributions(self, reversed_df, multi_swipe_df):
        try:
            plt.figure(figsize=(10, 4))
            sns.histplot(data=reversed_df, x="transactionAmount", kde=True, label="Reversed Transactions")
            sns.histplot(data=multi_swipe_df, x="transactionAmount", kde=True, label="Multi Swipe")
            plt.xlabel("Transaction Amount")
            plt.ylabel("Frequency")
            plt.title("Distribution of Transaction Amounts")
            plt.legend()

            # Save plot
            plt.savefig("reports/figures/transaction_distributions.png")
            plt.close()

        except Exception as e:
            logger.save_logs(f"Error in plotting distributions: {str(e)}", log_level="error")
            raise


def make_json_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return make_json_serializable(obj.to_dict())
    elif pd.isna(obj):
        return None
    return obj


def save_metrics(processor_metrics, analyzer_metrics, output_path):
    """Save metrics to JSON file"""
    try:
        metrics = {
            "processing_metrics": make_json_serializable(processor_metrics),
            "analysis_metrics": make_json_serializable(analyzer_metrics),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.save_logs(f"Metrics saved to {output_path}", log_level="info")
    except Exception as e:
        logger.save_logs(f"Error saving metrics: {str(e)}", log_level="error")
        raise


def main():
    """Main function to orchestrate the advanced data processing pipeline"""
    try:
        # Initialize logging
        logger.save_logs("Starting advanced data processing pipeline", log_level="info")

        # Load the interim data
        logger.save_logs("Loading interim data", log_level="info")
        df = pd.read_hdf("data/inprogress/interim_transactions.h5")

        # Initialize transaction analyzer
        logger.save_logs("Starting transaction analysis", log_level="info")
        transaction_analyzer = TransactionAnalyzer(df)

        # Analyze reversed transactions
        reversed_df = transaction_analyzer.identify_reversed_transactions()

        # Analyze multi-swipe transactions
        multi_swipe_df = transaction_analyzer.identify_multi_swipe(time_window=5)

        # Add visualization - ADD HERE
        logger.save_logs("Generating transaction visualizations", log_level="info")
        visualizer = DataVisualizer(df)
        visualizer.plot_transaction_distributions(reversed_df, multi_swipe_df)

        # Initialize feature processor
        logger.save_logs("Starting feature processing", log_level="info")
        feature_processor = FeatureProcessor(df)

        # Remove special transactions and handle missing values
        processed_df = feature_processor.remove_special_transactions(reversed_df, multi_swipe_df)
        processed_df = feature_processor.handle_missing_values()

        # Derive new features
        processed_df = feature_processor.derive_features()
        feature_processor.save_processing_text()

        # Initialize feature encoder
        logger.save_logs("Starting feature encoding", log_level="info")
        feature_encoder = FeatureEncoder(processed_df)

        # Encode features
        encoded_df = feature_encoder.encode_features()

        # Initialize feature scaler
        logger.save_logs("Starting feature scaling", log_level="info")
        feature_scaler = FeatureScaler(encoded_df)

        # Scale features
        final_df = feature_scaler.scale_features()
        feature_scaler.save_scaling_text()

        # Create processed directory if it doesn't exist
        os.makedirs("data/processed", exist_ok=True)

        # Save final processed data in both CSV and HDF formats
        processed_csv_path = "data/processed/processed_transactions.csv"
        processed_hdf_path = "data/processed/processed_transactions.h5"

        # Save as CSV
        final_df.to_csv(processed_csv_path, index=False)
        logger.save_logs(f"Saved processed data to CSV: {processed_csv_path}", log_level="info")

        # Save as HDF
        final_df.to_hdf(processed_hdf_path, key="data", mode="w")
        logger.save_logs(f"Saved processed data to HDF: {processed_hdf_path}", log_level="info")

        # Save metrics
        metrics_path = "reports/advanced_wrangling_metrics.json"
        combined_metrics = {
            "transaction_analysis": transaction_analyzer.metrics,
            "feature_processing": feature_processor.metrics,
            "feature_encoding": feature_encoder.metrics,
            "feature_scaling": feature_scaler.metrics,
        }

        save_metrics(
            processor_metrics=combined_metrics,
            analyzer_metrics={
                "final_shape": final_df.shape,
                "final_columns": list(final_df.columns),
                "processing_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            output_path=metrics_path,
        )

        # Generate summary statistics
        # Generate summary statistics
        summary_stats = {
            "initial_rows": int(len(df)),
            "final_rows": int(len(final_df)),
            "initial_columns": int(len(df.columns)),
            "final_columns": int(len(final_df.columns)),
            "memory_usage": float(final_df.memory_usage(deep=True).sum() / 1024**2),  # in MB
            "processing_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_types_summary": {str(k): int(v) for k, v in final_df.dtypes.value_counts().to_dict().items()},
            "removed_transactions": {"reversed": int(len(reversed_df)), "multi_swipe": int(len(multi_swipe_df))},
        }

        # Save summary statistics
        summary_path = "reports/advanced_wrangling_summary.json"
        with open(summary_path, "w") as f:
            json.dump(make_json_serializable(summary_stats), f, indent=4)
        logger.save_logs(f"Saved processing summary to {summary_path}", log_level="info")

        # Print processing summary
        print("\nAdvanced Processing Summary:")
        print("-" * 50)
        print(f"Initial Rows: {summary_stats['initial_rows']:,}")
        print(f"Final Rows: {summary_stats['final_rows']:,}")
        print(f"Initial Columns: {summary_stats['initial_columns']}")
        print(f"Final Columns: {summary_stats['final_columns']}")
        print(f"Memory Usage: {summary_stats['memory_usage']:.2f} MB")
        print(f"Removed Reversed Transactions: {summary_stats['removed_transactions']['reversed']:,}")
        print(f"Removed Multi-Swipe Transactions: {summary_stats['removed_transactions']['multi_swipe']:,}")
        print(f"Processing Timestamp: {summary_stats['processing_timestamp']}")
        print("-" * 50)

        logger.save_logs("Advanced processing pipeline completed successfully", log_level="info")
        return final_df

    except Exception as e:
        logger.save_logs(f"Error in advanced processing pipeline: {str(e)}", log_level="error")
        raise
    finally:
        logger.save_logs("Advanced data processing pipeline finished", log_level="info")


if __name__ == "__main__":
    main()
