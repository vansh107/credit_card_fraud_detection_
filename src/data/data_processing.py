# data_procrssing.py
import json
import logging
import os

# Add this at the beginning of the file with other imports
import warnings

import numpy as np
import pandas as pd
from src.logger import CustomLogger, create_log_path
from tabulate import tabulate

warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)

# path to save the log files
log_file_path = create_log_path("process_dataset")
# create the custom logger object
logger = CustomLogger(logger_name="process_dataset", log_filename=log_file_path)
# set the level of logging to INFO
logger.set_log_level(level=logging.INFO)


class DataLoader:
    """Class to handle data loading operations"""

    def __init__(self, file_path, chunk_size=10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.df_raw = None

    def load_data(self):
        """Load data from JSON file using chunks"""
        try:
            chunks = []
            for chunk in pd.read_json(self.file_path, lines=True, chunksize=self.chunk_size):
                chunks.append(chunk)
            self.df_raw = pd.concat(chunks, ignore_index=True)
            logger.save_logs(f"Successfully loaded data from {self.file_path}", log_level="info")
            return self.df_raw
        except Exception as e:
            logger.save_logs(f"Error loading data: {str(e)}", log_level="error")
            raise


class DataProcessor:
    """Class to handle data processing operations"""

    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}

    def process_data(self):
        """Process the loaded data"""
        try:
            # Add your data processing logic here

            # Record metrics
            self.metrics["total_rows"] = len(self.df)
            self.metrics["columns"] = list(self.df.columns)

            logger.save_logs("Data processing completed successfully", log_level="info")
            return self.df
        except Exception as e:
            logger.save_logs(f"Error processing data: {str(e)}", log_level="error")
            raise

    # def save_processed_data(self, output_path):
    #     """Save processed data to CSV"""
    #     try:
    #         os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #         self.df.to_csv(output_path, index=False)
    #         logger.save_logs(f"Processed data saved to {output_path}", log_level='info')
    #     except Exception as e:
    #         logger.save_logs(f"Error saving processed data: {str(e)}", log_level='error')
    #         raise


class DataAnalyzer:
    """Class to handle data analysis operations"""

    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}
        self.exploration_text = []

    def data_brief(self, th_value_count=10, nrow=10):
        """Generate detailed data summary"""
        try:
            categorical_columns = self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            numerical_columns = self.df.select_dtypes(include=["int", "float"]).columns.tolist()
            datetime_columns = self.df.select_dtypes(include=["datetime"]).columns.tolist()

            summary_data = []
            summary_data.append(["Dimensions (rows x columns)", f"{self.df.shape[0]} x {self.df.shape[1]}"])

            # All columns
            all_columns = [", ".join(self.df.columns[i : i + nrow]) for i in range(0, len(self.df.columns), nrow)]
            summary_data.append(["All Columns", "\n".join(all_columns)])

            # Datetime columns
            datetime_summary = [f"Total Datetime Features = {len(datetime_columns)}\n"] + [
                ", ".join(datetime_columns[i : i + nrow]) for i in range(0, len(datetime_columns), nrow)
            ]
            summary_data.append(["Datetime Columns", "\n".join(datetime_summary)])

            # Categorical columns
            cat_summary = [f"'{col}': {self.df[col].nunique()} unique values" for col in categorical_columns]
            cat_summary.insert(0, f"Total Categorical Columns = {len(categorical_columns)}\n")
            summary_data.append(["Categorical Columns", "\n".join(cat_summary)])

            # Numerical columns
            num_summary = [f"Total Numerical Features = {len(numerical_columns)}\n"] + [
                ", ".join(numerical_columns[i : i + nrow]) for i in range(0, len(numerical_columns), nrow)
            ]
            summary_data.append(["Numerical Columns", "\n".join(num_summary)])

            # Generate tabulated summary
            basic_summary = tabulate(summary_data, headers=["Feature Type", "Basic Brief"], tablefmt="grid")
            self.exploration_text.append("Basic Data Summary:\n" + basic_summary + "\n")

            # Value counts for categorical columns
            cat_value_counts = []
            for col in categorical_columns:
                if self.df[col].nunique() <= th_value_count:
                    value_counts = dict(sorted(self.df[col].value_counts().items(), key=lambda x: x[1], reverse=True))
                    cat_value_counts.append([col, str(value_counts)])

            if cat_value_counts:
                cat_table = tabulate(cat_value_counts, headers=["Column", "Value Counts"], tablefmt="grid")
                self.exploration_text.append(f"\nCategorical Columns (≤{th_value_count} unique values):\n" + cat_table)

            return summary_data
        except Exception as e:
            logger.save_logs(f"Error in data brief: {str(e)}", log_level="error")
            raise

    def analyze_missing_data(self):
        """Analyze missing data in the dataset"""
        try:
            missing_counts = self.df.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0]
            if not missing_counts.empty:
                missing_percentages = (missing_counts / len(self.df)) * 100
                missing_data = [
                    [col, count, f"{pct:.2f}%"]
                    for col, count, pct in zip(missing_counts.index, missing_counts.values, missing_percentages.values)
                ]

                missing_table = tabulate(
                    missing_data, headers=["Column", "Missing Count", "Missing Percentage"], tablefmt="grid"
                )
                self.exploration_text.append("\nMissing Data Analysis:\n" + missing_table)

            return missing_counts.index.tolist()
        except Exception as e:
            logger.save_logs(f"Error in missing data analysis: {str(e)}", log_level="error")
            raise

    def preprocess_n_analyze_data(self):
        """Analyze the processed data"""
        try:
            pd.set_option("future.no_silent_downcasting", True)
            processing_steps = []  # List to store processing steps

            # Step 1: Basic data processing
            processing_steps.append("1. Basic Data Cleaning:")
            processing_steps.append("   - Replaced blank cells and only empty spaces with NaN values")
            self.df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

            # Step 2: DateTime Conversion
            processing_steps.append("\n2. DateTime Conversions:")
            datetime_columns = ["transactionDateTime", "currentExpDate", "accountOpenDate", "dateOfLastAddressChange"]

            for col in datetime_columns:
                if col in self.df.columns:
                    if col == "currentExpDate":
                        self.df[col] = pd.to_datetime(self.df[col], format="%m/%Y")
                        processing_steps.append(f"   - Converted {col} to datetime using format '%m/%Y'")
                    else:
                        self.df[col] = pd.to_datetime(self.df[col])
                        processing_steps.append(f"   - Converted {col} to datetime using standard format")

            # Step 3: Data Type Summary
            processing_steps.append("\n3. Final Data Types:")
            for col, dtype in self.df.dtypes.items():
                processing_steps.append(f"   - {col}: {dtype}")

            # Step 4: Generate data brief
            processing_steps.append("\n4. Data Brief Generation:")
            processing_steps.append("   - Generated comprehensive data summary")
            processing_steps.append("   - Analyzed column types and unique values")
            self.data_brief()

            # Step 5: Missing Data Analysis
            processing_steps.append("\n5. Missing Data Analysis:")
            missing_columns = self.analyze_missing_data()
            if missing_columns:
                processing_steps.append("   Columns with missing values:")
                for col in missing_columns:
                    missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                    processing_steps.append(f"   - {col}: {missing_pct:.2f}% missing")
            else:
                processing_steps.append("   - No missing values found")

            # Step 6: Numerical Summary
            processing_steps.append("\n6. Statistical Summary Generation:")
            processing_steps.append("   - Generated descriptive statistics for numerical columns")
            numerical_summary = self.df.describe()
            num_table = tabulate(numerical_summary, headers="keys", tablefmt="grid")
            self.exploration_text.append("\nNumerical Summary:\n" + num_table)

            # Save processing steps to file
            os.makedirs("reports", exist_ok=True)
            with open("reports/preprocess_n_analysis.txt", "w") as f:
                f.write("DATA PREPROCESSING AND ANALYSIS STEPS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Timestamp: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("-" * 50 + "\n\n")
                f.write("\n".join(processing_steps))
                f.write("\n\nProcessing Statistics:\n")
                f.write(f"Total Rows Processed: {len(self.df):,}\n")
                f.write(f"Total Columns Processed: {len(self.df.columns)}\n")
                f.write(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

            # Save exploration text
            self.save_exploration_text()

            # Record metrics
            self.metrics.update(
                {
                    "null_counts": self.df.isnull().sum().to_dict(),
                    "data_types": self.df.dtypes.astype(str).to_dict(),
                    "missing_columns": missing_columns,
                    "shape": self.df.shape,
                    "processing_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            logger.save_logs("Data analysis completed successfully", log_level="info")
            return self.df

        except Exception as e:
            logger.save_logs(f"Error analyzing data: {str(e)}", log_level="error")
            raise

    def save_exploration_text(self):
        """Save exploration text to file"""
        try:
            os.makedirs("reports", exist_ok=True)
            with open("reports/data_exploration.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(self.exploration_text))
            logger.save_logs("Data exploration text saved successfully", log_level="info")
        except Exception as e:
            logger.save_logs(f"Error saving exploration text: {str(e)}", log_level="error")
            raise


def save_metrics(processor_metrics, analyzer_metrics, output_path):
    """Save metrics to JSON file"""
    try:
        metrics = {"processing_metrics": processor_metrics, "analysis_metrics": analyzer_metrics}

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.save_logs(f"Metrics saved to {output_path}", log_level="info")
    except Exception as e:
        logger.save_logs(f"Error saving metrics: {str(e)}", log_level="error")
        raise


def main():
    """Main function to orchestrate the data processing pipeline"""
    try:
        # Initialize data loader
        logger.save_logs("Starting data processing pipeline", log_level="info")
        data_loader = DataLoader("data/raw/extracted/transactions.txt")

        # Load the data
        logger.save_logs("Loading raw data", log_level="info")
        df_raw = data_loader.load_data()

        # Process the data
        logger.save_logs("Starting data processing", log_level="info")
        processor = DataProcessor(df_raw)
        processed_df = processor.process_data()  # add to data_processing_metrics.json

        # Analyze the data
        logger.save_logs("Starting data analysis", log_level="info")
        analyzer = DataAnalyzer(processed_df)
        analyzed_df = analyzer.preprocess_n_analyze_data()

        # Create processed directory if it doesn't exist
        os.makedirs("data/inprogress", exist_ok=True)

        # Save final processed data in both CSV and HDF formats
        interim_csv_path = "data/inprogress/interim_transactions.csv"
        interim_hdf_path = "data/inprogress/interim_transactions.h5"

        # Save as CSV
        analyzed_df.to_csv(interim_csv_path, index=False)
        logger.save_logs(f"Saved basic processed data to CSV: {interim_csv_path}", log_level="info")

        # Save as HDF
        analyzed_df.to_hdf(interim_hdf_path, key="data", mode="w")
        logger.save_logs(f"Saved basic processed data to HDF: {interim_hdf_path}", log_level="info")

        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)

        # Save metrics
        metrics_path = "reports/data_processing_metrics.json"
        save_metrics(processor_metrics=processor.metrics, analyzer_metrics=analyzer.metrics, output_path=metrics_path)
        logger.save_logs(f"Saved metrics to {metrics_path}", log_level="info")

        # Generate summary statistics
        summary_stats = {
            "total_rows": len(analyzed_df),
            "total_columns": len(analyzed_df.columns),
            "memory_usage": float(analyzed_df.memory_usage(deep=True).sum() / 1024**2),  # Convert to float
            "processing_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_types_summary": {
                str(k): int(v) for k, v in analyzed_df.dtypes.value_counts().to_dict().items()
            },  # Convert keys and values
            "missing_values_summary": int(analyzed_df.isnull().sum().sum()),  # Convert to int
        }

        # Save summary statistics
        summary_path = "reports/processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=4)
        logger.save_logs(f"Saved processing summary to {summary_path}", log_level="info")

        # Print processing summary
        print("\nProcessing Summary:")
        print("-" * 50)
        print(f"Total Rows Processed: {summary_stats['total_rows']:,}")
        print(f"Total Columns: {summary_stats['total_columns']}")
        print(f"Memory Usage: {summary_stats['memory_usage']:.2f} MB")
        print(f"Missing Values: {summary_stats['missing_values_summary']:,}")
        print(f"Processing Timestamp: {summary_stats['processing_timestamp']}")
        print("-" * 50)

        logger.save_logs("Main processing pipeline completed successfully", log_level="info")
        return analyzed_df

    except Exception as e:
        logger.save_logs(f"Error in main processing pipeline: {str(e)}", log_level="error")
        raise
    finally:
        logger.save_logs("Basic Data processing pipeline finished", log_level="info")


if __name__ == "__main__":
    main()
