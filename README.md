MLOPS - Credit Card Fraud Detection
==============================

## Data Source
Data used in this project is sourced from the [Capital One Data Science Challenge GitHub Repository](https://github.com/CapitalOneRecruiting/DS).

<img src="https://github.com/user-attachments/assets/0b8d2663-ef94-42b7-9c9a-1f3ad6eb0bfd" alt="Challenge Image" width="300"/>

This Repo is dedicated to end-to-end Machine Learning Project with MLOps

## Docker commands to execute complete pipeline with DVC

```bash
# 1. Build the Docker Image
docker-compose build
# 2. Run the Container
docker-compose up
```
```bash
# 3. Stop and Remove Containers
docker-compose down
```

## DVC Pipeline Execution command without using docker:

```bash
python3 -m venv .mlops_venv  # Create a new virtual environment in the .mlops_venv directory
source .mlops_venv/bin/activate  # Activate the virtual environment

pip install -e .  # Install the current package in editable mode

dvc init  # Initialize a new DVC repository

# Download the dataset and set the source
dvc get https://github.com/CapitalOneRecruiting/DS transactions.zip -o data/raw/zipped/
dvc add data/raw/zipped/transactions.zip

dvc dag  # Display the DVC pipeline as a directed acyclic graph (DAG)

# To execute a machine learning pipeline defined in DVC, you can use the following command
# This will execute Data Preprocessing, Feature Engineering, Model Training, and Evaluation stages
# as defined in the dvc.yaml file, in the correct order and only if there are changes in data or any other code changes
dvc repro

# Add Google Drive as a remote storage for DVC
# Replace 'myremote' with your preferred remote name
# Replace 'folder_id' with the actual ID of your Google Drive folder
dvc remote add -d myremote gdrive://folder_id/path/to/dvc/storage

python3 src/gdrive_setup/setup_dvc_remote.py  # Run a script to set up the DVC remote configuration with gdrve client secret keys

dvc push  # Push dvc data changes to the Google drive or any other remote source like AWS (as set in the files)
```

<img width="389" alt="image" src="https://github.com/user-attachments/assets/36c2dbdd-268d-4470-b003-a0ec20daafed" />


Project Organization
------------

    mlops/                           # Root project directory
    ├── config/                      # Configuration files
    │   ├── gdrive.json              # Google Drive credentials
    │   └── gdrive_setup.md          # GDrive setup instructions
    │
    ├── data/                         # Data storage directory
    │   ├── external/                 # Third-party data
    │   ├── inprogress/               # Intermediate processing results
    │   │   ├── interim_transactions.csv
    │   │   ├── interim_transactions.h5
    │   │   └── readme.md
    │   ├── processed/                 # Final processed datasets
    │   │   ├── processed_transactions.csv
    │   │   └── processed_transactions.h5
    │   └── raw/                       # Original data files
    │       ├── extracted/             # Extracted data
    │       │   └── transactions.txt
    │       └── zipped/                # Zipped data
    │           ├── transactions.zip
    │           └── transactions.zip.dvc
    │
    ├── docs/                          # Project documentation
    │
    ├── logs/                          # Pipeline execution logs
    │   ├── advanced_wrangling/
    │   ├── extract_dataset/
    │   └── process_dataset/
    │
    ├── models/                         # Trained model artifacts
    │
    ├── notebooks/                      # Jupyter analysis notebooks
    │   ├── 1_load_data_exploration.ipynb
    │   ├── 2_data_visualization.ipynb
    │   ├── 3_data_wrangling_modeling.ipynb
    │   ├── 4-model-testing.ipynb
    │   └── 5-model-deployment.ipynb
    │
    ├── references/                      # External references docs
    │
    ├── reports/                         # Pipeline Analysis & Training outputs
    │   ├── figures/
    │   │   └── transaction_distributions.png
    │   ├── advanced_data_exploration.txt
    │   ├── advanced_feature_analysis.txt
    │   ├── advanced_wrangling_metrics.json
    │   ├── advanced_wrangling_summary.json
    │   ├── data_processing_metrics.json
    │   ├── feature_scaling.txt
    │   └── other metrics and summary files
    │
    ├── src/
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── data_collection.py        # Data extraction and collection
    │   │   ├── data_processing.py        # Basic data preprocessing
    │   │   └── make_dataset.py
    │   │
    │   ├── data-wrangling-advance/
    │   │   ├── __init__.py
    │   │   └── adv_data_processing.py     # Advanced data processing
    │   │
    │   ├── gdrive_setup/
    │   │   └── setup_dvc_remote.py        # Google Drive setup for DVC
    │   │
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── predict_model.py           # Model prediction scripts
    │   │   └── train_model.py             # Model training scripts
    │   │
    │   ├── undersampling-experiments/
    │   │   ├── __init__.py
    │   │   └── build_features.py
    │   │
    │   └── visualization/
    │       ├── __init__.py
    │       └── visualize.py               # Visualization utilities
    │
    ├── Dockerfile                         # Docker configuration
    ├── docker-compose.yml                 # Docker resource mapping
    ├── dvc.yaml                           # DVC pipeline definition
    ├── dvc.lock                           # DVC pipeline state
    ├── requirements.txt                   # Production dependencies
    ├── dev-requirements.txt               # Development dependencies
    ├── contraints.txt                     # Version constraints
    ├── tmp_requirements.txt
    ├── setup.py                           # Package setup
    ├── setup.md                           # Setup guide
    ├── test_environment.py
    └── tox.ini                            # Testing config


--------

# Notebook Description
--------
- `1_load_data_exploration.ipynb`: Jupyter Notebook for loading and understanding the dataset.
- `2_data_visualization.ipynb`: Jupyter Notebook for data visualization and plotting.
- `3_data_wrangling_modeling.ipynb`: Jupyter Notebook for data wrangling, EDA, data preparation, and building machine learning models.
- `4-model-testing.ipynb`: Jupyter Notebook for model testing.
- `5-model-deployment.ipynb`: Jupyter Notebook for data visualization and plotting (in progress).

- **Notebook 1: 1_load_data_exploration.ipynb**: 
In this initial notebook, I focused on establishing a strong foundation for the project. I meticulously loaded the dataset from github file and extracted the zip file in data folder, ensuring its integrity and consistency. This step was crucial to ensure that subsequent analyses and modeling were built upon reliable data. Afterwards, I worked on basic data exploration of Categorical, Numerical, and Datetime attributes and data structure.

- **Notebook 2: 2_data_visualization.ipynb**: 
With a solid foundation in place, I delved into the world of data visualization. This notebook was dedicated to unraveling the hidden patterns within the dataset. Through an array of plots, charts, and visualizations, I deciphered the distribution of features, uncovered potential correlations, and gained crucial insights into the underlying trends. These visual revelations served as guiding lights for subsequent decision-making.

- **Notebook 3: 3_data_wrangling_modeling.ipynb**: 
In the final phase of my exploration, I undertook comprehensive data wrangling and modeling endeavors. This notebook encapsulated the essence of my project, combining the insights from previous notebooks into actionable steps. Here, I embarked on an intricate journey: 
    - Duplicate Transaction Identification: I delved into the identification and analysis of multi-swipe and reversed duplicate transactions. This endeavor provided a deeper understanding of these transactions' impact on the overall dataset. 
    - Feature Engineering, Cleaning, and Normalization: With an eye for improvement, I engaged in feature engineering to harness the latent potential of the dataset. Additionally, I handled missing values and employed normalization techniques to ensure data consistency and reliability.
    - Effective Imbalanced Data Handling: Recognizing the importance of tackling data imbalance, I implemented an undersampling strategy with n iterations. This method effectively addressed the challenge while retaining the integrity of the dataset.
    - Advanced Modeling with Rigorous Evaluation: Armed with well-preprocessed data, I ventured into modeling armed with cross-validation and hyperparameter tuning. Rigorous evaluation using key metrics helped ascertain the model's performance and suitability for the fraud detection task.


--------
## Future Work:<br>

#### Data Preprocessing
- Implement MICE (Multiple Imputation by Chained Equations) for missing value imputation
- Apply various data transformation techniques on right-skewed attributes
- Utilize PCA (Principal Component Analysis) for dimensionality reduction

#### Statistical Analysis
- Conduct statistical tests such as hypothesis testing, t-tests, and F-statistics among features

#### Advanced Techniques
- **Clustering for Data Segmentation**: Apply algorithms like K-Means or DBSCAN to segment data into meaningful clusters, using cluster labels as additional features
- **Fraud Trend Analysis**: Identify temporal and transaction-related patterns specific to fraudulent activities
- **Iterative Undersampling**: Perform undersampling for each cluster to balance class distribution while maintaining dataset diversity

#### Model Development
- **Model Selection and Tuning**: Explore various classification models (e.g., Random Forest, Gradient Boosting, XGBoost, Support Vector Machines) with hyperparameter tuning for each cluster
- **Ensemble Strategies**: Implement techniques like stacking to combine predictions from different models, weighting them based on performance and cluster association throughout model training

#### Evaluation and Monitoring
- Regularly evaluate models on validation and holdout sets
- Implement monitoring mechanisms to detect model degradation or concept drift

#### Feature Engineering
- Create time-based features, transaction frequency metrics, and transaction value ratios

#### Continuous Improvement
- Update and refine the model with new data
- Stay informed about new techniques and research in fraud detection
