import os

# Paths
DATA_PATH = './datasets/cancer_gene_expression.csv'
MODEL_SAVE_PATH = './models/rf_model.pkl'

# Random state for reproducibility
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.2

# Feature selection
N_FEATURES = 300

# Model params
RF_MAX_FEATURES = 0.2
