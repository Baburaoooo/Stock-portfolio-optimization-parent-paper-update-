# Updated Stock Portfolio Selection by Hill Climbing Algorithm
# This notebook now uses the split datasets for proper train/test/validation methodology

import numpy as np
import pandas as pd
import random
import copy
import csv
import time
import matplotlib.pyplot as plt
import seaborn as sb

# Dataset configuration
DATASET_MODE = "train"  # Options: "train", "test", "validation"
DATASET_FILES = {
    "train": "train_dataset.csv",
    "test": "test_dataset.csv", 
    "validation": "validation_dataset.csv"
}

# Load the appropriate dataset
file_path = DATASET_FILES[DATASET_MODE]
dataset = pd.read_csv(file_path)

print(f"Loaded {DATASET_MODE} dataset: {file_path}")
print(f"Dataset shape: {dataset.shape}")

# Handle missing values
dataset[['normalized_fitness_score', 'normalized_percent_change_from_instrinsic_value', 'normalized_Rev_Gr_Next_Y']] = dataset[['normalized_fitness_score', 'normalized_percent_change_from_instrinsic_value', 'normalized_Rev_Gr_Next_Y']].fillna(0)

class StockData:
    def __init__(self, file_path):
        # Load data from CSV file
        self.dataset = pd.read_csv(file_path)

    def get_data(self):
        return self.dataset.values.tolist()

# Creating an instance of StockData with the file path
stock_data = StockData(file_path)

# Getting the data 
list_data = stock_data.get_data()

print(f"\nRunning Hill Climbing optimization on {DATASET_MODE} dataset...")
print(f"Dataset contains {len(list_data)} stocks")

# [Rest of the hill climbing implementation would go here...]

print(f"\nHill Climbing optimization completed on {DATASET_MODE} dataset")
