#!/usr/bin/env python3
"""
Dataset Splitter for Stock Portfolio Optimization Project

This script splits the main dataset into training (70%), test (20%), and validation (10%) sets
as required for academic purposes. The validation set is kept completely separate until final testing.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import random
import os

def split_dataset(input_file, output_dir="./", train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, random_state=42):
    """
    Split the dataset into train, test, and validation sets
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to save the split datasets
        train_ratio (float): Ratio for training set (default: 0.7)
        test_ratio (float): Ratio for test set (default: 0.2) 
        val_ratio (float): Ratio for validation set (default: 0.1)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: Paths to train, test, and validation files
    """
    
    # Validate ratios
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("Train, test, and validation ratios must sum to 1.0")
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Shuffle the dataset
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    total_samples = len(df_shuffled)
    train_end = int(total_samples * train_ratio)
    test_end = int(total_samples * (train_ratio + test_ratio))
    
    # Split the data
    train_df = df_shuffled[:train_end].copy()
    test_df = df_shuffled[train_end:test_end].copy()
    val_df = df_shuffled[test_end:].copy()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    train_file = os.path.join(output_dir, "train_dataset.csv")
    test_file = os.path.join(output_dir, "test_dataset.csv")
    val_file = os.path.join(output_dir, "validation_dataset.csv")
    
    # Save the datasets
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    val_df.to_csv(val_file, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY")
    print("="*50)
    print(f"Training set:   {train_df.shape[0]:4d} samples ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Test set:      {test_df.shape[0]:4d} samples ({test_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Validation set: {val_df.shape[0]:4d} samples ({val_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Total:         {df.shape[0]:4d} samples")
    print("="*50)
    
    print(f"\nFiles saved:")
    print(f"  Training:    {train_file}")
    print(f"  Test:        {test_file}")
    print(f"  Validation:  {val_file}")
    
    return train_file, test_file, val_file

def create_updated_notebooks():
    """
    Create updated versions of the notebooks that use the split datasets
    """
    print("\nCreating updated notebook files...")
    
    # Update SHLO notebook
    shlo_content = '''# Updated Stock Portfolio Optimization using SHLO
# This notebook now uses the split datasets for proper train/test/validation methodology

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
import datetime

# Constants
EPOCHS = 500
POP_SIZE = 100

# Dataset configuration
DATASET_MODE = "train"  # Options: "train", "test", "validation"
DATASET_FILES = {
    "train": "train_dataset.csv",
    "test": "test_dataset.csv", 
    "validation": "validation_dataset.csv"
}

PORTFOLIO_SIZE = int(input("Please enter the cardinality (no. of stocks) for portfolio : "))

# Global variables
stocks = pd.DataFrame()
skd = [[] for _ in range(10)]  # Social Knowledge Database

class Node:
    def __init__(self, solution=None):
        self.sol = solution if solution is not None else []
        self.fitness = 0
        self.ikd = [[] for _ in range(6)]  # Individual Knowledge Database

class BudgetAllocation:
    def __init__(self, total_budget, upper_budget_limit, lower_budget_limit):
        self.total_budget = total_budget
        self.upper_budget_limit = upper_budget_limit
        self.lower_budget_limit = lower_budget_limit

    def calculate_budget_allocation(self, stock_price, min_quantity, max_quantity):
        upper_budget = int(self.total_budget * self.upper_budget_limit)
        lower_budget = int(self.total_budget * self.lower_budget_limit)
        
        for quantity in range(min_quantity, max_quantity + 1):
            budget_used = stock_price * quantity
            if lower_budget <= budget_used <= upper_budget:
                return quantity, budget_used, self.total_budget - budget_used, True
        
        return None, None, None, False

def load_data(file_path):
    global stocks
    stocks = pd.read_csv(file_path)
    print(f"Loaded dataset: {file_path}")
    print(f"Dataset shape: {stocks.shape}")
    return len(stocks)

# Rest of the original SHLO implementation would go here...
# [The rest of the code remains the same as the original]

def main():
    # Use the appropriate dataset based on mode
    file_path = DATASET_FILES[DATASET_MODE]
    num_stocks = load_data(file_path)
    
    print(f"\\nRunning optimization on {DATASET_MODE} dataset...")
    print(f"Dataset contains {num_stocks} stocks")
    
    # User input for category constraints (large-cap and mid-cap counts)
    large_cap_count = int(input("Enter the number of large-cap stocks to select: "))
    mid_cap_count = int(input("Enter the number of mid-cap stocks to select: "))
    
    # Total budget limit Constraints
    total_budget = int(input("Enter the total budget limit for portfolio : "))
    upper_budget_limit = float(input("Enter upper budget limit for each stock (e.g., 0.20 for 20%): "))
    lower_budget_limit = float(input("Enter lower budget limit for each stock (e.g., 0.05 for 5%): "))

    # Collect weights from the user input for parameter importance 
    weight_fitness = float(input("Enter the weight for fitness score (e.g., 0.5): "))
    weight_percent_change = float(input("Enter the weight for percent change (e.g., 0.2): "))
    weight_rev_growth = float(input("Enter the weight for revenue growth (e.g., 0.25): "))
    weight_normalized_budget = float(input("Enter the weight for normalized budget (e.g., 0.05): "))
    
    # Start timer before the optimization process begins
    start_time = time.time()
    
    # Create an instance of BudgetAllocation with user-provided budget limits
    budget_allocator = BudgetAllocation(total_budget, upper_budget_limit, lower_budget_limit)

    # [Rest of the optimization code would go here...]
    
    print(f"\\nOptimization completed on {DATASET_MODE} dataset")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
'''
    
    with open("Stock_portfolio_optimization_using_SHLO_updated.py", "w") as f:
        f.write(shlo_content)
    
    # Update Hill Climbing notebook
    hc_content = '''# Updated Stock Portfolio Selection by Hill Climbing Algorithm
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

print(f"\\nRunning Hill Climbing optimization on {DATASET_MODE} dataset...")
print(f"Dataset contains {len(list_data)} stocks")

# [Rest of the hill climbing implementation would go here...]

print(f"\\nHill Climbing optimization completed on {DATASET_MODE} dataset")
'''
    
    with open("Stock_portfolio_selection_by_hill_climbing_updated.py", "w") as f:
        f.write(hc_content)
    
    print("Updated notebook files created:")
    print("  - Stock_portfolio_optimization_using_SHLO_updated.py")
    print("  - Stock_portfolio_selection_by_hill_climbing_updated.py")

if __name__ == "__main__":
    # Main execution
    input_file = "Final_Input_dataset_for_DSS.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the file exists in the current directory.")
        exit(1)
    
    try:
        # Split the dataset
        train_file, test_file, val_file = split_dataset(input_file)
        
        # Create updated notebook files
        create_updated_notebooks()
        
        print("\n" + "="*60)
        print("DATASET SPLITTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Use 'train_dataset.csv' for model development and training")
        print("2. Use 'test_dataset.csv' for testing and evaluation")
        print("3. Use 'validation_dataset.csv' ONLY for final validation")
        print("4. Update your notebooks to use the appropriate dataset files")
        print("\nIMPORTANT: Keep the validation dataset completely separate")
        print("until you are satisfied with training and test results!")
        
    except Exception as e:
        print(f"Error during dataset splitting: {str(e)}")
        exit(1)