# Updated Stock Portfolio Optimization using SHLO
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
    
    print(f"\nRunning optimization on {DATASET_MODE} dataset...")
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
    
    print(f"\nOptimization completed on {DATASET_MODE} dataset")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
