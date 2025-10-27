#!/usr/bin/env python3
"""
Test script to verify that the dataset splits work correctly
with the portfolio optimization algorithms.
"""

import pandas as pd
import numpy as np

def test_dataset_loading():
    """Test that all split datasets can be loaded correctly"""
    print("Testing dataset loading...")
    
    datasets = {
        'train': 'train_dataset.csv',
        'test': 'test_dataset.csv', 
        'validation': 'validation_dataset.csv'
    }
    
    for mode, filename in datasets.items():
        try:
            df = pd.read_csv(filename)
            print(f"✓ {mode.capitalize()} dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
            
            # Check required columns exist
            required_cols = ['Stock_ID', 'Symbol', 'Company Name', 'Stock Price', 'Category', 
                           'Combined_Fitness_Score', 'Intrinsic Value', 'percent_from_instrinsic_value']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ⚠️  Missing columns: {missing_cols}")
            else:
                print(f"  ✓ All required columns present")
                
        except Exception as e:
            print(f"✗ Error loading {mode} dataset: {e}")
            return False
    
    return True

def test_data_integrity():
    """Test data integrity across splits"""
    print("\nTesting data integrity...")
    
    # Load all datasets
    train_df = pd.read_csv('train_dataset.csv')
    test_df = pd.read_csv('test_dataset.csv')
    val_df = pd.read_csv('validation_dataset.csv')
    
    # Check total samples
    total_samples = len(train_df) + len(test_df) + len(val_df)
    print(f"Total samples across splits: {total_samples}")
    
    if total_samples == 457:
        print("✓ Total samples match original dataset")
    else:
        print(f"✗ Total samples mismatch. Expected: 457, Got: {total_samples}")
        return False
    
    # Check for duplicate Stock_IDs across splits
    train_ids = set(train_df['Stock_ID'])
    test_ids = set(test_df['Stock_ID'])
    val_ids = set(val_df['Stock_ID'])
    
    overlaps = train_ids.intersection(test_ids).union(train_ids.intersection(val_ids)).union(test_ids.intersection(val_ids))
    
    if len(overlaps) == 0:
        print("✓ No duplicate Stock_IDs across splits")
    else:
        print(f"✗ Found duplicate Stock_IDs: {overlaps}")
        return False
    
    # Check data ranges
    print(f"Stock_ID ranges:")
    print(f"  Train: {train_df['Stock_ID'].min()} - {train_df['Stock_ID'].max()}")
    print(f"  Test:  {test_df['Stock_ID'].min()} - {test_df['Stock_ID'].max()}")
    print(f"  Val:   {val_df['Stock_ID'].min()} - {val_df['Stock_ID'].max()}")
    
    return True

def test_algorithm_compatibility():
    """Test that the datasets are compatible with the optimization algorithms"""
    print("\nTesting algorithm compatibility...")
    
    # Test with training dataset
    try:
        df = pd.read_csv('train_dataset.csv')
        
        # Simulate the data loading process from the original notebooks
        stocks = df.copy()
        
        # Check if we have enough stocks for portfolio optimization
        min_portfolio_size = 5  # Minimum reasonable portfolio size
        if len(stocks) >= min_portfolio_size:
            print(f"✓ Training dataset has sufficient stocks ({len(stocks)}) for portfolio optimization")
        else:
            print(f"✗ Training dataset has insufficient stocks ({len(stocks)}) for portfolio optimization")
            return False
        
        # Check if required columns have valid data
        numeric_cols = ['Stock Price', 'Combined_Fitness_Score', 'Intrinsic Value', 
                       'percent_from_instrinsic_value', 'Rev Gr. Next Y']
        
        for col in numeric_cols:
            if col in stocks.columns:
                non_null_count = stocks[col].count()
                if non_null_count > 0:
                    print(f"  ✓ {col}: {non_null_count}/{len(stocks)} non-null values")
                else:
                    print(f"  ✗ {col}: No valid values")
                    return False
        
        print("✓ Dataset is compatible with optimization algorithms")
        return True
        
    except Exception as e:
        print(f"✗ Error testing algorithm compatibility: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("DATASET SPLIT VERIFICATION TESTS")
    print("="*60)
    
    tests = [
        test_dataset_loading,
        test_data_integrity,
        test_algorithm_compatibility
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
        print()
    
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Dataset splits are ready for use!")
        print("\nYou can now proceed with:")
        print("1. Training phase using train_dataset.csv")
        print("2. Testing phase using test_dataset.csv") 
        print("3. Final validation using validation_dataset.csv")
    else:
        print("✗ SOME TESTS FAILED - Please check the dataset splits")
    
    print("="*60)

if __name__ == "__main__":
    main()
