# Dataset Split Documentation

## Overview
This document explains the train/test/validation dataset split implemented for the Stock Portfolio Optimization Project as required for academic purposes.

## Dataset Split Summary

The main dataset `Final_Input_dataset_for_DSS.csv` has been split into three separate datasets:

| Dataset | Samples | Percentage | File Size |
|---------|---------|------------|-----------|
| **Training** | 319 | 69.8% | 38.9 KB |
| **Test** | 92 | 20.1% | 11.2 KB |
| **Validation** | 46 | 10.1% | 5.6 KB |
| **Total** | 457 | 100% | 55.4 KB |

## Files Created

### Split Datasets
- `train_dataset.csv` - Training dataset (319 samples)
- `test_dataset.csv` - Test dataset (92 samples)  
- `validation_dataset.csv` - Validation dataset (46 samples)

### Updated Code Files
- `Stock_portfolio_optimization_using_SHLO_updated.py` - Updated SHLO implementation
- `Stock_portfolio_selection_by_hill_climbing_updated.py` - Updated Hill Climbing implementation
- `dataset_splitter.py` - Script used to create the splits

## Dataset Structure

Each split dataset contains the same columns as the original:
- Stock_ID
- Symbol
- Company Name
- Stock Price
- Category
- Combined_Fitness_Score
- Intrinsic Value
- percent_from_instrinsic_value
- Rev Gr. Next Y
- Lower_limit
- Upper_limit
- normalized_fitness_score
- normalized_percent_change_from_instrinsic_value
- normalized_Rev_Gr_Next_Y

## Usage Guidelines

### 1. Training Phase
- Use `train_dataset.csv` for algorithm development and parameter tuning
- This is where you experiment with different optimization parameters
- Run multiple iterations to find the best hyperparameters

### 2. Testing Phase
- Use `test_dataset.csv` for evaluating your optimized algorithms
- Compare performance between SHLO and Hill Climbing algorithms
- Fine-tune based on test results

### 3. Validation Phase (FINAL)
- **IMPORTANT**: Use `validation_dataset.csv` ONLY for final validation
- This dataset should remain completely untouched until you are satisfied with training and test results
- This provides an unbiased estimate of your algorithm's performance

## Implementation Notes

### Random Seed
- All splits use `random_state=42` for reproducibility
- The same random seed ensures consistent results across runs

### Split Method
- Simple random sampling without stratification
- Maintains the original distribution of stock categories and characteristics

### Code Updates
The updated Python files include:
- Dataset mode configuration (`DATASET_MODE`)
- Automatic file path selection based on mode
- Proper dataset loading and validation

## Running the Updated Code

### For Training:
```python
DATASET_MODE = "train"  # Uses train_dataset.csv
```

### For Testing:
```python
DATASET_MODE = "test"   # Uses test_dataset.csv
```

### For Validation:
```python
DATASET_MODE = "validation"  # Uses validation_dataset.csv
```

## Academic Compliance

This implementation follows the standard machine learning practice of:
1. **70% Training** - For model development and parameter optimization
2. **20% Testing** - For performance evaluation and algorithm comparison
3. **10% Validation** - For final unbiased performance assessment

The validation set is kept completely separate until final testing, ensuring no data leakage and providing an honest assessment of algorithm performance.

## Verification

To verify the splits are correct:
```bash
# Check total samples
python3 -c "
import pandas as pd
train = pd.read_csv('train_dataset.csv')
test = pd.read_csv('test_dataset.csv')
val = pd.read_csv('validation_dataset.csv')
print(f'Total samples: {len(train) + len(test) + len(val)}')
print(f'Original samples: 457')
print(f'Match: {len(train) + len(test) + len(val) == 457}')
"
```

## Next Steps

1. **Development**: Use the training set to develop and optimize your algorithms
2. **Evaluation**: Use the test set to compare algorithm performance
3. **Final Validation**: Use the validation set only for final performance assessment
4. **Documentation**: Record results from each phase for your research paper

## Important Reminders

- ⚠️ **Never use the validation set during development**
- ⚠️ **Keep validation results separate until final testing**
- ⚠️ **Document which dataset was used for each experiment**
- ⚠️ **Maintain reproducibility by using the same random seed**

This implementation ensures your research follows proper machine learning methodology and provides credible results for academic evaluation.
