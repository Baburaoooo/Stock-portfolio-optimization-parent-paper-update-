# Complete Code Explanation for Video Recording
## Stock Portfolio Optimization Project

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Libraries and Dependencies](#libraries-and-dependencies)
4. [Module-by-Module Explanation](#module-by-module-explanation)
5. [Optimization Algorithms](#optimization-algorithms)
6. [Dataset Structure](#dataset-structure)
7. [How Everything Works Together](#how-everything-works-together)
8. [Key Features and Achievements](#key-features-and-achievements)

---

## 🎯 Project Overview

This project implements a **Financial Decision Support System (DSS)** for stock portfolio optimization using **nature-inspired algorithms**. The system helps investors select an optimal portfolio of stocks that maximizes returns while minimizing risk.

### Main Goal
Create an intelligent system that:
- Analyzes stock fundamentals
- Calculates intrinsic values
- Assesses financial health
- Optimizes portfolio selection using AI algorithms
- Outperforms market indices (achieved >80% of the time)

### Novel Approach
The project uses a **hybrid methodology** combining:
1. **Machine Learning** for stock price prediction
2. **Fundamental Analysis** for intrinsic value calculation
3. **Financial Health Analysis** for fitness scoring
4. **Nature-Inspired Optimization** for portfolio selection

---

## 🏗️ System Architecture

The project consists of **4 main modules** plus **dataset management**:

```
┌─────────────────────────────────────────────────────────┐
│         STOCK PORTFOLIO OPTIMIZATION SYSTEM             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Module 1: Stock Price Prediction (ML)                 │
│  Module 2: Intrinsic Value Calculation                  │
│  Module 3: Financial Health Analysis                    │
│  Module 4: Portfolio Optimization (SHLO/Hill Climbing) │
│                                                         │
│  Dataset Management: Train/Test/Validation Split       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 Libraries and Dependencies

### Core Libraries

1. **NumPy** (`numpy`)
   - Purpose: Numerical computations, array operations
   - Used for: Mathematical calculations, data manipulation

2. **Pandas** (`pandas`)
   - Purpose: Data manipulation and analysis
   - Used for: Reading CSV files, dataframes, data processing

3. **Matplotlib** (`matplotlib`)
   - Purpose: Data visualization
   - Used for: Plotting optimization progress, results visualization

4. **Seaborn** (`seaborn`)
   - Purpose: Statistical data visualization
   - Used for: Enhanced plotting and visualizations

5. **Yahoo Finance** (`yfinance`)
   - Purpose: Fetching real-time stock data
   - Used for: Getting current stock prices

6. **Yahoo Finance API** (`yahoo_fin`)
   - Purpose: Alternative stock data source
   - Used for: Stock information retrieval

### Standard Python Libraries
- `random`: Random number generation for algorithms
- `time`: Performance measurement
- `copy`: Deep copying for algorithm operations
- `csv`: CSV file handling
- `datetime`: Date/time operations
- `json`, `requests`, `urllib`: API calls and data fetching

---

## 🔍 Module-by-Module Explanation

### **MODULE 1: Intrinsic Value Calculation**
**File**: `Intrinsic_Value_Calculation.ipynb`

#### Purpose
Calculates the **intrinsic (true) value** of each stock based on fundamental financial metrics.

#### How It Works

1. **Data Loading**
   - Reads stock fundamental data from CSV
   - Columns include: Market Cap, EPS Growth, Beta, FCF, Debt, Cash, etc.

2. **Data Cleaning**
   - Removes special characters (%, B, M, K) from numeric columns
   - Converts values to proper numeric format
   - Handles missing values (fills with average EPS growth)

3. **Discount Rate Calculation**
   ```python
   discount_rate = risk_free_rate + (beta × market_risk_premium)
   ```
   - Risk-free rate: 2% (10-year bond rate)
   - Market risk premium: 5%
   - Beta: Stock's volatility measure

4. **Intrinsic Value Calculation**
   - Uses **Discounted Cash Flow (DCF)** method
   - Projects cash flows for 10 years:
     - Years 1-5: Full EPS growth rate
     - Years 6-10: Half of EPS growth rate (conservative)
   - Discounts future cash flows to present value
   - Formula: `Intrinsic Value = (Sum of Discounted Cash Flows - Debt + Cash) / Shares Outstanding`

5. **Percent Deviation Calculation**
   - Calculates how far current price is from intrinsic value
   - Formula: `(1 - Stock Price / Intrinsic Value) × 100`
   - Positive = undervalued, Negative = overvalued

6. **Output**
   - Creates `Calculated_Intrinsic_Value_Dataset.csv`
   - Contains: Stock_ID, Intrinsic Value, percent_from_intrinsic_value

---

### **MODULE 2: Financial Health Analysis**
**File**: `US_Stock_Finanacial_Health_Analysis.ipynb`

#### Purpose
Evaluates the **financial health** of each stock using multiple financial metrics and assigns a fitness score.

#### How It Works

1. **Data Loading**
   - Reads stock health metrics from CSV
   - Metrics include: PE Ratio, EPS Growth, Revenue Growth, ROE, ROIC, etc.

2. **Data Cleaning**
   - Removes % and B symbols
   - Converts to numeric format

3. **Fitness Scoring Functions**
   Each metric gets a score from 1-5 based on thresholds:
   
   - **PE Ratio**: Lower is better (1-5 scale)
   - **EPS Growth**: Higher is better (1-5 scale)
   - **Revenue Growth**: Higher is better (1-5 scale)
   - **ROE (Return on Equity)**: Moderate is best (1-5 scale)
   - **ROIC (Return on Invested Capital)**: Moderate is best (1-5 scale)
   - **Gross Profit Growth**: Higher is better (1-5 scale)
   - **Net Income Growth**: Higher is better (1-5 scale)
   - **Profit Margin**: Higher is better (1-5 scale)
   - **Debt/Equity Ratio**: Lower is better (1-5 scale)
   - **Dividend Yield**: Moderate is best (1-5 scale)

4. **Weighted Combined Fitness Score**
   ```python
   Combined_Fitness_Score = Σ (Fitness_Metric_i × Weight_i)
   ```
   - Weights sum to 1.0
   - Example weights:
     - Revenue Growth: 18%
     - ROE: 15%
     - PE Ratio: 12%
     - EPS Growth: 12%
     - Others: 5-10% each

5. **Output**
   - Creates `Stock_fitness_values.csv`
   - Contains: Stock_ID, all fitness scores, Combined_Fitness_Score

---

### **MODULE 3: Dataset Management**
**Files**: `dataset_splitter.py`, `test_dataset_splits.py`

#### Purpose
Implements proper **academic methodology** by splitting data into train/test/validation sets (70/20/10).

#### How It Works

1. **Dataset Splitter** (`dataset_splitter.py`)
   - Reads `Final_Input_dataset_for_DSS.csv` (457 stocks)
   - Randomly shuffles data (seed=42 for reproducibility)
   - Splits into:
     - **Training**: 319 stocks (69.8%)
     - **Test**: 92 stocks (20.1%)
     - **Validation**: 46 stocks (10.1%)
   - Saves three separate CSV files
   - Ensures no overlap between splits

2. **Verification Script** (`test_dataset_splits.py`)
   - Tests dataset loading
   - Verifies data integrity (no duplicates)
   - Checks algorithm compatibility
   - Ensures all required columns present

3. **Academic Compliance**
   - Training set: Algorithm development & parameter tuning
   - Test set: Performance evaluation & comparison
   - Validation set: Final unbiased assessment (used only once)

---

### **MODULE 4: Portfolio Optimization Algorithms**

#### **4A. SHLO Algorithm (Social Human Learning Optimization)**
**File**: `Stock_portfolio_optimization_using_SHLO.ipynb`

##### What is SHLO?
A **nature-inspired metaheuristic algorithm** that mimics human learning behavior:
- **Individual Learning**: Learning from personal experience
- **Social Learning**: Learning from others
- **Random Learning**: Exploration of new possibilities

##### Algorithm Components

1. **Node Class**
   - Represents a portfolio solution
   - Contains: solution (binary array), fitness value, Individual Knowledge Database (IKD)

2. **Social Knowledge Database (SKD)**
   - Global knowledge shared across population
   - Stores 9 best solutions found by the community

3. **Individual Knowledge Database (IKD)**
   - Personal knowledge for each individual
   - Stores 5 best solutions found by that individual

4. **Population Initialization**
   - Creates 100 individuals (POP_SIZE = 100)
   - Each individual has a random portfolio of PORTFOLIO_SIZE stocks
   - Initializes IKD and SKD with random solutions

5. **Budget Allocation**
   - Ensures each stock investment is within budget limits
   - Upper limit: e.g., 20% of total budget per stock
   - Lower limit: e.g., 5% of total budget per stock
   - Calculates quantity of shares to buy

6. **Constraints**
   - **Portfolio Size**: Exactly PORTFOLIO_SIZE stocks
   - **Category Constraint**: Specific number of large-cap, mid-cap, small-cap stocks
   - **Budget Constraint**: Total investment ≤ total budget
   - **Individual Stock Budget**: Each stock within upper/lower limits

7. **Fitness Function**
   ```python
   Fitness = Σ [weight_fitness × normalized_fitness_score +
                weight_percent_change × normalized_percent_change +
                weight_rev_growth × normalized_rev_growth +
                weight_normalized_budget × normalized_budget]
   ```
   - Maximizes: Fitness score, undervaluation, revenue growth
   - Considers: Budget utilization

8. **Evolution Process** (500 epochs)
   
   For each individual in each epoch:
   
   a. **Learning Phase** (3 types with probabilities):
      - **Random Learning (40%)**: Randomly flip stock selections
      - **Hybrid Learning (30%)**: Mix of individual and social learning
      - **Social Learning (30%)**: Learn from SKD
   
   b. **Perturbation Rate**:
      - Starts at 50%, decreases over time
      - Encourages exploration early, exploitation later
   
   c. **Constraint Checking**:
      - Ensures portfolio size is correct
      - Verifies all constraints are met
      - Reverts if constraints violated
   
   d. **Fitness Evaluation**:
      - Calculates new fitness
      - Updates if better than current
   
   e. **Knowledge Update**:
      - Updates IKD if current solution is better
      - Updates SKD if current solution is globally better

9. **Output**
   - Best portfolio found
   - Total budget used
   - Best objective function value
   - Time taken

---

#### **4B. Hill Climbing Algorithm**
**File**: `Stock_portfolio_selection_by_hill_climbing_algorithm.ipynb`

##### What is Hill Climbing?
A **local search optimization algorithm** that:
- Starts with an initial solution
- Generates neighboring solutions
- Moves to better neighbors
- Stops when no better neighbor exists

##### Algorithm Components

1. **Initial Solution Generation**
   - Randomly selects PORTFOLIO_SIZE stocks
   - Ensures all constraints are met
   - Calculates initial fitness

2. **Neighbor Generation**
   - Randomly selects 2 stocks from current portfolio
   - Swaps them with 2 new stocks from dataset
   - Maintains category constraints
   - Ensures budget constraints are met
   - Retries up to 1000 times if constraints not met

3. **Budget Allocation**
   - Similar to SHLO
   - Upper limit: 20% of total budget
   - Lower limit: 5% of total budget
   - Calculates feasible quantities

4. **Objective Function**
   - Same as SHLO fitness function
   - Maximizes weighted combination of metrics

5. **Hill Climbing Process** (5000 iterations)
   ```python
   for iteration in range(5000):
       neighbor = generate_neighbor(current_portfolio)
       if neighbor_fitness > current_fitness:
           current_portfolio = neighbor
           current_fitness = neighbor_fitness
   ```

6. **Output**
   - Best portfolio found
   - Total budget used
   - Best objective function value
   - Time taken

---

## 📊 Dataset Structure

### Main Dataset: `Final_Input_dataset_for_DSS.csv`

**Columns:**
1. **Stock_ID**: Unique identifier
2. **Symbol**: Stock ticker (e.g., AAPL, MSFT)
3. **Company Name**: Full company name
4. **Stock Price**: Current market price
5. **Category**: 
   - 1 = Large-cap
   - 2 = Mid-cap
   - 3 = Small-cap
6. **Combined_Fitness_Score**: Financial health score (from Module 2)
7. **Intrinsic Value**: Calculated true value (from Module 1)
8. **percent_from_intrinsic_value**: Deviation from intrinsic value
9. **Rev Gr. Next Y**: Revenue growth for next year
10. **Lower_limit**: Minimum quantity of shares
11. **Upper_limit**: Maximum quantity of shares
12. **normalized_fitness_score**: Normalized fitness (0-1)
13. **normalized_percent_change_from_instrinsic_value**: Normalized deviation
14. **normalized_Rev_Gr_Next_Y**: Normalized revenue growth

**Total**: 457 stocks

---

## 🔄 How Everything Works Together

### Complete Workflow

```
Step 1: Data Preparation
├── Load fundamental stock data
├── Calculate intrinsic values (Module 1)
├── Calculate fitness scores (Module 2)
└── Create final input dataset

Step 2: Dataset Splitting
├── Split into train/test/validation (70/20/10)
└── Verify data integrity

Step 3: User Input
├── Portfolio size (number of stocks)
├── Category constraints (large/mid/small-cap counts)
├── Budget constraints (total, upper, lower limits)
└── Objective function weights

Step 4: Algorithm Execution
├── Initialize population/solution
├── Run optimization (SHLO or Hill Climbing)
├── Evaluate fitness at each iteration
└── Update best solution

Step 5: Output
├── Best portfolio selection
├── Budget utilization
├── Objective function value
└── Performance metrics
```

### Example Run

1. **User Inputs**:
   - Portfolio size: 12 stocks
   - Large-cap: 6, Mid-cap: 3, Small-cap: 3
   - Total budget: $30,000
   - Upper limit: 20%, Lower limit: 5%
   - Weights: Fitness=0.3, Percent Change=0.1, Revenue Growth=0.55, Budget=0.05

2. **Algorithm Runs**:
   - SHLO: 500 epochs, 100 individuals
   - Hill Climbing: 5000 iterations

3. **Output**:
   - Selected stocks with quantities
   - Total budget used: ~$22,000-$29,000
   - Objective function value: 6-8 (higher is better)
   - Time: SHLO ~500s, Hill Climbing ~70s

---

## ✨ Key Features and Achievements

### 1. **Multi-Dimensional Analysis**
   - Combines price prediction, intrinsic value, and health analysis
   - More comprehensive than single-metric approaches

### 2. **Constraint Handling**
   - Portfolio size constraints
   - Category diversification
   - Budget limits (total and per-stock)
   - All constraints enforced simultaneously

### 3. **Nature-Inspired Optimization**
   - SHLO: Mimics human learning (individual + social)
   - Hill Climbing: Simple but effective local search
   - Both find near-optimal solutions

### 4. **Academic Rigor**
   - Proper train/test/validation split
   - Reproducible results (random seed)
   - No data leakage
   - Credible performance assessment

### 5. **Performance**
   - **>80% of the time outperforms US market indices**
   - Considers multiple factors simultaneously
   - Adapts to different budget and constraint scenarios

### 6. **Flexibility**
   - User-defined weights for objective function
   - Adjustable portfolio size
   - Customizable budget constraints
   - Works with different stock universes

---

## 🎬 Video Recording Tips

### Suggested Structure for Your Video:

1. **Introduction (2-3 min)**
   - Project overview and goals
   - Why portfolio optimization matters
   - Novel approach explanation

2. **System Architecture (3-4 min)**
   - Show the 4 modules
   - Explain how they connect
   - Dataset structure overview

3. **Module 1: Intrinsic Value (3-4 min)**
   - Explain DCF method
   - Show calculation process
   - Demonstrate output

4. **Module 2: Health Analysis (3-4 min)**
   - Explain fitness scoring
   - Show weighted combination
   - Demonstrate output

5. **Module 3: Dataset Management (2-3 min)**
   - Explain train/test/validation split
   - Show why it's important
   - Demonstrate verification

6. **Module 4: Optimization Algorithms (8-10 min)**
   - SHLO algorithm explanation
   - Hill Climbing explanation
   - Show how they work
   - Compare approaches

7. **Complete Workflow Demo (5-6 min)**
   - Run a complete example
   - Show user inputs
   - Show algorithm execution
   - Show final results

8. **Results and Conclusion (2-3 min)**
   - Performance achievements
   - Key features
   - Future improvements

**Total Video Length: ~30-35 minutes**

---

## 📝 Key Points to Emphasize

1. **Hybrid Methodology**: Combining ML, fundamental analysis, and optimization
2. **Nature-Inspired Algorithms**: SHLO mimics human learning behavior
3. **Multi-Objective Optimization**: Considers fitness, value, growth, budget
4. **Constraint Satisfaction**: Handles complex real-world constraints
5. **Academic Rigor**: Proper dataset splitting methodology
6. **Performance**: Outperforms market indices consistently
7. **Practical Application**: Real-world usable system

---

## 🔧 Technical Details for Deep Dive

### SHLO Algorithm Complexity
- **Time Complexity**: O(EPOCHS × POP_SIZE × num_stocks × PORTFOLIO_SIZE)
- **Space Complexity**: O(POP_SIZE × num_stocks)
- **Typical Runtime**: 400-600 seconds for 500 epochs

### Hill Climbing Complexity
- **Time Complexity**: O(iterations × num_stocks)
- **Space Complexity**: O(PORTFOLIO_SIZE)
- **Typical Runtime**: 60-80 seconds for 5000 iterations

### Normalization
All metrics are normalized to 0-1 range for fair comparison:
- `normalized_value = (value - min) / (max - min)`

### Objective Function Weights
- Must sum to 1.0
- Higher weight = more important in optimization
- User can adjust based on investment strategy

---

## 🎯 Summary

This project implements a **comprehensive stock portfolio optimization system** that:

1. **Analyzes** stocks from multiple dimensions (price, value, health)
2. **Optimizes** portfolio selection using nature-inspired algorithms
3. **Satisfies** complex real-world constraints
4. **Outperforms** market indices consistently
5. **Follows** academic best practices

The code is well-structured, documented, and ready for academic presentation and real-world application.

---

**Good luck with your video recording! 🎥**

