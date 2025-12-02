# GNN-Enhanced Stock Portfolio Optimization Results

## Experiment Overview

This document presents the results of integrating Graph Neural Networks (GNN) with 
traditional optimization algorithms (SHLO and Hill Climbing) for stock portfolio optimization.

### Innovation

**Novel Contribution**: We introduce a simple Graph Neural Network layer that models 
relationships between stocks based on:
1. **Category similarity** - Stocks in the same market cap category are connected
2. **Feature similarity** - Stocks with similar normalized metrics are connected
3. **Message passing** - Information is aggregated from neighboring stocks

The GNN produces enhanced feature representations that capture not just individual 
stock characteristics but also their relationships within the stock universe.

### Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Portfolio Size | 10 stocks |
| Large-cap Stocks | 5 |
| Mid-cap Stocks | 2 |
| Small-cap Stocks | 3 |
| Total Budget | $10,000 |
| Budget per Stock | 5% - 20% |

### Objective Function Weights

| Weight | Value |
|--------|-------|
| Fitness Score | 0.50 |
| Percent Change from Intrinsic | 0.20 |
| Revenue Growth | 0.25 |
| Budget Utilization | 0.05 |

---

## Results Summary

### Performance Comparison

| Algorithm | Objective Value | Budget Used | Budget % | Time (s) |
|-----------|-----------------|-------------|----------|----------|
| **Baseline SHLO** | 5.5655 | $5,695 | 57.0% | 4.83 |
| **GNN-SHLO** | 6.3696 | $6,523 | 65.2% | 4.98 |
| **Baseline HC** | 7.0068 | $6,875 | 68.8% | 0.06 |
| **GNN-HC** | 7.5926 | $6,351 | 63.5% | 0.09 |

### Improvement Analysis

| Comparison | Baseline | GNN-Enhanced | Improvement |
|------------|----------|--------------|-------------|
| SHLO | 5.5655 | 6.3696 | +14.45% |
| Hill Climbing | 7.0068 | 7.5926 | +8.36% |

---

## Selected Portfolios

### Baseline SHLO Portfolio

| Stock ID | Symbol | Company | Price | Category | Quantity |
|----------|--------|---------|-------|----------|----------|
| 31 | TM | Toyota Motor | $185 | Large | 3 |
| 34 | AMD | Advanced Micro Devices | $138 | Large | 4 |
| 53 | INTU | Intuit | $610 | Large | 1 |
| 59 | QCOM | QUALCOMM | $138 | Large | 4 |
| 60 | HDB | HDFC Bank | $65 | Large | 8 |
| 103 | KHC | The Kraft Heinz Company | $57 | Mid | 9 |
| 165 | DECK | Deckers Outdoor | $719 | Mid | 1 |
| 182 | MGA | Magna International | $54 | Small | 10 |
| 387 | PNR | Pentair plc | $81 | Small | 7 |
| 394 | KMX | CarMax, Inc. | $81 | Small | 7 |

**Total Budget Used**: $5,695 (57.0%)
**Objective Value**: 5.5655

### GNN-SHLO Portfolio

| Stock ID | Symbol | Company | Price | Category | Quantity |
|----------|--------|---------|-------|----------|----------|
| 5 | NVDA | NVIDIA | $480 | Large | 2 |
| 51 | CMCSA | Comcast | $53 | Large | 10 |
| 76 | RY | Royal Bank of Canada | $95 | Large | 6 |
| 138 | PTC | PTC Inc. | $169 | Mid | 3 |
| 228 | RACE | Ferrari N.V. | $420 | Large | 2 |
| 278 | ADSK | Autodesk, Inc. | $251 | Large | 4 |
| 329 | HWM | Howmet Aerospace Inc. | $67 | Mid | 8 |
| 369 | LW | Lamb Weston Holdings, Inc. | $101 | Small | 5 |
| 404 | TECH | Bio-Techne Corporation | $77 | Small | 7 |
| 407 | PFGC | Performance Food Group Company | $76 | Small | 7 |

**Total Budget Used**: $6,523 (65.2%)
**Objective Value**: 6.3696

### Baseline HC Portfolio

| Stock ID | Symbol | Company | Price | Category | Quantity |
|----------|--------|---------|-------|----------|----------|
| 60 | HDB | HDFC Bank | $65 | Large | 8 |
| 76 | RY | Royal Bank of Canada | $95 | Large | 6 |
| 380 | NBIX | Neurocrine Biosciences, Inc. | $139 | Small | 4 |
| 165 | DECK | Deckers Outdoor | $719 | Mid | 1 |
| 402 | MEDP | Medpace Holdings, Inc. | $405 | Small | 2 |
| 5 | NVDA | NVIDIA | $480 | Large | 2 |
| 22 | ASML | ASML Holding | $735 | Large | 1 |
| 375 | BAP | Credicorp Ltd. | $175 | Small | 3 |
| 304 | ACGL | Arch Capital Group Ltd. | $87 | Mid | 6 |
| 37 | NFLX | Netflix | $479 | Large | 2 |

**Total Budget Used**: $6,875 (68.8%)
**Objective Value**: 7.0068

### GNN-HC Portfolio

| Stock ID | Symbol | Company | Price | Category | Quantity |
|----------|--------|---------|-------|----------|----------|
| 290 | MPWR | Monolithic Power Systems, Inc. | $732 | Mid | 1 |
| 375 | BAP | Credicorp Ltd. | $175 | Small | 3 |
| 5 | NVDA | NVIDIA | $480 | Large | 2 |
| 183 | VRTX | Vertex Pharmaceuticals Incorpo | $413 | Large | 2 |
| 380 | NBIX | Neurocrine Biosciences, Inc. | $139 | Small | 4 |
| 15 | NVO | Novo Nordisk | $98 | Large | 6 |
| 60 | HDB | HDFC Bank | $65 | Large | 8 |
| 3 | GOOGL | Alphabet | $133 | Large | 4 |
| 407 | PFGC | Performance Food Group Company | $76 | Small | 7 |
| 125 | LEN | Lennar | $145 | Mid | 4 |

**Total Budget Used**: $6,351 (63.5%)
**Objective Value**: 7.5926

---

## GNN Architecture

### Graph Construction
- **Nodes**: Each stock is a node with features:
  - Normalized fitness score
  - Normalized percent change from intrinsic value
  - Normalized revenue growth
  - Normalized stock price
  - Category one-hot encoding (3 dimensions)

- **Edges**: Stocks are connected if:
  - Same market cap category, OR
  - Cosine similarity > 0.3

### Network Architecture
```
Input Features (8 dims) 
    → GNN Layer 1 (16 hidden units, ReLU)
    → GNN Layer 2 (8 output units)
    → Enhanced Embeddings
```

### Message Passing
Each layer performs:
1. **Aggregation**: Collect features from neighboring nodes
2. **Transformation**: Linear projection with learnable weights
3. **Activation**: ReLU (except final layer)

### Feature Enhancement
Enhanced scores = 0.8 × Original scores + 0.2 × GNN embeddings

---

## Conclusions

### Key Findings

1. **GNN Enhancement Improves Both Algorithms**
   - SHLO improved by 14.45%
   - Hill Climbing improved by 8.36%

2. **Graph-based Feature Learning is Beneficial**
   - Capturing stock relationships helps identify better portfolio combinations
   - Stocks that are similar to high-performing stocks get boosted scores

3. **Minimal Overhead**
   - GNN adds negligible computational cost
   - Simple 2-layer architecture is sufficient for improvement

### Future Work

1. **Dynamic Graphs**: Update edges based on temporal correlations
2. **Attention Mechanisms**: Learn edge importance dynamically
3. **Deeper Networks**: Explore more layers for complex relationships
4. **Sector Information**: Include industry/sector as additional graph structure

---

## Technical Notes

- **Random Seed**: 42 (for reproducibility)
- **GNN Implementation**: Pure NumPy (no deep learning framework required)
- **Adjacency Matrix**: Symmetric normalized Laplacian
- **Activation**: ReLU for hidden layers, linear for output

---

*Generated by GNN-Enhanced Portfolio Optimization System*
