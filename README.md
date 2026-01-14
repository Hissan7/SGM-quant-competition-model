![KCL Logo](KCL.png) ![SGM Logo](SGM.png)

# Strand Global Macro Quant Competition  

**Macro-Driven Portfolio Optimisation with Risk, Turnover, and Dividend Awareness**

This project is a supervised cross-sectional return model using sell-side information and uncertainty-aware signals, integrated into a robust, constrained portfolio optimisation framework.

The goal is to manage a portfolio of equities while:

- Aligning allocations with the global business cycle.
- Maintaining a low-to-medium risk profile.
- Respecting real-world trading constraints
and integrating dividends, volatility forecasting, and correlation structure.

### Project structure 

```
SGM-quant-competition-model/
│
├── data/
│   └── processed/
│       └── universe.csv          # Equity universe with price targets
│
├── src/
│   ├── optimiser.py              # Core convex optimisation model
│   ├── simulate.py               # End-to-end portfolio simulation
│   ├── ablation.py               # Model variant comparisons
│   ├── macro_controller.py       # Macro regime & rotation logic
│   ├── risk_model.py             # Volatility & covariance estimation
│   ├── ml_signal.py              # ML-based expected return modelling
│   ├── dividends.py              # Dividend yield estimation
│   ├── performance.py            # Sharpe, drawdown, equity curve
│   └── data_prep.py              # Data cleaning & preprocessing
│
├── results/
│   ├── metrics/                  # CSV outputs
│   └── plots/                    # Generated figures
│
├── README.md
├── requirements.txt
└── understanding.txt             # Personal learning notes
```

The model begins with high defensive exposure and gradually rotates into growth and AI-linked equities over a 6–9 month horizon, subject to volatility, turnover, and position limits.

## Installation and setup 

Below shows the steps to install and setup the environment

### 1. Create and activate a virtal environment 

```sh
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

### 2. Install dependencies 

```sh
pip install -r requirements.txt
```

## How to run the project 

There is a main simulation that can be run to output metrics respective to the excel file provided, as well as other smaller individual features.

### 1. Main simulation 

This is a portfolio optimization simulator that rebalances an investment portfolio over time (10 months by default) while managing risk, turnover, and macro sector allocations.

Core workflow:

1. Loads universe data - Reads a CSV of stocks/assets with their characteristics

2. Categorizes assets - Groups them into buckets (likely by sector or strategy type)

3. Gets dividend yields - Fetches trailing twelve-month dividend data for each ticker

4. Builds risk model - Creates a covariance matrix (Sigma) from 3 years of historical data to predict portfolio volatility

5. Monthly optimization loop - For each of 10 months:

6. Solves for optimal portfolio weights given constraints

7. Tracks metrics like turnover, sector exposures, predicted volatility

8. Uses previous month's weights as starting point

Key constraints being managed:

1. Macro allocation targets - Maintains minimum/maximum weights in three buckets: Defensive, Growth/AI, and Cyclical

2. Turnover limit - Keeps portfolio changes under 30% per month

3. Volatility cap - Constrains predicted annual volatility to 10%

4. Slack variables - Can relax macro constraints when they conflict with the vol cap (mentioned in months 5-9)

Run using :

```sh
python -m src.simulate
```

#### Outputs 

1. `results/metrics/simulation.csv` - The main csv showing weight allocation in each of the defensive, growth and cyclical sectors while staying under the capped 10% volatility constraint.

2. `results/metrics/weights_by_month.csv` - Updated weights for each company stock over the 9 month period.

3. Portfolio plots :

    - `rotation.png` : The main plot showing a visualisation of the rotation of portoflio weights for each of the industry sectors 
    
    - `turnover.png` : Line graph of the turnover relative to the 30% limit cap 

    - `predicted_vol` : Line graph to show volatility over the 9 month period 

    - `dividend_by_bucket.png` : A bar chart showing the average dividend yield for each of the three sectors. 

4. Expected terminal output :

```
Universe tickers: 25 ['TTE', 'SJM', 'DG.VI', '1ZB.F', '7KY.F', 'GOOG', 'NVDA', 'AMZN', 'PFE', 'GIS', 'PAHGF', 'DAG', 'NU', 'NPIFF', 'CCA.TO', 'BCE', 'DVA', 'AMRZ', 'CRWV', 'BIIB', 'VRSN', 'STZ', 'JSG.L', 'AXA SA', 'VRT']
Dividend yields meta: {'bad': [], 'good': ['1ZB.F', '7KY.F', 'AMRZ', 'AMZN', 'BCE', 'BIIB', 'CCA.TO', 'CRWV', 'DAG', 'DG.VI', 'DVA', 'GIS', 'GOOG', 'JSG.L', 'NPIFF', 'NU', 'NVDA', 'PAHGF', 'PFE', 'SJM', 'STZ', 'TTE', 'VRSN', 'VRT'], 'period': '1y'}
Dividend yields :
    Ticker  dividend_yield
8      PFE        0.067306
10   PAHGF        0.059178
15     BCE        0.055757
14  CCA.TO        0.054965
9      GIS        0.053377
0      TTE        0.042426
1      SJM        0.041742
2    DG.VI        0.040633
13   NPIFF        0.036815
22   JSG.L        0.030846
21     STZ        0.025839
20    VRSN        0.009298
5     GOOG        0.002479
24     VRT        0.001053
6     NVDA        0.000220
11     DAG        0.000000
7     AMZN        0.000000
16     DVA        0.000000
17    AMRZ        0.000000
18    CRWV        0.000000
19    BIIB        0.000000
4    7KY.F        0.000000
3    1ZB.F        0.000000
23  AXA SA        0.000000
12      NU        0.000000
Sigma shape: (25, 25)
Risk model: {'mode': 'ewma', 'n_obs': 773, 'bad': ['AXA SA'], 'good': ['TTE', 'SJM', 'DG.VI', '1ZB.F', '7KY.F', 'GOOG', 'NVDA', 'AMZN', 'PFE', 'GIS', 'PAHGF', 'DAG', 'NU', 'NPIFF', 'CCA.TO', 'BCE', 'DVA', 'AMRZ', 'CRWV', 'BIIB', 'VRSN', 'STZ', 'JSG.L', 'VRT']}
   month  pred_vol_annual  ...  growth_min_slack  defensive_min_slack
0      0              0.1  ...      5.377728e-11         5.334000e-11
1      1              0.1  ...      2.301068e-12         2.282312e-12
2      2              0.1  ...      6.818250e-13         6.810999e-13
3      3              0.1  ...      1.202922e-09         1.202921e-09
4      4              0.1  ...      0.000000e+00         0.000000e+00
5      5              0.1  ...      2.522701e-02         0.000000e+00
6      6              0.1  ...      6.003591e-02         0.000000e+00
7      7              0.1  ...      6.003591e-02         0.000000e+00
8      8              0.1  ...      6.003591e-02         0.000000e+00
9      9              0.1  ...      6.003591e-02         4.200618e-10

[10 rows x 5 columns]
Saved:
- results/metrics/simulation.csv
- results/metrics/weights_by_month.csv
- results/plots/rotation.png
- results/plots/turnover.png
- results/plots/predicted_vol.png

note:Macro minimum relaxed in months 5–9 due to vol cap feasibility.

   month      turnover  w_defensive  w_growth_ai  w_cyclical
0      0  2.459431e-01     0.599011     0.153364    0.247625
1      1  5.474669e-03     0.596274     0.153364    0.250362
2      2  1.205599e-02     0.590246     0.153364    0.256390
3      3  3.544812e-05     0.590228     0.153364    0.256408
4      4  6.289375e-02     0.558781     0.184811    0.256408
5      5  2.190540e-01     0.533655     0.239962    0.226383
6      6  2.357900e-08     0.533655     0.239962    0.226383
7      7  2.533605e-08     0.533655     0.239962    0.226383
8      8  2.746314e-08     0.533655     0.239962    0.226383
9      9  6.192627e-09     0.533655     0.239962    0.226383
```

### 2. Model ablation study

```sh
python -m src.ablation
```

This is an ablation study that tests how different optimizer features affect portfolio performance by running the same simulation with different combinations of features turned on/off.

#### What it does:

Tests four different variants of the portfolio optimizer:

1. baseline - No special features (use_macro=False, use_uncertainty=False, use_ml=False)

2. macro_only - Only macro sector constraints enabled

3. macro_robust - Macro constraints plus uncertainty handling

4. macro_robust_ml - All features enabled (macro + uncertainty + machine learning)

#### Simulation process:

For each variant:

1. Runs a 10-month portfolio optimization

2. Tracks key metrics each month: status, objective value, turnover, weights in defensive/growth/cyclical buckets, portfolio concentration (sum of squared weights)

3. Uses previous month's weights as starting point for next month

4. Compiles all monthly data into a dataframe

#### Outputs:

Computes summary statistics across all months for each variant:

1. Mean and max turnover
2. Mean and final growth allocation
3. Mean defensive allocation
4. Mean portfolio concentration
5. Mean objective value

Creates visualizations:

1. Growth allocation paths over time (comparing all 4 variants with different line styles and markers)

2. Turnover paths over time with the 30% limit shown

3. Pivot table showing growth weights by month for each variant

4. Difference calculation between macro_only and macro_robust variants

5. Expected terminal output : 

```
Growth weights by month:
 variant  baseline  macro_only  macro_robust  macro_robust_ml
month                                                       
0        0.244553    0.202751      0.202751         0.204595
1        0.244553    0.202751      0.202751         0.205576
2        0.244553    0.205540      0.205540         0.210602
3        0.244553    0.212751      0.212751         0.214595
4        0.244553    0.212751      0.212751         0.214595
5        0.244553    0.265189      0.265189         0.265189
6        0.244552    0.299998      0.299998         0.299998
7        0.244552    0.299998      0.299998         0.299998
8        0.244552    0.299998      0.299998         0.299998
9        0.244552    0.299998      0.299998         0.299998

macro_only - macro_robust:
 month
0   -2.000000e-08
1   -2.000000e-08
2   -2.000000e-08
3   -2.000000e-08
4   -2.000000e-08
5   -0.000000e+00
6   -0.000000e+00
7   -0.000000e+00
8   -0.000000e+00
9   -0.000000e+00
dtype: float64
Saved:
- results/metrics/ablation_monthly.csv
- results/metrics/ablation_summary.csv
- results/plots/ablation_growth_paths.png
- results/plots/ablation_turnover_paths.png

           variant  mean_turnover  ...  mean_concentration  mean_objective
0         baseline       0.011444  ...            0.041252        0.153207
1       macro_only       0.051252  ...            0.044844        0.149589
3  macro_robust_ml       0.044883  ...            0.043610        0.103501
2     macro_robust       0.051252  ...            0.044801        0.065529

[4 rows x 8 columns]
```






Author: Hissan Omar  
Competition: Strand Global Macro Quant Competition (KCL Quant Society)  
Submission Type: Quantitative Research Document + Codebase  
Language: Python  
