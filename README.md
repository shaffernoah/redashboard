# Advanced Zillow Market Analytics Dashboard

A sophisticated real estate market analysis tool built with Streamlit that provides deep insights into housing market dynamics using Zillow's comprehensive dataset.

## Features

### 1. Market Overview Dashboard
Advanced analytics providing deep market insights:

#### Market Momentum Score (0-100)
A composite score incorporating multiple market indicators:
- **Price Appreciation**: Year-over-year price changes weighted against historical trends
- **Inventory Changes**: Impact of supply dynamics
- **Sales Above List**: Measure of market competitiveness
- **Market Velocity**: Rate of price changes in recent months

#### Price-to-Rent (P/R) Analysis
- **P/R Ratio**: A fundamental metric for market valuation
  - High P/R ratio (>20) suggests potentially overvalued market
  - Low P/R ratio (<15) might indicate buying opportunities
- **Investment Score**: Combines P/R ratio with rental growth trends
  - Accounts for historical P/R averages
  - Weights current position against long-term means
  - Incorporates rental growth momentum

#### Market Cycle Analysis
Uses time series decomposition to identify market phases:
- **Expansion**: Rising prices with positive momentum
- **Peak**: High prices with slowing momentum
- **Contraction**: Declining prices
- **Trough**: Low prices with potential turnaround
- **Confidence Score**: Measures reliability of phase identification

Technical details:
- Employs seasonal decomposition (STL method)
- Analyzes trend, seasonal, and residual components
- Uses cubic spline interpolation for missing data
- Applies confidence penalties for data quality issues

#### Market Pressure Indicator (-100 to +100)
Measures supply-demand balance:
- **Negative scores**: Buyer's market conditions
- **Positive scores**: Seller's market conditions
- Incorporates:
  - Supply changes (inventory levels)
  - Demand indicators (price trends)
  - Market absorption rates

#### Market Stability Analysis
Quantifies market risk and volatility:
- **Volatility Metrics**: Annualized price volatility
- **Rolling Measures**: Short-term vs long-term stability
- **Risk Score**: Composite measure of market stability

### 2. Market Analysis Dashboard
Core market metrics and visualizations:
- Median home values
- Rental rates
- Inventory levels
- List prices
- Year-over-year comparisons
- Interactive time series visualizations

### 3. Market Activity Dashboard
Current market conditions:
- Active inventory
- New listings
- Sales metrics
- Price adjustments
- Market velocity indicators

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zillow-market-analytics.git
cd zillow-market-analytics
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Data Sources

The dashboard uses Zillow's public data files:
- Home Value Index (ZHVI)
- Observed Rent Index (ZORI)
- Inventory
- New Listings
- Sales Above List Price

Data should be placed in the root directory with the following files:
- Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
- Metro_zori_uc_sfrcondomfr_sm_month.csv
- Metro_invt_fs_uc_sfrcondo_sm_week.csv
- Metro_mlp_uc_sfrcondo_sm_week.csv
- Metro_new_listings_uc_sfrcondo_week.csv
- Metro_pct_sold_above_list_uc_sfrcondo_week.csv

## Statistical Methods

### Time Series Decomposition
The market cycle analysis uses STL (Seasonal and Trend decomposition using Loess) to separate time series into:
- **Trend**: Long-term price direction
- **Seasonal**: Recurring patterns (e.g., summer peaks)
- **Residual**: Unexpected variations

### Market Momentum
Calculated using exponentially weighted combinations of:
- Price velocity (first derivative)
- Price acceleration (second derivative)
- Volume indicators
- Normalized against historical volatility

### Volatility Metrics
- Uses rolling standard deviation of returns
- Annualized using √12 factor
- Incorporates both price and volume volatility
- Adjusts for market size and liquidity

### Investment Scoring
Combines multiple factors:
- P/R ratio z-score (distance from historical mean)
- Rental growth momentum
- Price trend stability
- Market liquidity factors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

- Data provided by Zillow Research
- Built with Streamlit
- Analysis powered by pandas, numpy, and scipy
