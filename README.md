# Team_Ravioli_Timeseries
The project aims to develop a trading strategy that maximizes Sharpe ratio when investing in the 88 futures securities on Quantics platform. Sharpe ratio measures risk-adjusted return. It is determined by the ratio of average return to the average volatility of the returns. 

### [Majority Voting] (link here)
It uses 3 technical indicators (`Moving Average Convergence Divergence`,`Relative Strength Index` and `BollingerBand`) to measure the market conditions and make an investment decision based on the majority sentiment reflected by the indicators.

### [ARIMA&SARIMA] (link here)
It uses time series models ARIMA/SARIMA with various modifications to predict the close price.

### [Holt Winter] (link here)
It uses the Holt-Winters model combined with clustering when deciding the portfolio allocation ratios.
