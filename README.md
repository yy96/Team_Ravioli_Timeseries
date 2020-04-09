# Time Series Analysis for Futures Investment
The project aims to develop a trading strategy that maximizes Sharpe ratio when investing in the 88 futures securities on Quantics platform. Sharpe ratio measures risk-adjusted return. It is determined by the ratio of average return to the average volatility of the returns. 

### [Majority Voting](https://github.com/yy96/Team_Ravioli_Timeseries/blob/master/majority_voting.py)

```sh
python majority_voting.py
```
It uses 3 technical indicators (`Moving Average Convergence Divergence`,`Relative Strength Index` and `BollingerBand`) to measure the market conditions and make an investment decision based on the majority sentiment reflected by the indicators.

### ARIMA&SARIMA 
It uses time series models ARIMA/SARIMA with various modifications to predict the close price.
modifications including:
- [pure ARIMA/SARIMA model](https://github.com/yy96/Team_Ravioli_Timeseries/blob/master/ARIMA_pure.py)
- [ARIMA/SARIMA model with model evaluation](https://github.com/yy96/Team_Ravioli_Timeseries/blob/master/ARIMA_with_eval.py)
- [ARIMA/SARIMA model with momentum indicator](https://github.com/yy96/Team_Ravioli_Timeseries/commit/64a94c471e1253f76c96cb14be0ca67105eb74fd)

### Holt-Winters 
[here](link here)
It uses the Holt-Winters model combined with clustering when deciding the portfolio allocation ratios.
