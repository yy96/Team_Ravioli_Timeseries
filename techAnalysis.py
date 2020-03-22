import numpy as np
import matplotlib.pyplot as plt
import ta
import pandas as pd

'''
Indicators chosen should be uncorrelated

Technical Analysis attemps to understand the market sentiment behind price trends by looking for patterns and trends
rather than analyzing securities fundamental attributes

Forecast price movement
'''
# TREND INDICATOR
## Moving Average Convergence Divergence (MACD)
'''
Use of Exponential Smoothing Average (EMA)
- MACD line calculated by subtracting the 26 period EMA from the 12 period EMA
- The 9-day EMA of the MCAD line is called the signal line
- buy security when MACD cross above its signal line
- sell when AMCD is below the signal line
'''
def MACD(CLOSE, settings):
    nMarkets = len(settings['markets'])
    pos = np.zeros(nMarkets)
    n_slow = settings['MACD_n_slow']
    n_fast = settings['MACD_n_fast']

    for market in range(nMarkets):
        close = pd.Series(CLOSE[:, market])

        macd = ta.trend.MACD(close = close, n_slow = n_slow, n_fast = n_fast)
        val = macd.macd_diff().iloc[-1]
        print(val)
        if val > 0:
            val = 1
        elif val < 0:
            val = -1

        pos[market] = val
    return pos




# MOMENTUM INDICATOR
## Stochastic Oscillator (STOCH)
'''
Predicted on the assumption that closing prices should close near the same direction
as the current trend
'''
def stochOsc(HIGH, LOW, CLOSE, settings):
    nMarkets = len(settings['markets'])
    pos = np.zeros(nMarkets)
    period = settings['stoch_period']

    for market in range(nMarkets):
        high = pd.Series(HIGH[:, market])
        low = pd.Series(LOW[:, market])
        close = pd.Series(CLOSE[:, market])

        so = ta.momentum.StochasticOscillator(high, low, close, n = period)
        val = so.stoch_signal().iloc[-1]

        last = 0
        if val > 80:
            last = -1
        elif val < 20:
            last = 1
        pos[market] = last
    return pos



## Relative Strength Index (RSI)
'''
Tracks oversold and overbought elvels by measuring the velocity of price movements
Buy when oversold (rsi < 30)
Sell when overbought (rsi > 70)
'''

def RSI(CLOSE, settings):
    nMarkets = len(settings['markets'])
    period = settings['rsi_period']
    pos = np.zeros(nMarkets)

    for market in range(nMarkets):
        close = pd.Series(CLOSE[:, market])
        RSI = ta.momentum.RSIIndicator(close, n=period)
        # print('FOR MARKET: ', settings['markets'][market])
        # print(RSI.rsi())
        rsi = RSI.rsi().iloc[-1]

        if rsi < 30:
            pos[market] = 1

        if rsi > 70:
            pos[market] = -1
    return pos

# VOLATILITY INDICATOR
## Bollinger Band - typical to use 20 days SMA
'''
closer the prices move to the upper band, means the mover overbought the market
closer prices move to lower band, more oversold the market
(the squeeze) - when bands come close togehter, low volatility, this is believed to follow with future increased volatility
(wider bands) - beleive to decrease in voltility

-ve: as use SMA, it weighs older price data the same as the most recent ones

@return positions determined by bollingerBand (volatility)
'''
def bollingerBands(CLOSE, settings):
    nMarkets = len(settings['markets'])
    pos = np.zeros(nMarkets)
    period = settings['bb_period']

    for market in range(nMarkets):
        close = pd.Series(CLOSE[:, market])
        bb = ta.volatility.BollingerBands(close = close, n = period)
        bb_high_indicator = bb.bollinger_hband_indicator()
        bb_low_indicator = bb.bollinger_lband_indicator()
        last = 0
        if bb_high_indicator.iloc[-1] == 1:
            last = -1
        elif bb_low_indicator.iloc[-1] == -1:
            last = 1

        pos[market] = last
    return pos


# VOLUME INDICATOR
## On-Balance Volume (OBV)
'''
believe that when volume increases sharply without significant changes in rpice
then the price will eventually jump upward or fall

summary of volume in an uptrend against a down trend

if a market closes high on the prior days close, the volume is added to the indicator
--> every up day, the volume is added to the indicator
--> if it is a down day, the volume is subtracted

** just a figure that show you when you are in an uptrend, volume keeps flowing in
** if downtrend, then the volume will be going down

** try to capture the liquidity

** the number is not relevant (as different securities are bound to have different volume)
** it is the differences in volumes on up days and down days that matters

more volumes on up days, will get a OBV a drastic increase
if there are 5 down days, but not a lot things happening, then the OBV won't be going down too much (although the prices has come down)

we use OBV to confirm the trend

volume comes in when the strength is strong and volume is strong
--> use the volume to confirm up trend
--> OR LOOK FOR DIVERGENCES
E.G.when prices go to a new high but the volume doesn't go to a new high,
then we know nto so many buyers are coming into the market although the value
'''
def OBV(VOLUME, CLOSE, settings):
    nMarkets = len(settings['markets'])
    pos = np.zeros(nMarkets)

    for market in range(nMarkets):
        close = pd.Series(CLOSE[:, market])
        volume = pd.Series(VOLUME[:, market])
        OBV_ind = ta.volume.OnBalanceVolumeIndicator(close, volume)

        obv = OBV_ind.on_balance_volume()
    return pos
