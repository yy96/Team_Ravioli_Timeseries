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

# settings
def sub_test_settings():
    settings= {}

    settings['markets'] = ['CASH'] + ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT',
           'F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG',
           'F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP',
           'F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB',
           'F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY',
           'F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB',
           'F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE',
           'F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB',
           'F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ',
           'F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH',
           'F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']

    # note here the dates set to be the conduct a backtest within the training dates first before the actual backtest!!!
    settings['beginInSample'] = '20171030'
    settings['endInSample'] = '20191231'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05
    
    settings['bb_period'] = 20            
    settings['stoch_period'] = 14
    settings['rsi_period'] = 14
    settings['stoch_period'] = 14
    settings['MACD_n_slow'] = 26
    settings['MACD_n_fast'] = 12

    return settings

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
    n_slow = settings['MACD_n_slow']
    n_fast = settings['MACD_n_fast']
    
    macd = ta.trend.MACD(close = CLOSE, n_slow = n_slow, n_fast = n_fast)
    val = macd.macd_diff().iloc[-1]
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

class MACD_strat(object):

    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, settings):
        nMarkets = len(settings['markets'])
        macd_pos = np.zeros(nMarkets)

        for i in range(1, nMarkets):
            val = MACD(pd.Series(CLOSE[:, i]), settings)
            macd_pos[i] = val
            
        return macd_pos, settings
    
    def mySettings(self):
        return sub_test_settings()




# MOMENTUM INDICATOR
## Relative Strength Index (RSI) 
'''
Tracks oversold and overbought elvels by measuring the velocity of price movements
Buy when oversold (rsi < 30)
Sell when overbought (rsi > 70)
'''

def RSI(CLOSE, settings):
    period = settings['rsi_period']
    
    RSI = ta.momentum.RSIIndicator(CLOSE, n = period)
    rsi = RSI.rsi().iloc[-1]
    if rsi < 30:
        return 1
    elif rsi > 70:
        return -1
    else:
        return 0

class RSI_strat(object):

    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, settings):
        nMarkets = len(settings['markets'])
        rsi_pos = np.zeros(nMarkets)

        for i in range(1, nMarkets):
            val = RSI(pd.Series(CLOSE[:, i]), settings)
            rsi_pos[i] = val
            
        return rsi_pos, settings
    
    def mySettings(self):
        return sub_test_settings()




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

def BollingerBand(CLOSE, settings):
    period = settings['bb_period']
    bb = ta.volatility.BollingerBands(close = CLOSE, n = period)
    bb_high_indicator = bb.bollinger_hband_indicator()
    bb_low_indicator = bb.bollinger_lband_indicator()
    if bb_high_indicator.iloc[-1] == 1:
        return -1
    elif bb_low_indicator.iloc[-1] == -1:
        return 1
    else:
        return 0

class BB_strat(object):

    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, settings):
        nMarkets = len(settings['markets'])
        bb_pos = np.zeros(nMarkets)

        for i in range(1, nMarkets):
            try:
                val = BollingerBand(pd.Series(CLOSE[:, i]), settings)
                bb_pos[i] = val
            except:
                pass
            
        return bb_pos, settings
    
    def mySettings(self):
        return sub_test_settings()
