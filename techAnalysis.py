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

    # Futures Contracts
    cash = ['CASH']
    # 15 currency 
    currency = ['F_AD', 'F_BP', 'F_CD', 'F_DX', 'F_EC', 'F_JY', 'F_MP', 'F_SF', 'F_LR', 'F_ND',
                'F_RR', 'F_RF', 'F_RP', 'F_RY', 'F_TR']
    # 5 interset
    interest = ['F_ED', 'F_SS', 'F_ZQ', 'F_EB', 'F_F']
    # 23 index
    index = ['F_ES', 'F_MD', 'F_NQ', 'F_RU', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_LX', 'F_VX', 
            'F_AE', 'F_DM', 'F_AH', 'F_DZ', 'F_FB', 'F_FM', 'F_FP', 'F_FY', 'F_NY', 'F_PQ', 
            'F_SH', 'F_SX', 'F_GD']
    # 10 bond
    bond = ['F_FV', 'F_TU', 'F_TY', 'F_US', 'F_DT', 'F_UB', 'F_UZ', 'F_GS', 'F_CF', 'F_GX']
    # 17 agri
    agriculture = ['F_BO', 'F_C', 'F_CC', 'F_CT', 'F_FC', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_NR', 
                    'F_O', 'F_OJ', 'F_S', 'F_SB', 'F_SM', 'F_W', 'F_DL']
    # 10 energy
    energy = ['F_CL', 'F_HO', 'F_NG', 'F_RB', 'F_BG', 'F_BC', 'F_LU', 'F_FL', 'F_HP', 'F_LQ']
    # 5 metal
    metal = ['F_GC', 'F_HG', 'F_PA', 'F_PL', 'F_SI']


    '''These 3 futures not listed on the website F_VF, F_VT, F_VW. Hence, only 85 futures above'''

    settings['markets'] = cash + currency + interest + index + bond + agriculture + metal + energy + ['F_VF', 'F_VT', 'F_VW']
    

    # note here the dates set to be the conduct a backtest within the training dates first before the actual backtest!!!
    settings['beginInSample'] = '20171030'
    settings['endInSample'] = '20191231'
    # settings['beginInSample'] = '20180119'
    # settings['endInSample'] = '20200330'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05
    
    settings['bb_period'] = 20              # sma period for bb
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
        macd_pos[0] = 1

        for i in range(1, nMarkets):
            val = MACD(pd.Series(CLOSE[:, i]), settings)
            macd_pos[i] = val
            
        return macd_pos, settings
    
    def mySettings(self):
        return sub_test_settings()




# MOMENTUM INDICATOR
## Stochastic Oscillator (STOCH) // NOT IN USE!
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
        rsi_pos[0] = 1

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
        bb_pos[0] = 1

        for i in range(1, nMarkets):
            try:
                val = BollingerBand(pd.Series(CLOSE[:, i]), settings)
                bb_pos[i] = val
            except:
                pass
            
        return bb_pos, settings
    
    def mySettings(self):
        return sub_test_settings()

    




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