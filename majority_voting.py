import numpy as np
import pandas as pd
from techAnalysis import BollingerBand, RSI, MACD
from techAnalysis import MACD_strat, RSI_strat, BB_strat
import quantiacsToolbox

def mainSettings():
    ''' Define your trading system settings here '''

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

    settings['beginInSample'] = '20180119'
    settings['endInSample'] = '20200331'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    settings['bb_period'] = 20              # sma period for bb
    settings['stoch_period'] = 14
    settings['rsi_period'] = 14
    settings['stoch_period'] = 14
    settings['MACD_n_slow'] = 26
    settings['MACD_n_fast'] = 12

    settings['macd_indicator'] = []
    settings['rsi_indicator'] = []
    settings['bb_indicator'] = []

    return settings



class main(object):
    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
        ''' This system uses trend following techniques to allocate capital into the desired equities'''
        nMarkets = len(settings['markets'])

        macd_indicator = settings['macd_indicator']
        rsi_indicator = settings['rsi_indicator']
        bb_indicator = settings['bb_indicator']

        overall_pos = np.zeros(nMarkets)

        for i in range(1, nMarkets):
            if macd_indicator[i] == 1:
                val = MACD(pd.Series(CLOSE[:, i]), settings)
                overall_pos[i] += val
            
            if rsi_indicator[i] == 1:
                val = RSI(pd.Series(CLOSE[:, i]), settings)
                overall_pos[i] += val
            
            if bb_indicator[i] == 1:
                val = BollingerBand(pd.Series(CLOSE[:, i]), settings)
                overall_pos[i] += val

        return overall_pos, settings

    def mySettings(self):
        return settings


    
if __name__ == '__main__':
    settings = mainSettings()
    nMarkets = len(settings['markets'])
    macd_indicator = np.zeros(nMarkets)
    rsi_indicator = np.zeros(nMarkets)
    bb_indicator = np.zeros(nMarkets)

    res_MACD = quantiacsToolbox.runts(MACD_strat, plotEquity = False)
    mktEquity_MACD = np.array(res_MACD['marketEquity'])

    res_RSI = quantiacsToolbox.runts(RSI_strat, plotEquity = False)
    mktEquity_RSI = np.array(res_RSI['marketEquity'])

    res_BB = quantiacsToolbox.runts(BB_strat, plotEquity = False)
    mktEquity_BB = np.array(res_BB['marketEquity'])

    for i in range(1, nMarkets):
        try:
            stat_MACD = quantiacsToolbox.stats(mktEquity_MACD[-56:, i])
            if stat_MACD['sharpe'] > 0:
                macd_indicator[i] = 1

            stat_RSI = quantiacsToolbox.stats(mktEquity_RSI[-56:, i])
            if stat_RSI['sharpe'] > 0:
                rsi_indicator[i] = 1

            stat_BB = quantiacsToolbox.stats(mktEquity_BB[-56:, i])
            if stat_BB['sharpe'] > 0:
                bb_indicator[i] = 1
        except:
            pass

    settings['macd_indicator'] = macd_indicator
    settings['rsi_indicator'] = rsi_indicator
    settings['bb_indicator'] = bb_indicator


    results = quantiacsToolbox.runts(main, plotEquity = True)


    
    
    
    
    
    
