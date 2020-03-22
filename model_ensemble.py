import numpy as np
import numpy
from helper import autoarima
from main import myTradingSystem as techanalysis

markets = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL',
        'F_CT','F_DX','F_EC','F_ES','F_FC','F_FV','F_GC','F_F',
        'F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD',
        'F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL',
        'F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU',
        'F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT',
        'F_UB','F_UZ','F_GS','F_LX','F_DL','F_ZQ','F_VX','F_SS',
        'F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ',
        'F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR',
        'F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY',
        'F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD']


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    techanalysis_output, _ = techanalysis(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)
    arima_output, settings = autoarima(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    #weighted_arima = [float(i)/sum(arima_output) for i in arima_output]
    #weighted_techanalysis = [float(i)/sum(techanalysis_output) for i in techanalysis_output]


    weights = np.array([a+b for a,b in zip(arima_output, techanalysis_output)])
    print (techanalysis_output)
    print (weights)
    #weights = [float(i)/sum(weights) for i in weights]

    return weights, settings

def mySettings():
    ''' Define your trading system settings here '''

    settings= {}

    # Futures Contracts
    settings['markets']  = ['CASH']+markets
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

    settings['models'] = [None]*len(settings['markets'])
    settings['counter']=0

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    print ('testing')
    results = quantiacsToolbox.runts(__file__)
