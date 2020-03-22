### This file contains multiple trading system (trend following, mean reversion etc)
## could also contain the majority voting results

# import necessary Packages below:
import numpy
import pandas as pd
import techAnalysis as techanalysis
#from pairsTrade import corrMatrix


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''
    nMarkets=CLOSE.shape[1]

    #pos_B = bollingerBands(CLOSE, settings)
    #pos = pos_B
    #pos_S = stochOsc(HIGH, LOW, CLOSE, settings)
    pos_RSI = techanalysis.RSI(CLOSE, settings)
    total = pos_RSI

    #pos_MACD = MACD(CLOSE, settings)
    #total = pos_MACD

    #total = numpy.add(pos_B, pos_S, pos_RSI)
    #total = numpy.add(pos_B, pos_RSI, pos_MACD)
    #total = numpy.add(pos_B, pos_MACD)

    lg = total > 0
    sh = total < 0
    pos = numpy.zeros(nMarkets)
    pos[lg] = 1
    pos[sh] = -1
    weights = pos/numpy.nansum(abs(pos))
    # weights = compute_weights(pos, settings)

    return weights, settings

def compute_weights(pos, settings):
    '''
    50% Bond
    30% Equities
    5% Hedging
    5% Commodities
    10% Cash
    '''
    #budget = settings['budget']

    total_exposure = numpy.sum(abs(pos))
    # Cash 1
    pos[0] = 0.1 * total_exposure

    # Currency 15
    currency = 0.05 * total_exposure
    pos[1:16] = (pos[1:16]/numpy.sum(pos[1:16])) * currency

    # Index 23
    index = 0.3 * total_exposure
    pos[16:39] = (pos[16:39]/numpy.sum(pos[16:39])) * index

    # Bond 10
    bond = 0.5 * total_exposure
    pos[39:49] = (pos[39:49]/numpy.sum(pos[39:49])) * bond

    # Commodity 32
    comm = 0.05 * total_exposure
    pos[49: 81] = (pos[49:81]/numpy.sum(pos[49:81])) * comm

    return pos




def mySettings():
    ''' Define your trading system settings here '''

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

    settings['markets'] = cash + currency + interest + index + bond + agriculture + metal + energy
    #settings['markets'] =  agriculture + energy + metal
    #settings['markets'] =  bond
    #settings['markets'] =  index
    #settings['markets'] =  currency
    #settings['markets'] = interest


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

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    # results = quantiacsToolbox.runts(__file__, plotEquity = False)
    results = quantiacsToolbox.runts(__file__)
