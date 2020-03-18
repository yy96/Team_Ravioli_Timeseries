### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
import numpy as np
import numpy
from pmdarima import auto_arima

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
    nMarkets=CLOSE.shape[1]
    log_diff = np.diff(np.log(CLOSE),axis=0)
    weights = np.zeros(nMarkets)
    periodLonger=200
    print('{} {}'.format(DATE[0],DATE[-1]))
    build = settings['counter']%28==0
    
    for i in range(1,nMarkets):
        cur_market = log_diff[-periodLonger:,i]
        try:
            if build:
                model = auto_arima(cur_market, error_action='ignore', suppress_warnings=True, seasonal=True, m=4)#sarima
                #model = auto_arima(cur_market, error_action='ignore', suppress_warnings=True)#arima
                params = np.array(model.params())
                if np.any(abs(params)>0.001):
                    settings['models'][i] = model
                    print('==={}:{}'.format(settings['markets'][i], params))
                else:
                    settings['models'][i] = None
            else:
                model = settings['models'][i]
                if model:
                    print('==={}'.format(settings['markets'][i]))
                                
            if model:
                model.fit(cur_market)
                pred = model.predict(n_periods=1)[0]
              
            if pred:
                weights[i] = pred/np.var(cur_market)
                #if pred>0: weights[i]=1
                #else: weights[i]=-1
            else: settings['prev'][i]=0
                
        except:
            pass
        
                
    settings['counter']+=1
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
    settings['models'] = [None]*len(settings['markets'])
    settings['counter']=0

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
