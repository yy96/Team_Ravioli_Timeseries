
### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

def eval_model(predicted, observed):
    pred = (predicted < 0).astype(int)
    obs = (observed < 0).astype(int)
    results = (pred == obs).astype(int)
    accuracy = np.mean(results)
    return accuracy
    
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets=CLOSE.shape[1]
    weights = np.zeros(nMarkets)
    periodLonger=200
    print('{} {}'.format(DATE[0],DATE[-1]))
    build = settings['counter']%14==0
    
    for i in range(1,nMarkets):
        curr_market = CLOSE[-periodLonger:,i]
        train = curr_market[:180]
        test = curr_market[180:]
        pred, model=None, None
        
        if build:
         
            train_model = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='add').fit(use_boxcox=True)
            predicted = np.diff(train_model.forecast(20))
            accuracy = eval_model(predicted, np.diff(test))
            print(round(accuracy,2))
            if accuracy > 0.5:model = train_model
            settings['models'][i] = model
        else:
            model = settings['models'][i]
            
        if model:
            pred = model.forecast(1)[0]
            a,b,c= model.params['smoothing_level'], model.params['smoothing_slope'], model.params['smoothing_seasonal']
            print('==={}:{} {} {}'.format(settings['markets'][i], round(a,2),round(b,2),round(c,2)))
            weights[i] = np.log(pred/curr_market[-1])
            
    settings['counter']+=1
    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}

    # Futures Contracts
    settings['markets']  = ['CASH','F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL',
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
