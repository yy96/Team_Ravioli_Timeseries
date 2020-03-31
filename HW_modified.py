import quantiacsToolbox as qt
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.cluster import AgglomerativeClustering as agg
from scipy.spatial.distance import cosine
import warnings,dtw
warnings.filterwarnings("ignore")

markets = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT',
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

start,end = '20180119', '20200331'
interval, period = 14,14
trend, seas = 'add', 'add'
threshold = 0.5

def eval_model(predicted, observed):
    pred = (predicted < 0).astype(int)
    obs = (observed < 0).astype(int)
    results = (pred == obs).astype(int)
    accuracy = np.mean(results)
    return accuracy
    
def myTradingSystem(DATE, CLOSE, settings):
    nMarkets=CLOSE.shape[1]
    weights = np.zeros(nMarkets)
    periodLonger=200
    print('\n{} {}'.format(DATE[0],DATE[-1]))
    build = settings['counter']%interval==0
    num = settings['counter']%interval
    filtered=0
    settings['counter']+=1
    
    for i in range(1,nMarkets):
        curr_market = CLOSE[-periodLonger:,i]
        train = curr_market[:180]
        test = curr_market[180:]
        pred, model=None, None
        
        if build:
            train_model = ExponentialSmoothing(train, seasonal_periods=period, trend=trend, seasonal=seas).fit(use_boxcox=True)
            predicted = np.diff(train_model.forecast(20))
            accuracy = eval_model(predicted, np.diff(test))
            accuracy = round(accuracy,2)
            if accuracy > threshold:
                model = ExponentialSmoothing(curr_market, seasonal_periods=period, 
                                             trend=trend, seasonal=seas).fit(use_boxcox=True)
            else: 
                filtered+=1
                print('{}==={}: inaccurate'.format(round(accuracy,2), settings['markets'][i]))
            settings['models'][i] = model
        else:
            model = settings['models'][i]
            accuracy = ''
            
        if model:
            pred = model.forecast(num+1)[-1]        
            if not np.isnan(pred):
                print('{}==={}'.format(accuracy, settings['markets'][i]))
                weights[i] = np.log(pred/curr_market[-1])  
            else: 
                filtered+=1
                print('{}==={}: no pred'.format(accuracy, settings['markets'][i]))
    if build: print(r'{}/{} markets filtered'.format(filtered, nMarkets-1))     
    return weights, settings


def mySettings():
    settings= {}
    settings['markets']  = ['CASH']+markets
    settings['beginInSample'] = start
    settings['endInSample'] = end
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
