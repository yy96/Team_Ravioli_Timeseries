import quantiacsToolbox as qt
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing as eps
from itertools import groupby
from helper import eval_exponential
import warnings
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

start,end = '20180119', '20200430'
threshold = 0.5 
interval, period = 7,7#77;714;730;
test_period = interval

def set_train_period(CLOSE, t_dict, a_dict):
    nMarkets=CLOSE.shape[1]
    for future_id in range(1, nMarkets):
        max_acc, train_period = 0, 0
        cur = np.array([k for k,g in groupby(CLOSE[:,future_id])])
        for i in range(100,500,90):
            accuracy = check(cur, i)
            if accuracy > max_acc:
                max_acc = accuracy
                train_period = i
        if max_acc==0:train_period=0
        print('{}:({},{}%)'.format(markets[future_id-1], train_period, int(max_acc*100)))    
        t_dict[future_id] = train_period
        a_dict[future_id] = max_acc
    return t_dict, a_dict

def check(data, lookback):
    accuracy=0
    temp_model = eps(data[-lookback:], seasonal_periods=period, trend="add", seasonal="add").fit(use_boxcox=True)
    if np.all(~np.isnan(temp_model.forecast(test_period))): ## pass nan test
        train = data[-lookback-test_period-3:]
        accuracy = eval_exponential(train, lookback, test_period, period)
    return accuracy 

def adjust_period(full_data, prev_acc, prev_period, test_period):
    if not prev_period: prev_period=100
    max_acc, lookback, model = prev_acc, prev_period, eps(full_data[-prev_period:], seasonal_periods=period, 
                                                          trend="add", seasonal="add").fit(use_boxcox=True)
    for i in [-60,-30,30,60]:
        try:
            new_period = prev_period+i
            if new_period<50: new_period=200
            elif new_period>500: new_period=350
            accuracy = check(full_data, new_period)
            if accuracy>max_acc:
                lookback = new_period
                model = temp_model
                max_acc = accuracy
        except:
            pass
    print("{}-->{}".format(prev_period,lookback))
    return lookback, model, round(max_acc,2)


def myTradingSystem(DATE, CLOSE, settings):
    num = settings['counter']%interval
    build = num==0
    settings['counter']+=1
    if settings['counter']==1:
        settings['train_period'], settings['accuracies'] = set_train_period(CLOSE, settings['train_period'], 
                                                                            settings['accuracies'])
    nMarkets=CLOSE.shape[1]
    weights = np.zeros(nMarkets)
    filtered=0
    CLOSE = np.log(CLOSE)
    print('\n{}'.format(DATE[-1]))
    
    for i in range(1,nMarkets):        
        cur = np.array([k for k,g in groupby(CLOSE[:,i])])
        pred, model=None, None
        
        if build:
            prev_period = settings['train_period'][i]
            prev_acc = settings['accuracies'][i]
            train_period, model, accuracy = adjust_period(cur, prev_acc, prev_period, test_period)                
            if accuracy<threshold:
                model=None
                filtered+=1
                print('   {}%==={}: inaccurate'.format(int(100*accuracy), settings['markets'][i]))
            settings['models'][i] = model
            settings['train_period'][i] = train_period
        else:
            model = settings['models'][i]
            accuracy=0
                    
        if model:
            pred = model.forecast(num+1)[-1]        
            if not np.isnan(pred):
                print(' {}%==={}'.format(int(accuracy*100), settings['markets'][i]))
                weights[i] = pred-cur[-1]
            else: 
                filtered+=1
                print('   {}%==={}: non-convergent'.format(int(accuracy*100), settings['markets'][i]))
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
    settings['train_period'] = [None]*len(settings['markets'])
    settings['accuracies'] = [0]*len(settings['markets'])
    settings['counter']=0
    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
