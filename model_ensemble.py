import numpy as np
from helper import eval_exponential
from helper import eval_autoarima
from helper import pred_autoarima
from helper import pred_exponential
from pmdarima import auto_arima
import math
import warnings
#warnings.filterwarnings("ignore")

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = CLOSE.shape[1]
    weights = np.zeros(nMarkets)
    test_period = 200
    print('{} {}'.format(DATE[0], DATE[-1]))
    build = settings['counter'] % 28 == 0

    for i in range(nMarkets):
        curr_market = CLOSE[-test_period:, i]
        train = curr_market[:180]
        test = curr_market[180:]

        print(settings['markets'][i])

        try:
            # when it reaches the end of the period and is the time to build a new model
            # build a new model for the ones that have good performance
            # update the not good performance futurers to None
            if build:
                # AUTO ARIMA evaluate
                arima_accuracy = eval_autoarima(train, test)
                print(round(arima_accuracy,2))

                #triple exponential evaluate
                exponential_accuracy = eval_exponential(train, test)
                print(round(exponential_accuracy, 2))

                # give a weighted result for the prediction
                if arima_accuracy > 0.5 and exponential_accuracy > 0.5:
                    arima_model = auto_arima(np.log(curr_market), error_action='ignore', suppress_warnings=True)
                    model = ['combined', arima_model]

                # use arima's result
                elif arima_accuracy > 0.5:
                    arima_model = auto_arima(np.log(curr_market), error_action='ignore', suppress_warnings=True)
                    model = ['arima', arima_model]

                # use exponential's result
                elif exponential_accuracy > 0.5:
                    model = ['exponential']

                # not using this futures
                else:
                    model = None

                settings['models'][i] = model

            else:
                model = settings['models'][i]
                print('==={}'.format(settings['markets'][i]))

            if model:
                if model[0] == 'combined':
                    arima_model = model[1]
                    arima_pred = pred_autoarima(arima_model, np.log(curr_market), settings['markets'][i])
                    arima_pred = np.exp(arima_pred)
                    exponential_pred = pred_exponential(curr_market, settings['markets'][i])

                    if math.isnan(exponential_pred):
                        exponential_pred = arima_pred

                    weights[i] = (arima_pred + exponential_pred) / 2 - curr_market[-1]

                elif model[0] == 'arima':
                    arima_model = model[1]
                    arima_pred = pred_autoarima(arima_model, np.log(curr_market), settings['markets'][i])
                    weights[i] = np.exp(np.float128(arima_pred)) - curr_market[-1]

                elif model[0] == 'exponential':
                    exponential_pred = pred_exponential(curr_market, settings['markets'][i])
                    weights[i] = exponential_pred - curr_market[-1]

        except:
            print('cash')
            pass

    settings['counter'] += 1
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
    settings['endInSample'] = '20200301'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05
    settings['models'] = [None]*len(settings['markets'])
    settings['counter']=0

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    print ('testing')
    results = quantiacsToolbox.runts(__file__)
