import numpy as np
import numpy
from pmdarima import auto_arima

def autoarima(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
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
