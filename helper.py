import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.api import ExponentialSmoothing as eps

# evaluate the performace of the model based on price signal
def eval_model(predicted, observed):
    return np.mean((predicted*observed >0).astype(int))

def eval_autoarima(train, test):
    # sarima_train_model = auto_arima(curr_market, error_action='ignore', suppress_warnings=True, seasonal=True, m=4)
    train = np.log(train)
    test = np.log(test)
    arima_train_model = auto_arima(train, error_action='ignore', suppress_warnings=True)  # arima
    arima_train_model.fit(train)
    arima_predicted = np.diff(arima_train_model.predict(n_periods=20))
    arima_accuracy = eval_model(arima_predicted, np.diff(test))

    return arima_accuracy

def eval_exponential(curr_market, train_period, test_period, period):
    acc_list = []
    for i in range(3):
        last_train = i+train_period
        train = curr_market[i:last_train]
        test = curr_market[last_train:last_train+test_period]
        model = eps(train, seasonal_periods=period, trend="add",seasonal="add").fit(use_boxcox=True)
        pred = np.diff(model.forecast(len(test)))
        acc_list.append(eval_model(pred, np.diff(test)))
    return sum(acc_list)/len(acc_list)

def pred_autoarima(arima_model, curr_market, market_name):
    # sarima_model = auto_arima(curr_market, error_action='ignore', suppress_warnings=True, seasonal=True, m=4)
    arima_model.fit(curr_market)
    print('==={}:{} {}'.format(market_name, arima_model.params(), arima_model.order))
    arima_pred = arima_model.predict(n_periods=1)[0]

    return arima_pred

def pred_exponential(curr_market, market_name):
    exponential_model = eps(curr_market, seasonal_periods=12, trend='add', seasonal='add').fit(use_boxcox=True)
    exponential_pred = exponential_model.forecast(1)[0]
    a, b, c = exponential_model.params['smoothing_level'], exponential_model.params['smoothing_slope'], \
              exponential_model.params['smoothing_seasonal']
    print('==={}:{} {} {}'.format(market_name, round(a, 2), round(b, 2), round(c, 2)))
    return exponential_pred

def simple_ensemble(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    techanalysis_output, _ = techanalysis(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)
    arima_output, settings = autoarima(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    #weighted_arima = [float(i)/sum(arima_output) for i in arima_output]
    #weighted_techanalysis = [float(i)/sum(techanalysis_output) for i in techanalysis_output]


    weights = np.array([a+b for a,b in zip(arima_output, techanalysis_output)])
    print (techanalysis_output)
    print (weights)
    #weights = [float(i)/sum(weights) for i in weights]

    return weights, settings
