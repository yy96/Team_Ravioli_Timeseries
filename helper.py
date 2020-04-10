import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.api import ExponentialSmoothing as eps
import os



#========== 1. HW helper functions ==========#
def eval_exponential(curr_market, train_period, test_period, period):
    """
    helper function to calculate the accuracy a model in predicting the right direction of change for the given dataset
    simulate the prediction process using the given training and testing periods;
    @param curr_market: the set of data to be modelled
    @param train_period: number of observed data points to build up exponential smoothing model
    @param test_period: number of forecasts to make using the trained model
    @param period: the seasonality period used to build holt-winters model
    @return accuracy: the averaged accuracy of 3 samples, each using "train_period" number of points to predict next "test_period" number of points
    """
    acc_list = []
    for i in range(3):
        last_train = i+train_period
        train = curr_market[i:last_train]
        test = curr_market[last_train-1:last_train+test_period]
        pred,model = eps_predict(train,period,len(test))
        acc_list.append(eval_model(pred, test))
    return sum(acc_list)/len(acc_list)


def eps_predict(train,period,num_forecasts):
    """
    helper function called by eval_exponential() to build model, generate prediction and validate prediction
    """
    model = eps(train, seasonal_periods=period, trend="add",seasonal="add").fit(use_boxcox=True)
    pred = model.forecast(num_forecasts)
    pred = validate(train[-1],pred)
    return pred,model


def validate(obs,pred):
    """
    helper function to check if the prediction is valid;
    if it is abnormally small/large then scale it to the normal level
    """
    validate = abs(obs/pred)
    x = np.logical_or(validate < 0.5, validate > 1.5).astype(int)
    y = x*(validate.astype(int)-1)
    return pred+y


def check(data, lookback,period,test_period):
    """
    helper function called by set_train_period() and adjust_period()
    @param data: the set of datapoints to be modelled
    @param lookback: the number of datapoints as training data to do multi-step ahead forecasts
    """
    model = eps(data[-lookback:], seasonal_periods=period, trend="add", seasonal="add").fit(use_boxcox=True)
    accuracy=0
    if np.mean(np.isnan(model.forecast(test_period).astype(int)))<0.5:
        train = data[-lookback-test_period-3:]
        accuracy = eval_exponential(train, lookback, test_period, max(int(lookback/30),3))
    return accuracy, model


# evaluate the performace of the model based on price signal
def eval_model(predicted, observed):
    pred = np.diff(predicted)
    obs = np.diff(observed)
    return np.mean((pred*obs>0).astype(int))



#========== 2. ARIMA helper functions ==========#
def eval_autoarima(train, test):
    # sarima_train_model = auto_arima(curr_market, error_action='ignore', suppress_warnings=True, seasonal=True, m=4)
    train = np.log(train)
    test = np.log(test)
    arima_train_model = auto_arima(train, error_action='ignore', suppress_warnings=True)  # arima
    arima_train_model.fit(train)
    arima_predicted = np.diff(arima_train_model.predict(n_periods=20))
    arima_accuracy = eval_model(arima_predicted, np.diff(test))
    return arima_accuracy


def pred_autoarima(arima_model, curr_market, market_name):
    # sarima_model = auto_arima(curr_market, error_action='ignore', suppress_warnings=True, seasonal=True, m=4)
    arima_model.fit(curr_market)
    print('==={}:{} {}'.format(market_name, arima_model.params(), arima_model.order))
    arima_pred = arima_model.predict(n_periods=1)[0]
    return arima_pred



#==========3. Ensemble helper functions ==========#
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



#========== # utility functions ==========#
def read_saved_data(path,date):
    if "{}.txt".format(date) in os.listdir(path):
        weights,acc = read_daily(path,date)
        return np.array(weights),acc
    return None


def save_output(x,y,z):
    cwd = os.getcwd()
    name = "{}_{}_{}".format(x,y,z)
    path = os.path.join(cwd, name)
    if name not in os.listdir(cwd):
        os.mkdir(path)
    return path


def save_initial(path, t_dict,a_dict):
    t_file=open(path+'/periods.txt','w')
    a_file=open(path+'/accuracies.txt','w')
    for t in t_dict:
         t_file.write("{}\n".format(str(t)))
    t_file.close()

    for a in a_dict:
         a_file.write("{}\n".format(str(a)))
    a_file.close()


def save_daily(path,date,weights,acc):
    file=open(path+'/{}.txt'.format(date),'w')
    for i in range(len(weights)):
         file.write("{},{}\n".format(str(weights[i]),str(acc[i])))
    file.close()


def read_daily(path,date):
    t=open(path+'/{}.txt'.format(date,'r')).readlines()
    weights, acc=[],[]
    for each in t:
        out = each.split(',')
        weights.append(float(out[0]))
        acc.append(float(out[1]))
    return weights,acc


def read_setup(path):
    t=open(path+'/periods.txt','r').readlines()
    a=open(path+'/accuracies.txt','r').readlines()
    t_dict = [int(i) for i in t]
    a_dict = [float(i) for i in a]
    return t_dict,a_dict
