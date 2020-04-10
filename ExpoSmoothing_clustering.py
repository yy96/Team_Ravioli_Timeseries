#========== Loading packages ==========# 
import quantiacsToolbox as qt
import numpy as np
from sklearn.cluster import AgglomerativeClustering as agg
from scipy.spatial.distance import cosine
from itertools import groupby
from helper import eval_exponential, validate, check
import warnings,os
warnings.filterwarnings("ignore")



#========== HoltWinters parameters ==========#
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

start,end = '20180119','20200331'
#start,end = '20171030', '20191231'
threshold = 0.6 ### accuracy threshold
interval = 10 ### refit interval
period = 5 ### assume markets has weekly seasonality
test_period = interval ###used to calculate model prediction accuracy in training data



#========== Cluster parameters ==========#
cluster_on = True
smooth_alpha=5 ### take average every 5 datapoints (weekly average) to reduce variations when comparing vector distance which affects clustering precision
num_cluster= 5 ### setting this too large might lead to one-member cluster, which is bad as we will be investing too much on this one market
cluster_period=500 ### setting this longer gives clustering algo more information and thus better clusters
dist_func = cosine ### can apply other distance functions such as correlation; mahattan distance, L1, L2 norm etc



#========== Helper functions: Dynamic lookback ==========#
def set_train_period(CLOSE, t_dict, a_dict):
    """
    helper function that is called only once at the start to search for lookback
    period that has highest accuracy in predicting direction of changefor each market.
    In the subsequent days, call adjust_period() instead to tune the lookback period based on previous lookback period
    """
    
    nMarkets=CLOSE.shape[1]
    for future_id in range(1, nMarkets):
        max_acc, train_period = 0, 0
        cur = np.array([k for k,g in groupby(CLOSE[:,future_id])])
        for i in range(50,501,50):
            accuracy, temp_model = check(cur, i,period, test_period)
            if accuracy > max_acc:
                max_acc = accuracy
                train_period = i
        if max_acc==0:train_period=0
        print('{}:({},{}%)'.format(markets[future_id-1], train_period, int(max_acc*100)))    
        t_dict[future_id] = train_period
        a_dict[future_id] = max_acc
    return t_dict, a_dict


def adjust_period(full_data, prev_period):
    """
    helper function to adjust the lookback period slightly around its previous lookback period
    called everytime the model is retrained 
    """
    
    if not prev_period: prev_period=100
    max_acc, best_period, model = 0, prev_period, None


    for i in [-30,0,30]:
        new_period = prev_period+i
        if new_period<50 or new_period>500: new_period=250

        try:
            accuracy, temp_model = check(full_data, new_period, period,test_period)
        except:
            continue
        
        if accuracy>max_acc:
            best_period = new_period
            model = temp_model
            max_acc = accuracy

    print("{}-->{}".format(prev_period,best_period, period))
    return best_period, model, round(max_acc,2)



#========== Helper functions: Clustering ==========#
def cluster(M, num_cluster, acc_list, dist_func, smooth_alpha):
    """
    @param M: the matrix of data points to be clustered. Each column is a market
    @param num_cluster: user-defined
    @param acc_list: list of accuracies with ith entry being the model accuracy for ith market's model; this affects 
                     each clusters' weightage. Cluster with higher average cluster will have a higher weightage
    @param dist_func: func(x,y) returns the distance between x and y
    @param smooth_alpha: smoothen each markets to make the clustering easier. When smooth_alpha=5,
                         by taking weekly average price instead of using daily price direcly 
    """
    M = normalize(smooth(M, smooth_alpha))
    D = compute_dist_matrix(M, dist_func)
    clusters = agg(n_clusters=num_cluster, affinity='precomputed', linkage = 'average').fit_predict(D)
    weightage = np.ones(num_cluster)
    for i in range(num_cluster):
        weightage[i] = np.mean(acc_list[clusters==i])
    weightage = weightage/np.sum(weightage)
    return clusters, weightage


def smooth(M, interval):
    """helper function called by cluster() to smooth a vector"""
    smoothed = np.mean(M[:interval,],axis=0)
    for i in range(1, int(M.shape[0]/interval)):
        start = i*interval
        new = np.mean(M[start:start+interval,],axis=0)
        smoothed = np.vstack((smoothed,new))
    return smoothed


def normalize(included_M):
    """helper function called by cluster() to normalize a vector"""
    m = np.mean(included_M, axis=0)
    se = np.std(included_M, axis=0)
    return (included_M-m)/se


def compute_dist_matrix(M, dist_func):
    """
    helper function called by cluster() to calculate pairwise distance among columns in M
    @return: a matrix where [i,j] entry is the distance between ith and jth markets' vectors in M
    """
    n_markets =  M.shape[1]
    dist_mat = np.zeros((n_markets,n_markets))
    for i in range(n_markets):
        for j in range(i+1, n_markets):
            dist = dist_func(M[:,i], M[:,j])
            dist_mat[i,j], dist_mat[j,i] = dist, dist
    return dist_mat


def group_norm(num_cluster, included, cluster, weights, show,weightage):
    """
    normalize the weights within each cluster such that
    the sum of weights of markets from ith cluster = weightage for ith cluster
    the sum of weightages of all clusters = 1
    """
    included = np.array(included) 
    for i in range(num_cluster):
        w = weightage[i]
        indices = included[cluster==i]
        cur_cluster = weights[indices]
        norm_cluster = (cur_cluster*w)/(np.nansum(np.abs(cur_cluster)))
        weights[indices] = norm_cluster
        names = np.array(markets)[indices-1]
        if show: print("\ncluster {}({}): {}\n".format(i, round(weightage[i],2),names))
    weightage = weightage/np.sum(weightage)
    return weights



#========== Trading System ==========#
def myTradingSystem(DATE, CLOSE, settings):
    ### 0.SET UP
    # num and build: counter used to retrain the model by interval
    # settings['train_period']: record individual market's lookback period found in the last retrain: at the start
    # settings['accuracies']: record the accuracy achieved in the last retain using the recorded lookback period and test period for each market
    # filtered: counter of how many markets are removed due to inaccurate model or non-convergent problem
    num = settings['counter']%interval
    build = num==0
    nMarkets=CLOSE.shape[1]
    weights = np.zeros(nMarkets)
    filtered=0
    date = DATE[-1]
    settings['counter']+=1
    if settings['counter']==1:###first day set up the lookback by search from 50~500
        periods, accuracies = set_train_period(CLOSE, settings['train_period'], settings['accuracies'])
        settings['train_period'] = periods
        settings['accuracies'] = accuracies
    

    ### 1. Data preprocessing: from raw data to input 
    # CLOSE: log the prices to 1)linearize the trend 2)reduce the impact of different price scale of different markets
    # CLOSE: remove consecutive duplicates as that will cause the model to produce nan values when it detects constant pattern
    print('\n{}'.format(date))
    for i in range(1,nMarkets):        
        cur = np.log(np.array([k for k,g in groupby(CLOSE[:,i])]))
        pred, model=None, None

        ### 2. Model building: process input to produce predictions
        # prev_period: the last lookback period used for this market in the last retrain; might be ajusted slightly if another number can achieve better performance
        # train_period, model, accuracy: the adjusted lookback period, model trained using the new period and the accuracy for the retrained model in the validation data
        if build:
            prev_period = settings['train_period'][i]
            train_period, model, accuracy = adjust_period(cur, prev_period)

            ### 3. Model filtering
            # threshold: if the accuracy is less than the threshold, exclude the market in the final output to avoid loss
            # settings['models']: if the model achieves acceptable accuracy, save it for next day prediction until next retrain
            if accuracy<threshold:
                model=None
                print(' {}%==={}: inaccurate'.format(int(100*accuracy), settings['markets'][i]))
            settings['models'][i] = model
            settings['train_period'][i] = train_period
            settings['accuracies'][i] = accuracy
        else:
            model = settings['models'][i]
            accuracy = settings['models'][i]
            accuracy= settings['accuracies'][i]
                    
        if model:
            ### 4. Prediction and equity allocation
            # pred: the HW model will predict the next log(price)
            # weights: take the predicted log(price) and minus the latest observed log(price) as the final output weights
            pred = validate(cur[-1],model.forecast(num+1)[-1])
            if not np.isnan(pred):
                print(' {}%==={}'.format(int(accuracy*100), settings['markets'][i]))
                weights[i] = pred-cur[-1]
            else: 
                print(' {}%==={}: non-convergent'.format(int(accuracy*100), settings['markets'][i]))

    acc = settings['accuracies']
    included = np.nonzero(weights)[0]
    settings['num_trades'].append(included.size)
    print("(average daily trades/ total markets): ({}/{})".format(round(np.mean(settings['num_trades']),2),nMarkets-1))

    if cluster_on and num_cluster>1:
        clusters, weightage = cluster(CLOSE[-cluster_period:,included], num_cluster, np.array(acc)[included], dist_func, smooth_alpha)
        weights = group_norm(num_cluster, included, clusters, np.array(weights), build, weightage)           
        
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
    settings['train_period'] = [0]*len(settings['markets'])
    settings['accuracies'] = [0]*len(settings['markets'])
    settings['num_trades']=[]
    settings['counter']=0
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
