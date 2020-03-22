import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

'''
moving average calculated by adding recent closing prices and then dividing
that by the number of time periods in the calculation average
'''
def SMA(data,number):
    close_price = data['CLOSE']
    p=np.zeros((len(close_price),1))
    for i in range(0,len(close_price)):
        if i <= (number-1):
            q = np.average(close_price[0:i+1])
            p[i]=q
        else:
            q = np.average(close_price[i-(number-1):i+1])
            p[i]=q
    SMA_price = p
    return SMA_price


'''
EMA is a moving average that places a greater weight and significance on the
most recent data points
'''
def EMA(data, number):
    close_price = data['CLOSE']
    p = np.zeros((len(close_price), 1))
    for i in range(0, len(close_price)):
        if i == 0:
            q = close_price[i]
            p[i][0] = q
        else:
            q = p[i - 1][0] * (number - 1) / (number + 1) + close_price[i] * 2 / (number + 1)
            p[i][0] = q
    EMA_price = p
    return EMA_price

def EMA_FUNCTION(data, number):
    p = np.zeros((len(data), 1))
    for i in range(0, len(data)):
        if i == 0:
            q = data[i][0]
            p[i][0] = q
        else:
            q = p[i - 1][0] * (number - 1) / (number + 1) + data[i][0] * 2 / (number + 1)
            p[i][0] = q
    return p

##DEFINE MACD_DIF  实现MACD的DIF线生成，name是股票名称，number1是较短的天数，number2是较长的天数，返回MACD_DIF的数据
def MACD_DIF(name, number1, number2):
    macd = EMA(name,number1)-EMA(name,number2)  ###基于指数移动平均生成，若想使用其他的移动平均法，需要在这里将EMA 换成其他的函数
    MACD_DIF = macd
    return MACD_DIF

##DEFINE MACD_DEA，实现MACD——DEA 线的生成，name是股票名称，number1是DIF线中较短的天数，number2是DIF线中较长的天数，number3是DEA线设的天数
def MACD_DEA(name, number1,number2,number3):
    macd = MACD_DIF(name, number1,number2)
    p =np.zeros((len(macd),1))
    for i in range(0,len(macd)):
        if i ==0:
            q = macd[i]
            p[i]=q
        else:
            q = p[i-1]*(number3-1)/(number3+1) + macd[i]*2/(number3+1)
            p[i] = q
    MACD_DEA = p
    return MACD_DEA
###返回MACD——DEA 的数据

##DEFINE MACD_BAR，实现MACD_BAR指示线，简单计算MACD两条线的差值
def BAR(name, number1,number2,number3):
    bar = MACD_DIF(name, number1,number2)-MACD_DEA(name,number1,number2,number3)
    return bar


'''
The relative strength index (RSI) is a momentum indicator that measures the
magnitude of recent price changes to evaluate overbought or oversold conditions
in the price of a stock or other asset.
'''
def RSI_S(data, number):
    close_price = data['CLOSE']
    diff = np.zeros((len(close_price), 1))
    rsi = np.zeros((len(close_price), 1))
    ##calculate the up and down value
    for i in range(0, len(close_price)):
        if i == 0:
            q = 0
            diff[i][0] = q
        else:
            q = close_price[i] - close_price[i - 1]
            diff[i][0] = q
    ##define the calculation part
    for i in range(0, len(diff)):
        ## define the value for the first position
        if i == 0:
            s = 0
            rsi[i] = s
        ##difine the calculation part for the 1-9 position
        elif i < number:
            up = 0
            down = 0
            for j in range(1, i + 1):
                if diff[j][0] < 0:
                    down = down + (diff[j][0]) * (-1)
                else:
                    up = up + diff[j][0]
            s = up / (up + down)
            s = s * 100
            rsi[i] = s
        ##define the calculation part for the rest position
        else:
            u = 0
            d = 0
            for j in range(i - number + 1, i):
                if diff[j][0] < 0:
                    d = d + (-1) * diff[j][0]
                else:
                    u = u + diff[j][0]
            s = u / (u + d)
            s = s * 100
            rsi[i] = s
    return rsi

def RSI_E(name, number):  ###时间基于指数移动平均的RSI线数据生成，name是股票名称，number是天数，可以通过添加阀值对30-70之外的数据进行判断
    data = read_data(name)
    close = data['Close']
    diff =np.zeros((len(close),1))
    up = np.zeros((len(close),1))
    down = np.zeros((len(close),1))
    ema_d = np.zeros((len(close),1))
    ema_u = np.zeros((len(close),1))
    rsi =np.zeros((len(close),1))
##calculate the up and down value
    for i in range(0,len(close)):
        if i ==0:
            q = 0
            diff[i][0]=q
        else:
            q = close[i]-close[i-1]
            diff[i][0]=q
    for i in range(0,len(diff)):
        if diff[i][0] > 0:
            s = diff[i][0]
            down[i][0] = 0
            up[i][0] = s
        else:
            t = (-1)*diff[i][0]
            up[i][0] = 0
            down[i][0] = t
##define the calculation part
    for i in range(0, len(up)):
## define the value for the first position
        if i ==0:
            a = 0
            ema_d[i][0] = a
            ema_u[i][0] = a
##define the value for the ema_u and ema_d
        else:
            ema_d[i][0] = ema_d[i-1][0]*(number-1)/(number+1) + down[i][0]*(2)/(number+1)
            ema_u[i][0] = ema_u[i-1][0]*(number-1)/(number+1) + up[i][0]*(2)/(number+1)
##define the value for the rsi
    for i in range(0,len(rsi)):
        if i ==0:
            rsi[i][0] = 0
        else:
            r = ema_u[i][0]/(ema_u[i][0]+ema_d[i][0])
            rsi[i][0] = r

    return rsi


##define the bbiboll 实现基于简单移动平均的BBIBOLL数据生成，number1,2,3,4分别为4条移动平均线使用的天数，BBIBOLL就是计算4天移动平均线的平均值
def BBIBOLL(name, number1, number2, number3, number4):
    p1 = SMA(name, number1)
    p2 = SMA(name, number2)
    p3 = SMA(name, number3)
    p4 = SMA(name, number4)
    bbiboll = np.zeros((len(p4), 1))
    for i in range(0, len(p1)):
        s = p1[i][0] + p2[i][0] + p3[i][0] + p4[i][0]
        s = s / 4
        bbiboll[i][0] = s
    return bbiboll


##define the upper line of the BBIBOLL
###实现为BBIBOLL线输出上行线的数据，number1，2,3,4为BBIBOLL的数据，number5为计算标准差的天数，
##number6为上行线与基准数据之差为number6倍标准差
def BBIBOLL_UP(name, number1, number2, number3, number4, number5, number6):
    p = BBIBOLL(name, number1, number2, number3, number4)
    for i in range(0, len(p)):
        if i < number5:
            a = np.mean(p[0:i + 1])
            sum = 0
            for j in range(0, i + 1):
                s = (p[j][0] - a) ** 2
                sum = sum + s
            sum = sum / (i + 1)
            standard = math.sqrt(sum)
            p[i][0] = p[i][0] + number6 * standard
        else:
            a = np.mean(p[(i - number5 + 1): i + 1])
            sum = 0
            for j in range(i - number5 + 1, i + 1):
                s = (p[j][0] - a) ** 2
                sum = sum + s
            sum = sum / (number5)
            standard = math.sqrt(sum)
            p[i][0] = p[i][0] + number6 * standard

    return p
    ###输出为上行线数据


##define the down line of bbiboll
##实现为BBIBOLL线输出下行线的数据，number1，2,3,4为BBIBOLL的数据，number5为计算标准差的天数
###，number6为下行线与基准数据之差为number6倍标准差
def BBIBOLL_DOWN(name, number1, number2, number3, number4, number5, number6):
    p = BBIBOLL(name, number1, number2, number3, number4)
    p = BBIBOLL(name, number1, number2, number3, number4)
    for i in range(0, len(p)):
        if i < number5:
            a = np.mean(p[0:i + 1])
            sum = 0
            for j in range(0, i + 1):
                s = (p[j][0] - a) ** 2
                sum = sum + s
            sum = sum / (i + 1)
            standard = math.sqrt(sum)
            p[i][0] = p[i][0] - number6 * standard
        else:
            a = np.mean(p[(i - number5 + 1): i + 1])
            sum = 0
            for j in range(i - number5 + 1, i + 1):
                s = (p[j][0] - a) ** 2
                sum = sum + s
            sum = sum / (number5)
            standard = math.sqrt(sum)
            p[i][0] = p[i][0] - number6 * standard

    return p
    ###输出为下行线数据


'''
volumn weighted average
'''
def VWAP(data, number):
    close = data['CLOSE']
    volume = data['VOL']
    vwap = np.zeros((len(close), 1))
    for i in range(0, len(close)):
        if i < number:
            vs = np.sum(volume[0:i + 1])
            ps = 0
            for j in range(0, i + 1):
                s = close[j] * volume[j]
                ps = ps + s
            vwap[i][0] = ps / vs
        else:
            vs = np.sum(volume[(i + 1 - number): i + 1])
            ps = 0
            for j in range(i - number + 1, i + 1):
                s = close[j] * volume[j]
                ps = ps + s
            vwap[i][0] = ps / vs
    return vwap


##实现VWAP的上行线数据，number1既是VWAP的天数也是计算标准差的天数，number2是标准差的倍数
def VWAP_UP(name, number1, number2):
    upp = VWAP(name, number1)
    u = np.zeros((len(upp), 1))
    up = np.zeros((len(upp), 1))
    for i in range(0, len(up)):
        if i < number1:
            u[i][0] = np.std(upp[0:i + 1])
            up[i][0] = number2 * u[i][0] + upp[i][0]
        else:
            u[i][0] = np.std(upp[(i + 1 - number1): i + 1])
            up[i][0] = number2 * u[i][0] + upp[i][0]

    return up


##实现VWAP的下行线数据，number1既是VWAP的天数也是计算标准差的天数，number2是标准差的倍数
def VWAP_DOWN(name, number1, number2):
    downn = VWAP(name, number1)
    d = np.zeros((len(downn), 1))
    down = np.zeros((len(downn), 1))
    for i in range(0, len(down)):
        if i < number1:
            d[i][0] = np.std(downn[0:i + 1])
            down[i][0] = -number2 * d[i][0] + downn[i][0]
        else:
            d[i][0] = np.std(downn[(i + 1 - number1): i + 1])
            down[i][0] = -number2 * d[i][0] + downn[i][0]

    return down


'''
shows the percentage change in a triple exponentially smoothed moving average
'''
def TRIX(data, number):
    close = data['CLOSE']
    trix = np.array(close).reshape((len(close), 1))
    trix = EMA_FUNCTION(trix, number)
    trix = EMA_FUNCTION(trix, number)
    trix = EMA_FUNCTION(trix, number)
    return trix

###define the function of TMA
##实现TMA的数据，即对TRIX的数据进行一次简单移动平均，天数为number2
def TMA(name,number1, number2):
    data = TRIX(name, number1)
    tma = SMA_FUNCTION(data, number2)
    return tma


###define the function of bias, and the data should be the moving average of the price
###实现BIAS,输入数据应该为价格的移动平均值
def BIAS(data, name):
    close = data['CLOSE']
    close = np.array(close).reshape((len(close), 1))
    bias = np.zeros((len(close), 1))
    for i in range(0, len(data)):
        s = close[i][0] - data[i][0]
        bias[i][0] = 100 * s / (data[i][0])
    return bias


#### define the update bias which could calculate the bias between moving averages data
###实现了计算两条指数平均线之间的BIAS
def BIAS_UPDATE(data1, data2):
    bias = np.zeros((len(data1), 1))
    for i in range(0, len(data1)):
        s = data1[i][0] - data2[i][0]
        bias[i][0] = 100 * s / (data2[i][0])
    return bias


def UP_DOWN(data, number1, number2):  ##根据输入的数据，计算并得到这个数据的上行线和下行线，number1表示计算标准差的空间
    upp = data  ###number2表示上下行线与原数据的差距是number2倍的标准差
    u = np.zeros((len(upp), 1))
    up = np.zeros((len(upp), 1))
    downn = data
    d = np.zeros((len(downn), 1))
    down = np.zeros((len(downn), 1))
    for i in range(0, len(up)):
        if i < number1:
            u[i][0] = np.std(upp[0:i + 1])
            up[i][0] = number2 * u[i][0] + upp[i][0]
        else:
            u[i][0] = np.std(upp[(i + 1 - number1): i + 1])
            up[i][0] = number2 * u[i][0] + upp[i][0]

    for i in range(0, len(up)):
        if i < number1:
            d[i][0] = np.std(downn[0:i + 1])
            down[i][0] = number2 * d[i][0] + downn[i][0]
        else:
            d[i][0] = np.std(downn[(i + 1 - number1): i + 1])
            down[i][0] = number2 * d[i][0] + downn[i][0]
    up_down = np.concatenate((down, up), axis=1)
    return up_down
    ###返回值是一组包含上行线和下行线数据的数据


def cross(data1, data2):  ####定义一个函数可以根据输入的两组数据，输出一组shape相同的表示平行，从下穿过和从上穿过的数据
    x = data1  ###数据1
    y = data2  ###数据2
    z = np.zeros((x.shape))
    for i in range(0, len(z)):
        if i == 0:
            if x[i] >= y[i]:
                z[i] = 0.5
            else:
                z[i] = -0.5
        else:
            if x[i - 1] >= y[i - 1] and x[i] >= y[i]:
                z[i] = 0.5  ###数据1组成的折线，在上一时间点和这一时间点都在数据2的上方
            elif x[i - 1] < y[i - 1] and x[i] < y[i]:
                z[i] = -0.5  ###数据1组成的折线，在上一时间点和这一时间点都在数据2的下方
            elif x[i - 1] < y[i - 1] and x[i] >= y[i]:
                z[i] = 1  ###数据1组成的折线，在上一时间点在数据2的下方，在这一时间点在数据2的上方，说明数据1从下往上穿过了数据2
            else:
                z[i] = -1  ###数据1组成的折线，在上一时间点在数据2的上方，在这一时间点在数据2的下方，说明数据1从上往下穿过了数据2
    return z
