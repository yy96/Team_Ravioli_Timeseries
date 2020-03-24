import numpy as np
import pandas as pd

def ma_20_200(CLOSE, settings):

    ma_200 = np.mean(CLOSE[-200:, :], axis=0)
    ma_20 = np.mean(CLOSE[-20:, :], axis=0)
    curr_prices = CLOSE[-1:, :]
    
    ma_200 = np.nan_to_num(ma_200)
    ma_20 = np.nan_to_num(ma_20)
    curr_prices = np.nan_to_num(curr_prices)

    to_long = (curr_prices < ma_20) & (curr_prices > ma_200)
    to_long = to_long[0]
    to_long = [1 if x else 0 for x in to_long]
    to_short = (curr_prices > ma_20) & (curr_prices > ma_200)
    to_short = to_short[0]
    to_short = [-1 if x else 0 for x in to_short]
    pos = [x+y for x, y in zip(to_long, to_short)]

    pos = np.array(pos).reshape(1, len(pos))

    return pos

def reversion_ema(CLOSE, settings, window_old=20, window_recent=10, lookback=60):

    df = pd.DataFrame(CLOSE[-lookback:, :])
    ema_old = df.ewm(span=window_old).mean().iloc[-1, :]
    ema_recent = df.ewm(span=window_recent).mean().iloc[-1, :]

    recent_larger = ema_recent > ema_old
    pos = [-1 if recent_is_larger else 1 for recent_is_larger in recent_larger.to_list()]
    pos = np.array(pos).reshape(1, len(pos))

    return pos

def z_reversion_helper(z_score):
    if abs(z_score) < 1:
        return 0
    return -z_score

def z_reversion(CLOSE, settings):
    period = 60

    # Calculate moving average and moving standard deviation of the past 60 days
    moving_data = CLOSE[-period:, :]
    moving_avg = np.mean(moving_data, axis=0)
    moving_std = np.std(moving_data, axis=0)

    # Calculate z-scores
    z_scores = (CLOSE[-1:, :] - moving_avg) / moving_std

    pos = np.apply_along_axis(func1d=z_reversion_helper, axis=0, arr=z_scores)

    return pos
