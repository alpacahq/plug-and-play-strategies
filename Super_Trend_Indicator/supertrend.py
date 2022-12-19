import vectorbtpro as vbt
import talib
import pandas as pd
import numpy as np
from config import ALPACA_KEY, ALPACA_SECRET

PERIOD_START = "2020-11-01"
PERIOD_END = "2022-01-01"
TIMEFRAME = "1h"

vbt.AlpacaData.set_custom_settings(
             client_config=dict(
                 api_key=ALPACA_KEY,
                 secret_key=ALPACA_SECRET
             )
         )

test_symbol = "GME"

data = vbt.AlpacaData.fetch(symbols=["GME", "AAPL"], start=PERIOD_START, end=PERIOD_END, timeframe=TIMEFRAME, limit=3000)

open = data.get("Open")
high = data.get("High")
low = data.get("Low")
close = data.get("Close")

def get_med_price(high, low):
    return (high + low) / 2

def get_atr(high, low, close, period):
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)  
    atr = tr.ewm(
        alpha=1 / period, 
        adjust=False, 
        min_periods=period).mean()  
    return atr

def get_basic_bands(med_price, atr, multiplier):
    matr = multiplier * atr
    upper = med_price + matr
    lower = med_price - matr
    return upper, lower

def get_final_bands(close, upper, lower):  
    trend = pd.Series(np.full(close.shape, np.nan), index=close.index)
    dir_ = pd.Series(np.full(close.shape, 1), index=close.index)
    long = pd.Series(np.full(close.shape, np.nan), index=close.index)
    short = pd.Series(np.full(close.shape, np.nan), index=close.index)

    for i in range(1, close.shape[0]):  
        if close.iloc[i] > upper.iloc[i - 1]:
            dir_.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i - 1]
            if dir_.iloc[i] > 0 and lower.iloc[i] < lower.iloc[i - 1]:
                lower.iloc[i] = lower.iloc[i - 1]
            if dir_.iloc[i] < 0 and upper.iloc[i] > upper.iloc[i - 1]:
                upper.iloc[i] = upper.iloc[i - 1]

        if dir_.iloc[i] > 0:
             trend.iloc[i] = long.iloc[i] = lower.iloc[i]
        else:
             trend.iloc[i] = short.iloc[i] = upper.iloc[i]
           
    return trend, dir_, long, short


def supertrend(high, low, close, period=7, multiplier=3):
    med_price = get_med_price(high, low)
    atr = get_atr(high, low, close, period)
    upper, lower = get_basic_bands(med_price, atr, multiplier)
    return get_final_bands(close, upper, lower)

def faster_supertrend_talib(high, low, close, period=7, multiplier=3):
    avg_price = talib.MEDPRICE(high, low)  
    atr = talib.ATR(high, low, close, period)  
    upper, lower = get_basic_bands(avg_price, atr, multiplier)
    return get_final_bands(close, upper, lower)


SuperTrend = vbt.IF(
    class_name='SuperTrend',
    short_name='st',
    input_names=['high', 'low', 'close'],
    param_names=['period', 'multiplier'],
    output_names=['supert', 'superd', 'superl', 'supers']
).with_apply_func(
    faster_supertrend_talib, 
    takes_1d=True,  
    period=7,  
    multiplier=3
)

    
st = SuperTrend.run(high, low, close)

pf = vbt.Portfolio.from_signals(
     close=close, 
     entries=entries, 
     exits=exits, 
     fees=0.001, 
     freq='1h'
 )