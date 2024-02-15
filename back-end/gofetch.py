import pandas as pd
import datetime as dt
import numpy as np

from pandas_datareader import data as pdr
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

end = dt.datetime.now()
start = end - dt.datetime(days = 3*365)

stocks =  ['AAPL', 'GOOGL','AMZN'] # Apple, Google and Amazon

df = pdr.get_data_yahoo(stocks, start, end)
df.head()

df.index
df.columns

Close = df.Close
Close.head()

Close.describe()

Close[Close.index > end - dt.timedelta(days = 100)].describe(percentiles=[0.1,0.5,0.9])

#matplotlib
Close.plot(figsize=(12,8))

#plotify
pyo.options.plotting.backend = 'plotly'
Close.plot()

#log returns
log_returns = np.log(df.Close/df.Close.shift(1)).dropna()

# Daily Standard Deviaiton of Returns
daily_std = log_returns.std() 
annualized_std = daily_std * np.sqrt(252)


TRADING_DAYS = 60
volatility = log_returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)

# RATIOs

# Sharpe ratio
Rf = 0.01/252
sharpe_ratio = (log_returns.rolling(window=TRADING_DAYS).mean() - Rf)*TRADING_DAYS / volatility

# Sortino ratio
sortino_vol = log_returns[log_returns<0].rolling(window=TRADING_DAYS, center=True, min_periods=10).std()*np.sqrt(TRADING_DAYS)
sortino_ratio = (log_returns.rolling(window=TRADING_DAYS).mean() - Rf)*TRADING_DAYS / sortino_vol

sortino_vol.plot().update_layout(autosize = False, width=600, height=300).show(renderer="colab")

sortino_ratio.plot().update_layout(autosize = False, width=600, height=300).show(renderer="colab")

#Modigliana ratio (M2 ratio)
m2_ratio = pd.DataFrame()

benchmark_vol = volatility['^AXJO']
for c in log_returns.columns:
    if c != '^AXJO':
        m2_ratio[c] = (sharpe_ratio[c]*benchmark_vol/TRADING_DAYS + Rf)*TRADING_DAYS

m2_ratio.plot().update_layout(autosize = False, width=600, height=300).show(renderer="colab")

# Max Drawdown
def max_drawdown(returns):
    cumulative_returns = (returns+1).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns/peak)-1
    return drawdown.min()


returns = df.Close.pct_change()
max_drawdowns = returns.apply(max_drawdown, axis=0)
max_drawdowns*100

# Calmar ratio
calmars = np.exp(log_returns.mean()*255)/abs(max_drawdowns)
calmars.plot.bar().update_layout(autosize = False, width=600, height=300).show(renderer="colab")
