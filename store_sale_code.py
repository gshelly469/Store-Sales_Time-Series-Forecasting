from pathlib import Path
import datetime

import pandas as pd # data processing, CSV file I/O
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import pickle


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# time variables (NOTE: use pre-June 2015 data cautiously as large chunks are missing)

# data time range to train on the full training set
full_train_start_day = datetime.datetime(2015, 6, 16)
full_train_end_day = datetime.datetime(2017, 8, 15)

# data time range for train/validation split 
train_start_day = full_train_start_day
train_end_day = datetime.datetime(2017, 7, 30)
val_start_day = datetime.datetime(2017, 7, 31)
val_end_day = datetime.datetime(2017, 8, 15)
# can be smart to set val_end_day to (2017, 7, 31) or (2017, 8, 1) when testing a change or debugging

# data time range of test set
test_start_day = datetime.datetime(2017, 8, 16)
test_end_day = datetime.datetime(2017, 8, 31)

if full_train_start_day > full_train_end_day:
    raise ValueError("full_train_start_day must be less than full_train_end_day . . . Did you change month without changing year?")


max_lag = 7

mod_1 = LinearRegression()
mod_2 = XGBRegressor()

hybrid_forecasting_type = "day_by_day_refit_all_days" # possible values: day_by_day_refit_all_days, day_by_day_fixed_past, or direct


plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette(
        "husl",
        n_colors=X[period].nunique(),
    )
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

# From Lesson 4

def lagplot(x, y=None, shift=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(shift)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    title = f"Lag {shift}" if shift > 0 else f"Lead {shift}"
    ax.set(title=f"Lag {shift}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x,
              y=None,
              lags=6,
              leads=None,
              nrows=1,
              lagplot_kwargs={},
              **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    orig = leads is not None
    leads = leads or 0
    kwargs.setdefault('ncols', math.ceil((lags + orig + leads) / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        k -= leads + orig
        if k + 1 <= lags:
            ax = lagplot(x, y, shift=k + 1, ax=ax, **lagplot_kwargs)
            title = f"Lag {k + 1}" if k + 1 >= 0 else f"Lead {-k - 1}"
            ax.set_title(title, fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

# From Lesson 6

def make_lags(ts, lags, lead_time=1, name='y'):
    return pd.concat(
        {
            f'{name}_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


def make_leads(ts, leads, name='y'):
    return pd.concat(
        {f'{name}_lead_{i}': ts.shift(-i)
         for i in reversed(range(leads))},
        axis=1)


def make_multistep_target(ts, steps, reverse=False):
    shifts = reversed(range(steps)) if reverse else range(steps)
    return pd.concat({f'y_step_{i + 1}': ts.shift(-i) for i in shifts}, axis=1)


def create_multistep_example(n, steps, lags, lead_time=1):
    ts = pd.Series(
        np.arange(n),
        index=pd.period_range(start='2010', freq='A', periods=n, name='Year'),
        dtype=pd.Int8Dtype,
    )
    X = make_lags(ts, lags, lead_time)
    y = make_multistep_target(ts, steps, reverse=True)
    data = pd.concat({'Targets': y, 'Features': X}, axis=1)
    data = data.style.set_properties(['Targets'], **{'background-color': 'LavenderBlush'}) \
                     .set_properties(['Features'], **{'background-color': 'Lavender'})
    return data


def load_multistep_data():
    df1 = create_multistep_example(10, steps=1, lags=3, lead_time=1)
    df2 = create_multistep_example(10, steps=3, lags=4, lead_time=2)
    df3 = create_multistep_example(10, steps=3, lags=4, lead_time=1)
    return [df1, df2, df3]


def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax

comp_dir = Path('../input/store-sales-time-series-forecasting')
df_sales_train = pd.read_csv('train.csv', parse_dates=['date'])
df_sales_test = pd.read_csv('test.csv', parse_dates=['date'])
df_trans = pd.read_csv('transactions.csv', parse_dates=['date'])
df_stores = pd.read_csv('stores.csv')
df_oil = pd.read_csv('oil.csv', parse_dates=['date'])

total_daily_sales = (
    df_sales_train.drop(columns = 'onpromotion')
    .groupby(['date', 'store_nbr'])
    .sum()
    .reset_index()
#     .unstack('store_nbr')
)

stores_temp = pd.merge(df_trans, total_daily_sales, on=['date', 'store_nbr']).drop(columns="id")

# set up 4 new features:

# NOTE: sales is # items sold, not monetary amount, so this isn't really avg ticket size, but average amount of items sold per transaction
# 1) average ticket per transaction (common retail metric) - total daily sales divided by total daily transactions
stores_temp['avg_ticket'] = stores_temp['sales'] / stores_temp['transactions']

# 2) "old" store is one that has already opened and has sales on first (non holiday) date of data set
old_stores = stores_temp[stores_temp['date'] == '2013-01-02'].store_nbr.values
df_stores['old'] = df_stores['store_nbr'].isin(old_stores)

# 3) start date for store
df_stores['start_date'] = [stores_temp[stores_temp['store_nbr']==num]['date'].dt.date.min() for num in range(1,55)]
df_stores['start_date'] = pd.to_datetime(df_stores['start_date'])

# 4) wage: wages paid in public sector is True when on 15th or last day of month
stores_temp['wage'] = (stores_temp['date'].dt.is_month_end) | (stores_temp['date'].dt.day == 15)

daily_store_totals_df = pd.merge(stores_temp, df_stores, on='store_nbr')

comp_dir = Path('../input/store-sales-time-series-forecasting')

holidays_events = pd.read_csv("holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',},
    parse_dates=['date'],
    infer_datetime_format=True,)

holidays_events['date'] = holidays_events['date'].replace({'2013-04-29':pd.to_datetime('2013-03-29')}) # 'Good Friday' mistake correction
holidays_events = holidays_events.set_index('date').to_period('D').sort_index() # note the sort after Good Friday correction

df_test = pd.read_csv('test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
 
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

store_sales = pd.read_csv('train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,)

store_sales['date'] = store_sales.date.dt.to_period('D')
# store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index() # MultiIndex

m_index = pd.MultiIndex.from_product([store_sales["store_nbr"].unique(),
                                      store_sales["family"].unique(),
                                      pd.date_range(start="2013-1-1", end="2017-8-15", freq="D").to_period('D')] # to get missing Christmas Days
                                     ,names=["store_nbr","family", "date"])
store_sales = store_sales.set_index(["store_nbr","family", "date"]).reindex(m_index, fill_value=0).sort_index()


store_sales = store_sales.unstack(['store_nbr', 'family']).fillna(0) # there are lots!
store_sales = store_sales.stack(['store_nbr', 'family'])
store_sales = store_sales[['sales','onpromotion']] # reorder columns to be in the expected order

calendar = pd.DataFrame(index=pd.date_range('2013-01-01', '2017-08-31')).to_period('D')
calendar['dofw'] = calendar.index.dayofweek

df_hev = holidays_events[holidays_events.locale == 'National'] # National level only for simplicity
df_hev = df_hev.groupby(df_hev.index).first() # Keep one event only

calendar['wd'] = True
calendar.loc[calendar.dofw > 4, 'wd'] = False
calendar = calendar.merge(df_hev, how='left', left_index=True, right_index=True)

calendar.loc[calendar.type == 'Bridge'  , 'wd'] = False
calendar.loc[calendar.type == 'Work Day', 'wd'] = True
calendar.loc[calendar.type == 'Transfer', 'wd'] = False
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == False), 'wd'] = False
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == True ), 'wd'] = True

print(calendar.tail(23))

print(holidays_events.loc[full_train_start_day:test_end_day])


class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None
        self.stack_cols = None
        self.y_resid = None

    def fit1(self, X_1, y, stack_cols=None):
        self.model_1.fit(X_1, y) # train model 1
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1), # predict from model 1
            index=X_1.index,
            columns=y.columns,
        )
        self.y_resid = y - y_fit # residuals from model 1, which X2 may want to access to create lag (or other) features
        self.y_resid = self.y_resid.stack(stack_cols).squeeze()  # wide to long
        
    def fit2(self, X_2, first_n_rows_to_ignore, stack_cols=None):
        self.model_2.fit(X_2.iloc[first_n_rows_to_ignore*1782: , :], self.y_resid.iloc[first_n_rows_to_ignore*1782:]) # Train model_2
        self.y_columns = y.columns # Save for predict method
        self.stack_cols = stack_cols # Save for predict method

    def predict(self, X_1, X_2, first_n_rows_to_ignore):
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1.iloc[first_n_rows_to_ignore: , :]),
            index=X_1.iloc[first_n_rows_to_ignore: , :].index,
            columns=self.y_columns,
        )
        y_pred = y_pred.stack(self.stack_cols).squeeze()  # wide to long
#         display(X_2.iloc[first_n_rows_to_ignore*1782: , :]) # uncomment when debugging
        y_pred += self.model_2.predict(X_2.iloc[first_n_rows_to_ignore*1782: , :]) # Add model_2 predictions to model_1 predictions
        return y_pred.unstack(self.stack_cols)

def make_dp_features(df):
    y = df.loc[:, 'sales']
    #fourier_a = CalendarFourier(freq='A', order=4)
    fourier_m = CalendarFourier(freq='M', order=4)
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True, # note how this generates terms Tue - Sun: s(2,7) through s(7,7) (trend column: Monday)
        additional_terms=[fourier_m],
        drop=True,
    )
    return y, dp

def make_X1_features(df, start_date, end_date, is_test_set=False):
    if is_test_set:
        X1 = df.rename_axis('date')
    else:
        y, dp = make_dp_features(df)
        X1 = dp.in_sample() # seasonal (weekly) and fourier (longer time frame) features are generated using DeterministicProcess
    
    # other features:
    
#     X1['wage_day'] = (X1.index.day == X1.index.daysinmonth) | (X1.index.day == 15) # wage day features seem better for XGBoost than linear regression
#     X1['wage_day_lag_1'] = (X1.index.day == 1) | (X1.index.day == 16)
    X1['NewYear'] = (X1.index.dayofyear == 1)
    X1['Christmas'] = (X1.index=='2016-12-25') | (X1.index=='2015-12-25') | (X1.index=='2014-12-25') | (X1.index=='2013-12-25')
    X1['wd']   = calendar.loc[start_date:end_date]['wd'].values
    X1['type'] = calendar.loc[start_date:end_date]['type'].values
    X1 = pd.get_dummies(X1, columns=['type'], drop_first=False)
    
    # can experiment with dropping some of the dummy features if you think they might be useless or worse
#     X1.drop(['type_Work Day', 'type_Event'], axis=1, inplace=True)
    
    if is_test_set:
        return X1
    else:
        return X1, y, dp


def encode_categoricals(df, columns):
    le = LabelEncoder()  # from sklearn.preprocessing
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

def make_X2_lags(ts, lags, lead_time=1, name='y', stack_cols=None):
    ts = ts.unstack(stack_cols)
    df = pd.concat(
        {
            f'{name}_lag_{i}': ts.shift(i, freq="D") # freq adds i extra day(s) to end: only one extra day is needed so rest will be dropped
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)
    df = df.stack(stack_cols).reset_index()
    df = encode_categoricals(df, stack_cols)
    df = df.set_index('date').sort_values(by=stack_cols) # return sorted so can correctly compute rolling means (if desired)
    return df

def make_X2_features(df, y_resid):
    stack_columns = ['store_nbr', 'family']
    
#     # promo_lag features
    shifted_promo_df = make_X2_lags(df.squeeze(), lags=2, name='promo', stack_cols=['store_nbr', 'family'])
    shifted_promo_df['promo_mean_rolling_7'] = shifted_promo_df['promo_lag_1'].rolling(window=7, center=False).mean()
    shifted_promo_df['promo_median_rolling_91'] = shifted_promo_df['promo_lag_1'].rolling(window=91, center=False).median().fillna(method='bfill')
    shifted_promo_df['promo_median_rolling_162'] = shifted_promo_df['promo_lag_1'].rolling(window=162, center=False).median().fillna(method='bfill')
    # for rolling window medians, backfilling seems reasonable as medians shouldn't change too much. Trying min_periods produced wacky (buggy?) results
    
    # y_lag features
    shifted_y_df = make_X2_lags(y_resid, lags=2, name='y_res', stack_cols=stack_columns)
    shifted_y_df['y_mean_rolling_7'] = shifted_y_df['y_res_lag_1'].rolling(window=7, center=False).mean()
#     shifted_y_df['y_mean_rolling_14'] = shifted_y_df['y_res_lag_1'].rolling(window=14, center=False).mean()
#     shifted_y_df['y_mean_rolling_28'] = shifted_y_df['y_res_lag_1'].rolling(window=28, center=False).mean()
    shifted_y_df['y_median_rolling_91'] = shifted_y_df['y_res_lag_1'].rolling(window=91, center=False).median().fillna(method='bfill')
    shifted_y_df['y_median_rolling_162'] = shifted_y_df['y_res_lag_1'].rolling(window=162, center=False).median().fillna(method='bfill')
    
    # other features
    df = df.reset_index(stack_columns)
    X2 = encode_categoricals(df, stack_columns)
    
#     X2["day_of_m"] = X2.index.day  # day of month (label encloded) feature for learning seasonality
#     X2 = encode_categoricals(df, ['day_of_m']) # encoding as categorical has tiny impact with XGBoost
    X2["day_of_w"] = X2.index.dayofweek # does absolutely nothing alone
    X2 = encode_categoricals(df, ['day_of_w'])
    old_stores_strings = list(map(str, old_stores))
    X2['old'] = X2['store_nbr'].isin(old_stores_strings) # True if store had existing sales prior to training time period   
    X2['wage_day'] = (X2.index.day == X2.index.daysinmonth) | (X2.index.day == 15) # is it bad to have this in both X1 AND X2?
    X2['wage_day_lag_1'] = (X2.index.day == 1) | (X2.index.day == 16)
    X2['promo_mean'] = X2.groupby(['store_nbr', 'family'])['onpromotion'].transform("mean") + 0.000001
    X2['promo_ratio'] = X2.onpromotion / (X2.groupby(['store_nbr', 'family'])['onpromotion'].transform("mean") + 0.000001)


    # if removing all lag features, then comment out the following two lines
    X2 = X2.merge(shifted_y_df, on=['date', 'store_nbr', 'family'], how='left')
    X2 = X2.merge(shifted_promo_df, on=['date', 'store_nbr', 'family'], how='left') # merges work if they are last line before return
    return X2

store_sales_in_date_range = store_sales.unstack(['store_nbr', 'family']).loc[full_train_start_day:full_train_end_day]

model = BoostedHybrid(model_1=mod_1, model_2=mod_2) # Boosted Hybrid

X_1, y, dp = make_X1_features(store_sales_in_date_range, full_train_start_day, full_train_end_day) # preparing X1 for hybrid model 1
model.fit1(X_1, y, stack_cols=['store_nbr', 'family']) # fit1 before make_X2_features, since X2 may want to create lag features from model.y_resid
X_2 = make_X2_features(store_sales_in_date_range # preparing X2 for hybrid model 2
                       .drop('sales', axis=1)
                       .stack(['store_nbr', 'family']),
                       model.y_resid)
model.fit2(X_2, max_lag, stack_cols=['store_nbr', 'family'])

y_pred = model.predict(X_1, X_2, max_lag).clip(0.0)

def truncateFloat(data):
    return tuple( ["{0:.2f}".format(x) if isinstance(x,float) else (x if not isinstance(x,tuple) else truncateFloat(x)) for x in data])

temp = X_2[(X_2.store_nbr == 1) & (X_2.family == 3)]
temp.iloc[max_lag: , :].apply(lambda s: truncateFloat(s)) # comment out next line if don't want to see nan rows

temp.apply(lambda s: truncateFloat(s)).head(10) # note that the fit method of BoostedHybrid class skips over nan rows

X_1.iloc[max_lag: , :].apply(lambda s: truncateFloat(s))

STORE_NBR = '1'  # 1 - 54
FAMILY = 'BEVERAGES' # display(store_sales.index.get_level_values('family').unique())

ax = y.loc(axis=1)[STORE_NBR, FAMILY].plot(**plot_params, figsize=(16, 4))
ax = y_pred.loc(axis=1)[STORE_NBR, FAMILY].plot(ax=ax)
ax.set_title(f'{FAMILY} Sales at Store {STORE_NBR}');

training_days = (train_end_day - train_start_day).days + 1
validation_days = (val_end_day - val_start_day).days + 1
print("training data set (excluding validation days) has", training_days, "days")
print("validation data set has", validation_days, "days\n")

store_sales_in_date_range = store_sales.unstack(['store_nbr', 'family']).loc[train_start_day:train_end_day]
store_data_in_val_range = store_sales.unstack(['store_nbr', 'family']).loc[val_start_day:val_end_day]
y_val = y[val_start_day:val_end_day] # use y to evaluate validation set, though we will treat y as unknown when training

model_for_val = BoostedHybrid(model_1=mod_1, model_2=mod_2)

if hybrid_forecasting_type == "day_by_day_refit_all_days":
    #initial fit on train portion of train/val split
    X_1_train, y_train, dp_val = make_X1_features(store_sales_in_date_range, train_start_day, train_end_day) # preparing X1 for hybrid part 1: LinearRegression
    model_for_val.fit1(X_1_train, y_train, stack_cols=['store_nbr', 'family']) # fit1 before make_X2_features, since X2 may want to create lag features from model.y_resid
    X_2_train = make_X2_features(store_sales_in_date_range
                           .drop('sales', axis=1)
                           .stack(['store_nbr', 'family']),
                           model_for_val.y_resid) # preparing X2 for hybrid part 2: XGBoost
    model_for_val.fit2(X_2_train, max_lag, stack_cols=['store_nbr', 'family'])
    y_fit = model_for_val.predict(X_1_train, X_2_train, max_lag).clip(0.0)

    # loop through forecast, one day ("step") at a time
    dp_for_full_X1_val_date_range = dp_val.out_of_sample(steps=validation_days)
    for step in range(validation_days):
        dp_steps_so_far = dp_for_full_X1_val_date_range.loc[val_start_day:val_start_day+pd.Timedelta(days=step),:]

        X_1_combined_dp_data = pd.concat([dp_val.in_sample(), dp_steps_so_far])
        X_2_combined_data = pd.concat([store_sales_in_date_range,
                                       store_data_in_val_range.loc[val_start_day:val_start_day+pd.Timedelta(days=step), :]])
        X_1_val = make_X1_features(X_1_combined_dp_data, train_start_day, val_start_day+pd.Timedelta(days=step), is_test_set=True)
        X_2_val = make_X2_features(X_2_combined_data
                                    .drop('sales', axis=1)
                                    .stack(['store_nbr', 'family']),
                                    model_for_val.y_resid) # preparing X2 for hybrid part 2: XGBoost

    #     print("last 3 rows of X_1_val: ")
    #     display(X_1_val.tail(3))
    #     temp_val2 = X_2_val[(X_2_val.store_nbr == 1) & (X_2_val.family == 3)]
    #     print("last 3 rows of X_2_val: ")
    #     display(temp_val2.tail(3).apply(lambda s: truncateFloat(s)))

        y_pred_combined = model_for_val.predict(X_1_val, X_2_val, max_lag).clip(0.0) # generate y with 
    #     print("last 3 rows of y_combined: ")
    #     display(y_pred_combined.tail(3).apply(lambda s: truncateFloat(s)))
        y_plus_y_val = pd.concat([y_train, y_pred_combined.iloc[-(step+1):]]) # add newly predicted rows of y_pred_combined
        model_for_val.fit1(X_1_val, y_plus_y_val, stack_cols=['store_nbr', 'family']) # fit on new combined X, y - note that fit prior to val date range will change slightly
        model_for_val.fit2(X_2_val, max_lag, stack_cols=['store_nbr', 'family'])

        rmsle_valid = mean_squared_log_error(y_val.iloc[step:step+1], y_pred_combined.iloc[-1:]) ** 0.5
        print(f'Validation RMSLE: {rmsle_valid:.5f}', "for", val_start_day+pd.Timedelta(days=step))
    #     print("end of round ", step)

    y_pred = y_pred_combined[val_start_day:val_end_day]
    print("\ny_pred: ")
    y_pred.apply(lambda s: truncateFloat(s))
    
    if type(model_for_val.model_2) == XGBRegressor:
        pickle.dump(model_for_val.model_2, open("xgb_temp.pkl", "wb"))
        m2 = pickle.load(open("xgb_temp.pkl", "rb"))
        print("XGBRegressor paramaters:\n",m2.get_xgb_params(), "\n")

rmsle_train = mean_squared_log_error(y_train.iloc[max_lag: , :].clip(0.0), y_fit) ** 0.5
rmsle_valid = mean_squared_log_error(y_val.clip(0.0), y_pred) ** 0.5
print()
print(f'Training RMSLE: {rmsle_train:.5f}')
print(f'Validation RMSLE: {rmsle_valid:.5f}')
    
y_predict = y_pred.stack(['store_nbr', 'family']).reset_index()
y_target = y_val.stack(['store_nbr', 'family']).reset_index().copy()
y_target.rename(columns={y_target.columns[3]:'sales'}, inplace=True)
y_target['sales_pred'] = y_predict[0].clip(0.0) # Sales should be >= 0
y_target['store_nbr'] = y_target['store_nbr'].astype(int)

print('\nValidation RMSLE by family')
y_target.groupby('family').apply(lambda r: mean_squared_log_error(r['sales'], r['sales_pred']))

print('\nValidation RMSLE by store')
y_target.sort_values(by="store_nbr").groupby('store_nbr').apply(lambda r: mean_squared_log_error(r['sales'], r['sales_pred']))

############TEST DATA

train_days = (full_train_end_day - full_train_start_day).days + 1
test_days = (test_end_day - test_start_day).days + 1

print("data trained over", train_days, "days")
print("test forecasting period is", test_days, "days through", test_end_day, "\n")
store_sales_in_date_range = store_sales.unstack(['store_nbr', 'family']).loc[full_train_start_day:full_train_end_day]
store_data_in_test_range = df_test.unstack(['store_nbr', 'family']).drop('id', axis=1)

# previously prepared data and fit "model" from data ranging from full_train_start_day to full_train_end_day. Can be used by when fitting test.
model_for_test = BoostedHybrid(model_1=mod_1, model_2=mod_2)

if hybrid_forecasting_type == "day_by_day_refit_all_days":
    #initial fit on train portion of train/test split
    X_1_train, y_train, dp_test = make_X1_features(store_sales_in_date_range, full_train_start_day, full_train_end_day) # preparing X1 for hybrid part 1: LinearRegression
    model_for_test.fit1(X_1_train, y_train, stack_cols=['store_nbr', 'family']) # fit1 before make_X2_features, since X2 may want to create lag features from model.y_resid
    X_2_train = make_X2_features(store_sales_in_date_range
                           .drop('sales', axis=1)
                           .stack(['store_nbr', 'family']),
                           model_for_test.y_resid) # preparing X2 for hybrid part 2: XGBoost
    model_for_test.fit2(X_2_train, max_lag, stack_cols=['store_nbr', 'family'])
    # y_full_train = model_for_test.predict(X_1_train, X_2_train, max_lag).clip(0.0) # do I need this line here?


    dp_for_full_X1_test_date_range = dp_test.out_of_sample(steps=test_days)
    for step in range(test_days):
        dp_steps_so_far = dp_for_full_X1_test_date_range.loc[test_start_day:test_start_day+pd.Timedelta(days=step),:]

        X_1_combined_dp_data = pd.concat([dp_test.in_sample(), dp_steps_so_far])
        X_2_combined_data = pd.concat([store_sales_in_date_range,
                                       store_data_in_test_range.loc[test_start_day:test_start_day+pd.Timedelta(days=step), :]])
        X_1_test = make_X1_features(X_1_combined_dp_data, train_start_day, test_start_day+pd.Timedelta(days=step), is_test_set=True)
        X_2_test = make_X2_features(X_2_combined_data
                                    .drop('sales', axis=1)
                                    .stack(['store_nbr', 'family']),
                                    model_for_test.y_resid) # preparing X2 for hybrid part 2: XGBoost
    #     print("last 3 rows of X_1_test: ")
    #     display(X_1_test.tail(3))
    #     temp_test2 = X_2_test[(X_2_test.store_nbr == 1) & (X_2_test.family == 3)]
    #     print("last 3 rows of X_2_test: ")
    #     display(temp_test2.tail(3).apply(lambda s: truncateFloat(s)))

        y_forecast_combined = model_for_test.predict(X_1_test, X_2_test, max_lag).clip(0.0) # generate y with 

    #     print("last 3 rows of y_forecast_combined: ")
    #     display(y_forecast_combined.tail(3).apply(lambda s: truncateFloat(s)))

        y_plus_y_test = pd.concat([y_train, y_forecast_combined.iloc[-(step+1):]]) # add newly predicted (last step+1) rows of y_test_combined
        model_for_test.fit1(X_1_test, y_plus_y_test, stack_cols=['store_nbr', 'family']) # fit on new combined X, y - note that fit prior to test date range will change slightly
        model_for_test.fit2(X_2_test, max_lag, stack_cols=['store_nbr', 'family'])
        print("finished forecast for", test_start_day+pd.Timedelta(days=step))

    y_forecast_combined[test_start_day:test_end_day]

    y_forecast = pd.DataFrame(y_forecast_combined[test_start_day:test_end_day].clip(0.0), index=X_1_test.index, columns=y.columns)
    print('\nFinished creating test set forecast\n')
    
    if type(model_for_test.model_2) == XGBRegressor:
        pickle.dump(model_for_test.model_2, open("xgb_temp.pkl", "wb"))
        m2 = pickle.load(open("xgb_temp.pkl", "rb"))
        print("XGBRegressor paramaters:\n",m2.get_xgb_params(), "\n")

y_submit = y_forecast.stack(['store_nbr', 'family'])
y_submit = pd.DataFrame(y_submit, columns=['sales'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission.csv', index=False)
