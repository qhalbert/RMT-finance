#encoding=utf8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os
import traceback


def normalization(ogn_fct_df, if_jy_df, N):
    ogn_fct_df = ogn_fct_df.mask(~if_jy_df)
    df_med = ogn_fct_df.median(axis=1)
    diff_med = (ogn_fct_df.sub(df_med, axis=0)).abs().median(axis=1)
    # ogn_fct_df = ogn_fct_df.clip(df_med - N * diff_med, df_med + N * diff_med, axis=0)
    # ogn_fct_df = ogn_fct_df.clip(ogn_fct_df.quantile(0.01, axis=1), ogn_fct_df.quantile(0.99, axis=1), axis=0)
    if_outlier = ogn_fct_df.gt(df_med + N * diff_med, axis=0) | ogn_fct_df.lt(df_med - N * diff_med, axis=0)
    ogn_fct_df = ogn_fct_df.mask(if_outlier)
    df_std = ogn_fct_df.std(axis=1)
    df_mean = ogn_fct_df.mean(axis=1)
    # zscore标准化
    std_fct_df = ogn_fct_df.sub(df_mean, axis=0).div(df_std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    return std_fct_df


def zxh(tm, ogn_fct_df, zxh_df, if_jy_df):
    try:
        if_jy = if_jy_df.loc[tm].values
        ogn_fct_on = ogn_fct_df.loc[tm].values[if_jy]
        zxh_meta_T = zxh_df.loc[tm].values.reshape(-1, if_jy.shape[0])
        zxh_met_T = zxh_meta_T[:, if_jy]
        zxh_met_T = zxh_met_T[np.abs(zxh_met_T).sum(axis=1) > 0, :]
        ogn_fct_on_met = ogn_fct_on.reshape(ogn_fct_on.shape[0], 1)
        zxh_met = zxh_met_T.T
        coef = np.linalg.inv((zxh_met_T).dot(zxh_met)).dot(zxh_met_T).dot(ogn_fct_on_met)
        ogn_fct_on_met_zxh = ogn_fct_on_met - zxh_met.dot(coef)
        ogn_fct_on_met_zxh = ogn_fct_on_met_zxh.ravel()
        fct_zxh_n = np.zeros(if_jy.shape)
        fct_zxh_n[if_jy] = ogn_fct_on_met_zxh
        return fct_zxh_n
    except:
        print(tm)
        traceback.print_exc()
        return np.zeros(if_jy_df.loc[tm].size)

stock_data_path = ''
stocks = pd.read_parquet(stock_data_path + 'stock_list/20220607.parquet')['stocks'].tolist()
stock_data_daily_path = stock_data_path + 'daily/'
stock_data_daily_style = stock_data_path + 'factor_style/'
date_end = datetime.datetime(2021, 11, 15)
shift = 1
Nn = 30
if_neutralize = False
sdt = datetime.datetime(2019, 1, 1)
sdt2 = datetime.datetime(2019, 1, 1)
# train_start = datetime.datetime(2015, 7, 1)
# train_end = sdt2
factor_path = stock_data_path + 'NPV/factor/'
# factor_path = '/mnt/Data/rpan/factor_pool/select/'
factor_neu_path = stock_data_path + 'NPV/factor_neu/'
factor_figure_path = stock_data_path + 'NPV/fig_test/'
factor_ic_path = stock_data_path + 'NPV/factor_ic/'
factor_excess_path = stock_data_path + 'NPV/factor_excess/'
for pat in [factor_path, factor_neu_path, factor_figure_path, factor_ic_path, factor_excess_path]:
    if not os.path.exists(pat):
        os.makedirs(pat)

fns = os.listdir(factor_path)

factor_names = map(lambda x: x.split('.')[0], fns)
if_st_daily = pd.read_parquet("{}/if_st.parquet".format(stock_data_daily_path)).replace([np.inf, -np.inf],
                                                                                        np.nan).fillna(method='ffill')[stocks]
if_st_daily = if_st_daily[if_st_daily.index > sdt].copy()
if_jy_daily = pd.read_parquet("{}/if_jy.parquet".format(stock_data_daily_path)).replace([np.inf, -np.inf],
                                                                                        np.nan).fillna(method='ffill')[stocks]
if_jy_daily = if_jy_daily[if_jy_daily.index > sdt].copy()
if_zdt_daily = pd.read_parquet("{}/if_zdt.parquet".format(stock_data_daily_path)).replace([np.inf, -np.inf],
                                                                                               np.nan).fillna(method='ffill')[stocks]
if_zdt_daily = if_zdt_daily[if_zdt_daily.index > sdt].copy()
on_day_daily = pd.read_parquet("{}/on_day.parquet".format(stock_data_daily_path)).replace([np.inf, -np.inf],
                                                                                          np.nan).fillna(method='ffill')[stocks]
on_day_daily = on_day_daily[on_day_daily.index > sdt].copy()

amount_daily = pd.read_parquet("{}/amount.parquet".format(stock_data_daily_path)).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')[stocks]
amount_daily = amount_daily[amount_daily.index > sdt].copy()
amount_daily = amount_daily.mask(if_jy_daily == 0).rolling(22, min_periods=1).mean().fillna(method='ffill')

# if_zz800_daily = pd.read_parquet("{}/if_size.parquet".format(stock_data_daily_path)).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
# # if_zz800_daily = if_zz800_daily[if_zz800_daily.index > sdt].copy()

if_remain_daily_1 = (if_st_daily.shift(-1).fillna(0) == 0) & (if_jy_daily.shift(-1).fillna(1) == 1) & (if_zdt_daily.shift(-1).fillna(0) == 0) & (on_day_daily > 50)
if_remain_daily = (if_st_daily == 0) & (if_jy_daily == 1) & (if_zdt_daily == 0) & (on_day_daily > 50)
if_remain_daily = if_remain_daily_1 & if_remain_daily
twap_daily = pd.read_parquet(stock_data_daily_path + 'twap.parquet')[stocks]
twap_daily = twap_daily[twap_daily.index > sdt].copy()

rtn_daily0 = twap_daily.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

print('read zxh data')
zxh_ind_fns = ['801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI', '801080.SI', '801110.SI', '801120.SI',
               '801130.SI', '801140.SI', '801150.SI', '801160.SI', '801170.SI', '801180.SI', '801200.SI', '801210.SI',
               '801230.SI', '801710.SI', '801720.SI', '801730.SI', '801740.SI', '801750.SI', '801760.SI', '801770.SI',
               '801780.SI', '801790.SI', '801880.SI', '801890.SI', '801950.SI', '801960.SI', '801970.SI', '801980.SI']
# zxh_ind_fns = []
# zxh_style_fns = ['ln_ld_size']
zxh_style_fns = ['earnings_yield', 'growth', 'lnturnover22', 'ln_ld_size', 'nonlinear_size', 'volatility22', 'rev22', 'roe_ttm', 'beta', 'lnbp']

zxh_list = []
for fn in zxh_ind_fns:
    df_s = pd.read_parquet(stock_data_daily_style + '%s.parquet' % fn)[stocks]
    df_s = df_s[df_s.index > sdt]
    df_s = df_s.mask(~if_remain_daily)
    zxh_list.append(df_s)
for fn in zxh_style_fns:
    df_s = pd.read_parquet(stock_data_daily_style + '%s.parquet' % fn)[stocks]
    df_s = df_s[df_s.index > sdt]
    df_s_n = normalization(df_s, if_remain_daily, 20)
    df_s_n = df_s_n.mask(~if_remain_daily)
    zxh_list.append(df_s_n)
df_zxhs = pd.concat(zxh_list, axis=0)
for factor_name in factor_names:
    print(factor_name)
    factor_fn = factor_path + '%s.parquet' % factor_name

    factor = pd.read_parquet(factor_fn)
    factor = factor.groupby(factor.index.normalize()).last().fillna(method='ffill')
    factor = factor[(factor.index > sdt)].copy()

    factort = pd.DataFrame(columns=twap_daily.columns, index=twap_daily.index)
    inner_columns = twap_daily.columns.join(factor.columns, how='inner')
    factort[inner_columns] = factor[inner_columns]
    factort = factort.fillna(method='ffill')
    # remove outlier

    factorta = factort.mask(~if_remain_daily)
    # up = factort.quantile(0.999, axis=1)
    # down = factort.quantile(0.001, axis=1)
    # if_not_outlier = factort.le(up, axis=0) & factort.ge(down, axis=0)
    # if_remain_plus_daily = if_remain_daily & factort.lt(up, axis=0) & factort.gt(down, axis=0)
    # if_remain_plus_daily = if_remain_daily & if_not_outlier
    # factort.mask(~if_remain_plus_daily).to_parquet('%s.parquet' % factor_name)
    factort = normalization(factorta, if_remain_daily, Nn)
    # factort.to_parquet('%s_n.parquet' % factor_name)
    factort = factort.mask(~if_remain_daily)
    n = 0
    df_gp = pd.DataFrame()
    df_ic = pd.DataFrame()
    df_excess_avg = pd.DataFrame()
    df_excess = pd.DataFrame()
    df_hsr = pd.DataFrame()
    factort_rank = factort.rank(axis=1, pct=True).sub(0.5 / factort.count(axis=1), axis=0).replace([np.inf, -np.inf], np.nan)
    fct_n_pct = 2 * (factort_rank - 0.5)
    fig1 = plt.figure(figsize=(20, 16))
    for N, lag in [(1, 1), (1, 2), (3, 3), (5, 6)]:
        n = n + 1
        rtn_daily = rtn_daily0.shift(-(shift + lag)).fillna(0)
        rtn_daily = rtn_daily.mask(~if_remain_daily)
        rtnn = twap_daily.pct_change(N).shift(1 - N - lag - shift).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        rtnn = rtnn.mask(~if_remain_daily)

        ics = factort.corrwith(rtnn, axis=1, method='spearman').replace([np.inf, -np.inf], np.nan).fillna(0)
        icsm = ics.resample('M').mean()
        f = icsm.cumsum().iloc[-1]
        # df_ic = pd.DataFrame({'uncentre': icsm.values}, index=icsm.index)
        df_ic['ic_%s' % lag] = icsm
        # df_ic_cum = df_ic.cumsum()
        gps = (fct_n_pct.rolling(N).mean() * rtn_daily).mean(axis=1).shift(lag + shift).fillna(0)
        gpsm = gps.resample('D').sum(min_count=1).dropna()
        # df_gp = pd.DataFrame({'uncentre': gpsm.values}, index=gpsm.index)
        df_gp['long_short_%s' % lag] = gpsm.cumsum()
        # df_gp = df_gp.cumsum()
        # ps = factort_rank.mask(factort_rank <= 0.92).fillna(0)
        if f > 0:
            ps = pd.DataFrame(np.where(factort_rank > 0.9, 1, 0), index=factort.index, columns=factort.columns).fillna(0)

        else:
            ps = pd.DataFrame(np.where(factort_rank < 0.1, 1, 0), index=factort.index, columns=factort.columns).fillna(0)
        # ps_ld_size = ld_size.mask(ps == 0, 0)
        # wsize = ps_ld_size.div(ps_ld_size.sum(axis=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        # ps = ps * wsize
        ps = ps.rolling(N).mean().fillna(0)
        hsr = ((ps - ps.shift(1)).abs().sum(axis=1) / (2 * ps.abs().sum(axis=1))).replace([np.inf, -np.inf], np.nan)
        hsrm = hsr.resample('M').mean()
        # df_hsr = pd.DataFrame({'uncentre': hsrm.values}, index=hsrm.index)
        df_hsr['turnover_%s' % lag] = hsrm

        ps = ps.mask(~if_remain_daily)
        w = ps.div(ps.sum(axis=1, min_count=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
        top = (rtn_daily * w).sum(axis=1, min_count=1).shift(lag + shift).fillna(0)

        df_zz500 = pd.read_csv(stock_data_path + '000905.XSHG.csv')
        df_zz500['time'] = pd.to_datetime(df_zz500['time'])
        df_zz500 = df_zz500.set_index('time')
        idx_rtn = df_zz500['twap'].pct_change(1).fillna(0)
        # idx_rtn = rtn_daily.mean(axis=1).shift(2).fillna(0)
        idx_rtn = idx_rtn[idx_rtn.index >= datetime.datetime(2017, 1, 1)]
        avg_rtn = rtn_daily.mean(axis=1).shift(lag + shift).fillna(0)
        top = top[top.index <= idx_rtn.index[-1]]

        excess_avg = (top - avg_rtn).cumsum()
        excess_avg = excess_avg[excess_avg.index > sdt2]
        excess_avg = excess_avg - excess_avg.iloc[0]
        # df_excess_avg = pd.DataFrame({'excess': excess_avg.values}, index=excess_avg.index)
        df_excess_avg['excess_avg_%s' % lag] = excess_avg
        # df_excess_avg = df_excess_avg[df_excess_avg.index > sdt]
        excess = (top - idx_rtn).cumsum()
        excess = excess[excess.index > sdt2]
        excess = excess - excess.iloc[0]
        # df_excess = pd.DataFrame({'excess': excess.values}, index=excess.index)
        df_excess['excess_zz500_%s' % lag] = excess

        bin = 0.1
        steps = np.arange(0, 1, bin) + bin
        step_1 = 0
        df_step = pd.DataFrame(index=rtn_daily.index)
        k = 0
        for step in steps:
            k = k + 1
            ps_step = pd.DataFrame(np.where((factort_rank >= step_1) & (factort_rank < step), 1, 0), index=factort_rank.index, columns=factort_rank.columns).fillna(method='ffill').fillna(0)
            ps_step = ps_step.rolling(N).mean().fillna(0)
            w_step = ps_step.div(ps_step.sum(axis=1, min_count=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
            rtn_step = (w_step * rtn_daily).sum(axis=1, min_count=1).shift(shift + lag).fillna(0)
            df_step['bin%s' % k] = rtn_step
            step_1 = step
        df_step = df_step[df_step.index > sdt2]
        idx_rtn_ = idx_rtn[idx_rtn.index > sdt2]
        df_step_excess = df_step.sub(idx_rtn_, axis=0)
        df_step_excess = df_step_excess.cumsum()
        ax1 = fig1.add_subplot(2, 2, n, title='bins_%s' % lag)
        df_step_excess.index.name = None
        df_step_excess.plot.line(ax=ax1, fontsize=12)
    plt.tight_layout()
    factor_figure_path_bins = factor_figure_path + '/figure_1234/bins/'
    if not os.path.exists(factor_figure_path_bins):
        os.makedirs(factor_figure_path_bins)
    plt.savefig(factor_figure_path_bins + 'bins_%s.png' % (factor_name))
    plt.close()
    df_ic.to_parquet(factor_ic_path + '%s.parquet' % factor_name)
    df_excess_avg.diff(1).fillna(0).to_parquet(factor_excess_path + '%s.parquet' % factor_name)
    df_hsr = df_hsr.set_index((df_hsr.index.year * 100 + df_hsr.index.month) % 10000)
    df_ic = df_ic.set_index((df_ic.index.year * 100 + df_ic.index.month) % 10000)
    fig = plt.figure(figsize=(20, 16))
    ax1 = fig.add_subplot(5, 1, 1, title='gross')
    ax2 = fig.add_subplot(5, 1, 2, title='ic')
    ax3 = fig.add_subplot(5, 1, 3, title='excess_avg')
    ax4 = fig.add_subplot(5, 1, 4, title='alpha_zz500')
    ax5 = fig.add_subplot(5, 1, 5, title='turnover')

    df_gp.index.name = None
    df_ic.index.name = None
    df_excess_avg.index.name = None
    df_excess.index.name = None

    df_gp.plot.line(ax=ax1, fontsize=12)

    df_ic.plot.bar(ax=ax2, fontsize=12)

    df_excess_avg.plot.line(ax=ax3, fontsize=12)
    df_excess.plot.line(ax=ax4, fontsize=12)
    # df_excess.to_parquet(path + '%s.parquet' % factor_name)
    df_hsr.plot.bar(ax=ax5, fontsize=12)
    plt.tight_layout()
    factor_figure_path_infos = factor_figure_path + '/figure_1234/infos/'
    if not os.path.exists(factor_figure_path_infos):
        os.makedirs(factor_figure_path_infos)
    plt.savefig(factor_figure_path_infos + '%s.png' % (factor_name))
    plt.close()
    df_gp.index.name = 'time'
    df_ic.index.name = 'time'
    df_excess_avg.index.name = 'time'
    df_excess.index.name = 'time'
    sr_1 = df_ic['ic_1']
    sr_1 = sr_1[sr_1.index > 1901]
    avg_sr_1 = sr_1.mean()
    std_sr_1 = sr_1.std()
    print(avg_sr_1, avg_sr_1 / std_sr_1)
    sr_1 = df_gp['long_short_1'].diff(1)
    sr_1 = sr_1[sr_1.index > sdt2]
    avg_sr_1 = sr_1.mean()
    std_sr_1 = sr_1.std()
    print(sr_1.sum(), avg_sr_1 / std_sr_1)
    sr_1 = df_excess_avg['excess_avg_1'].diff(1)
    sr_1 = sr_1[sr_1.index > sdt2]
    avg_sr_1 = sr_1.mean()
    std_sr_1 = sr_1.std()
    print(sr_1.sum(), avg_sr_1 / std_sr_1)
    sr_1 = df_excess['excess_zz500_1'].diff(1)
    sr_1 = sr_1[sr_1.index > sdt2]
    avg_sr_1 = sr_1.mean()
    std_sr_1 = sr_1.std()
    print(sr_1.sum(), avg_sr_1 / std_sr_1)

    if if_neutralize:
        factort = normalization(factorta, if_remain_daily, 20)
        factort = factort.mask(~if_remain_daily)
        factort_neu = [zxh(tm, factort, df_zxhs, if_remain_daily) for tm in factort.index]
        factort_neu = pd.DataFrame(factort_neu, index=factort.index, columns=factort.columns)
        factort_neu = factort_neu.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        # factort_neu = factort_neu.rolling(9, min_periods=1).mean()
        factort_neu.to_parquet(factor_neu_path + '%s_neu.parquet' % factor_name)
        factort_neu = normalization(factort_neu, if_remain_daily, Nn)
        factort_neu = factort_neu.mask(~if_remain_daily)
        factort = factort_neu
        n = 0
        df_gp = pd.DataFrame()
        df_ic = pd.DataFrame()
        df_excess_avg = pd.DataFrame()
        df_excess = pd.DataFrame()
        df_hsr = pd.DataFrame()
        factort_rank = factort.rank(axis=1, pct=True).sub(0.5 / factort.count(axis=1), axis=0).replace(
            [np.inf, -np.inf], np.nan)
        fct_n_pct = 2 * (factort_rank - 0.5)
        fig1 = plt.figure(figsize=(20, 16))
        for N, lag in [(1, 1), (1, 2), (3, 3), (5, 6)]:
            n = n + 1
            rtn_daily = rtn_daily0.shift(-(shift + lag)).fillna(0)
            rtn_daily = rtn_daily.mask(~if_remain_daily)
            rtnn = twap_daily.pct_change(N).shift(1 - N - lag - shift).replace([np.inf, -np.inf], np.nan).fillna(
                method='ffill')
            rtnn = rtnn.mask(~if_remain_daily)
            ics = factort.corrwith(rtnn, axis=1, method='spearman').replace([np.inf, -np.inf], np.nan).fillna(0)
            icsm = ics.resample('M').mean()
            f = icsm.cumsum().iloc[-1]
            # df_ic = pd.DataFrame({'uncentre': icsm.values}, index=icsm.index)
            df_ic['ic_%s' % lag] = icsm
            # df_ic_cum = df_ic.cumsum()
            gps = (fct_n_pct.rolling(N).mean() * rtn_daily).mean(axis=1).shift(lag + shift).fillna(0)
            gpsm = gps.resample('D').sum(min_count=1).dropna()
            # df_gp = pd.DataFrame({'uncentre': gpsm.values}, index=gpsm.index)
            df_gp['long_short_%s' % lag] = gpsm.cumsum()
            # df_gp = df_gp.cumsum()
            # ps = factort_rank.mask(factort_rank <= 0.92).fillna(0)
            if f > 0:
                ps = pd.DataFrame(np.where(factort_rank > 0.9, 1, 0), index=factort.index,
                                  columns=factort.columns).fillna(0)
            else:
                ps = pd.DataFrame(np.where(factort_rank < 0.1, 1, 0), index=factort.index,
                                  columns=factort.columns).fillna(0)
            ps = ps.rolling(N).mean().fillna(0)
            hsr = ((ps - ps.shift(1)).abs().sum(axis=1) / (2 * ps.abs().sum(axis=1))).replace([np.inf, -np.inf], np.nan)
            hsrm = hsr.resample('M').mean()
            # df_hsr = pd.DataFrame({'uncentre': hsrm.values}, index=hsrm.index)
            df_hsr['turnover_%s' % lag] = hsrm

            ps = ps.mask(~if_remain_daily)
            w = ps.div(ps.sum(axis=1, min_count=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
            top = (rtn_daily * w).sum(axis=1, min_count=1).shift(lag + shift).fillna(0)

            df_zz500 = pd.read_csv(stock_data_path + '000905.XSHG.csv')
            df_zz500['time'] = pd.to_datetime(df_zz500['time'])
            df_zz500 = df_zz500.set_index('time')
            idx_rtn = df_zz500['twap'].pct_change(1).fillna(0)
            # idx_rtn = rtn_daily.mean(axis=1).shift(2).fillna(0)
            idx_rtn = idx_rtn[idx_rtn.index >= datetime.datetime(2017, 1, 1)]
            avg_rtn = rtn_daily.mean(axis=1).shift(lag + shift).fillna(0)
            top = top[top.index <= idx_rtn.index[-1]]

            excess_avg = (top - avg_rtn).cumsum()
            excess_avg = excess_avg[excess_avg.index > sdt2]
            excess_avg = excess_avg - excess_avg.iloc[0]
            # df_excess_avg = pd.DataFrame({'excess': excess_avg.values}, index=excess_avg.index)
            df_excess_avg['excess_avg_%s' % lag] = excess_avg
            # df_excess_avg = df_excess_avg[df_excess_avg.index > sdt]
            excess = (top - idx_rtn).cumsum()
            excess = excess[excess.index > sdt2]
            excess = excess - excess.iloc[0]
            # df_excess = pd.DataFrame({'excess': excess.values}, index=excess.index)
            df_excess['excess_zz500_%s' % lag] = excess

            bin = 0.1
            steps = np.arange(0, 1, bin) + bin
            step_1 = 0
            df_step = pd.DataFrame(index=rtn_daily.index)
            k = 0
            for step in steps:
                k = k + 1
                ps_step = pd.DataFrame(np.where((factort_rank >= step_1) & (factort_rank < step), 1, 0),
                                       index=factort_rank.index, columns=factort_rank.columns).fillna(
                    method='ffill').fillna(0)
                ps_step = ps_step.rolling(N).mean().fillna(0)
                w_step = ps_step.div(ps_step.sum(axis=1, min_count=1), axis=0).replace([np.inf, -np.inf],
                                                                                       np.nan).fillna(0)
                rtn_step = (w_step * rtn_daily).sum(axis=1, min_count=1).shift(shift + lag).fillna(0)
                df_step['bin%s' % k] = rtn_step
                step_1 = step
            df_step = df_step[df_step.index > sdt2]
            idx_rtn_ = idx_rtn[idx_rtn.index > sdt2]
            df_step_excess = df_step.sub(idx_rtn_, axis=0)
            df_step_excess = df_step_excess.cumsum()
            ax1 = fig1.add_subplot(2, 2, n, title='bins_%s' % lag)
            df_step_excess.index.name = None
            df_step_excess.plot.line(ax=ax1, fontsize=12)
        plt.tight_layout()
        factor_figure_path_bins = factor_figure_path + '/figure_1234/bins/'
        if not os.path.exists(factor_figure_path_bins):
            os.makedirs(factor_figure_path_bins)
        plt.savefig(factor_figure_path_bins + 'bins_%s_neu.png' % (factor_name))
        plt.close()
        df_ic.to_parquet(factor_ic_path + '%s_neu.parquet' % factor_name)
        df_excess_avg.diff(1).fillna(0).to_parquet(factor_excess_path + '%s_neu.parquet' % factor_name)
        df_hsr = df_hsr.set_index((df_hsr.index.year * 100 + df_hsr.index.month) % 10000)
        df_ic = df_ic.set_index((df_ic.index.year * 100 + df_ic.index.month) % 10000)
        fig = plt.figure(figsize=(20, 16))
        ax1 = fig.add_subplot(5, 1, 1, title='gross')
        ax2 = fig.add_subplot(5, 1, 2, title='ic')
        ax3 = fig.add_subplot(5, 1, 3, title='excess_avg')
        ax4 = fig.add_subplot(5, 1, 4, title='alpha_zz500')
        ax5 = fig.add_subplot(5, 1, 5, title='turnover')

        df_gp.index.name = None
        df_ic.index.name = None
        df_excess_avg.index.name = None
        df_excess.index.name = None

        df_gp.plot.line(ax=ax1, fontsize=12)
        df_ic.plot.bar(ax=ax2, fontsize=12)
        df_excess_avg.plot.line(ax=ax3, fontsize=12)
        df_excess.plot.line(ax=ax4, fontsize=12)
        df_hsr.plot.bar(ax=ax5, fontsize=12)
        plt.tight_layout()
        factor_figure_path_infos = factor_figure_path + '/figure_1234/infos/'
        if not os.path.exists(factor_figure_path_infos):
            os.makedirs(factor_figure_path_infos)
        plt.savefig(factor_figure_path_infos + '%s_neu.png' % (factor_name))
        plt.close()

        df_gp.index.name = 'time'
        df_ic.index.name = 'time'
        df_excess_avg.index.name = 'time'
        df_excess.index.name = 'time'
