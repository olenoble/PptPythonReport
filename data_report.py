import pandas as pd
import pylab as plt
import dateutil as du
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


path_ref = '/home/olivier/Code/Python/Reports/reports/'
os.chdir(path_ref)

############################################################
#### Get data

def get_data_csv(filename, datecolumn='Date'):
	data = pd.read_csv(filename)
	data[datecolumn] = [du.parser.parse(x) for x in data[datecolumn]]
	data.set_index(datecolumn, inplace=True)
	return data


# get strategy index level
idx = get_data_csv('index_level.csv')
undl = get_data_csv('prices_ts.csv')

# benchmark data
benchmark = get_data_csv('XLK.csv')

# weight data
weight = get_data_csv('weight_hist.csv')

############################################################
#### combine data and generate analytics
compare_data = pd.merge(idx, benchmark, left_index=True, right_index=True)
compare_data = compare_data.apply(lambda x: 100 * x / x[0], 0)


def annualized_return(x):
	years = (max(x.index) - min(x.index)).days / 365.0
	return -1 + (x[-1] / x[0]) ** (1 / years)

def volatility(x, scaling=252):
	logret = np.diff(np.log(x))
	return np.sqrt(scaling * np.var(logret, ddof=1))

def absolute_move(x, scaling=12):
	x_change = np.diff(x)
	return np.sqrt(scaling * np.var(x_change, ddof=1))

def sharpe_ratio(x):
	return annualized_return(x) / volatility(x)

def max_drawdown(x):
	return min(x / x.cummax() - 1.0)

results = [compare_data.apply(annualized_return, 0),
		   compare_data.apply(volatility, 0),
		   compare_data.apply(sharpe_ratio, 0),
		   compare_data.apply(max_drawdown, 0)
		   ]


results_df = pd.concat(results, 1)
results_df.columns = ['Annualized Return', 'Volatility', 'Sharpe', 'Max Drawdown']
results_df = results_df.T

weight_total_allocation = weight.apply(sum, 1)
results_totwgt = [np.mean(weight_total_allocation),
		   	      weight_total_allocation[-1],
		   	      min(weight_total_allocation),
		   	      max(weight_total_allocation)
		   	      ]
results_totwgt_df = pd.DataFrame(results_totwgt, index=['Average', 'Current', 'Min', 'Max'], columns=['Total Allocation'])

results_wgt = [weight.apply(np.mean, 0),
		   	   weight.apply(absolute_move, 0, args=(12,))
		   	   ]

results_wgt_df = pd.concat(results_wgt, 1)
results_wgt_df.columns = ['Average Weight', 'Volatility']


#results_wgt_df.to_html(float_format=lambda x: str(100 * np.round(x, 4)) + '%')

## check MoM and YoY returns
monthly_period = pd.period_range(start=min(compare_data.index), end=max(compare_data.index), freq='1M')
yearly_period = pd.period_range(start=min(compare_data.index), end=max(compare_data.index), freq='12M')

MoM_dates = [x.to_timestamp() for x in monthly_period]
YoY_dates = [x.to_timestamp() for x in yearly_period]

def amend_periods(x, first_date, last_date):
	x = [max(t, first_date) for t in x] + [last_date]
	return np.unique(x)

MoM_dates = amend_periods(MoM_dates, min(compare_data.index), max(compare_data.index))
YoY_dates = amend_periods(YoY_dates, min(compare_data.index), max(compare_data.index))


def aj(datelist, x):
	if isinstance(x, pd.DataFrame):
		return x.apply(lambda y: y.asof(datelist))
	else:
		x.asof(datelist)


mom_returns = aj(MoM_dates, compare_data).apply(lambda x: x / x.shift() - 1, 0)[1:]
yoy_returns = aj(YoY_dates, compare_data).apply(lambda x: x / x.shift() - 1, 0)[1:]
yoy_returns.index = [x.date().strftime('%b%y') for x in yoy_returns.index]


############################################################
#### Sheet 1
from jinja2 import Environment, FileSystemLoader


ax = compare_data.plot()
fig = ax.get_figure()
fig.savefig(path_ref + 'strategy_performance.svg')

perf_html = results_df.to_html(float_format=lambda x: str(100 * np.round(x, 4)) + '%')

template_vars = {'title' : 'Strategy Performance Overview',
                 'perf_table': perf_html,
                 'strategy_plot': 'strategy_performance.svg'}

def performance_html_generate(x, outfile='performance.html'):
	env = Environment(loader=FileSystemLoader('.'))
	template = env.get_template("performance_template.html")
	html_out = template.render(x)

	f = open(outfile,'w')
	f.write(html_out)
	f.close()


performance_html_generate(template_vars)

############################################################
#### Sheet 2
mom_stats = [mom_returns.apply(lambda x: x[x == min(x)].index[0], 0),
		     mom_returns.apply(min, 0),
		     mom_returns.apply(lambda x: x[x == max(x)].index[0], 0),
		     mom_returns.apply(max, 0)
		   ]

mom_stats_df = pd.concat(mom_stats, 1)
mom_stats_df.columns = ['Worst Month', 'Worst Month Return', 'Best Month', 'Best Month Return']
mom_stats_html = mom_stats_df.to_html(float_format=lambda x: str(100 * np.round(x, 4)) + '%')

daily_ret = compare_data.apply(lambda x: x / x.shift() - 1, 0)[1:]
correl = np.correlate(daily_ret['Strategy'], daily_ret['XLK'])[0]
covmat = np.cov(daily_ret.values.T)
beta = covmat[1,0] / covmat[0,0]
alpha = np.mean(daily_ret['Strategy']) - beta * np.mean(daily_ret['XLK'])

benchmark_check_df = pd.DataFrame({'Correlation': correl, 'beta': beta, 'alpha': alpha}, index=['Strategy vs Benchmark']).T
benchmark_check_html = benchmark_check_df.to_html(float_format=lambda x: str(100 * np.round(x, 4)) + '%')


ax = yoy_returns.plot(kind='bar')
fig = ax.get_figure()
fig.savefig(path_ref + 'yoy_returns.svg')

ax = mom_returns.plot(x='Strategy', y='XLK', kind='scatter', title='Monthly Returns')
fig = ax.get_figure()
fig.savefig(path_ref + 'mom_returns.svg')


template_vars = {'title' : 'Strategy Performance Breakdown',
                 'perf_table': mom_stats_html,
                 'benchmark_reg': benchmark_check_html,
                 'yoy_graph': 'yoy_returns.svg',
                 'mom_compare': 'mom_returns.svg'}

def performance2_html_generate(x, outfile='performance_breakdown.html'):
	env = Environment(loader=FileSystemLoader('.'))
	template = env.get_template("performance_breakdown_template.html")
	html_out = template.render(x)

	f = open(outfile,'w')
	f.write(html_out)
	f.close()


performance2_html_generate(template_vars)
#import pdfkit
#pdfkit.from_url('performance.html', 'test.pdf')

undl_perf = aj(MoM_dates, undl).apply(lambda x: x / x.shift() - 1, 0)[1:]
undl_perf = undl_perf.apply(np.mean, 0)

x = weight.apply(np.mean, 0)
x_label = [x.index[i] + ' - ' + str(100 * np.round(x.values[i], 4)) + '%' for i in range(len(x))]
y = x / sum(x)

range_colors = [min(-1.0 * abs(undl_perf)), max(abs(undl_perf))]

my_norm = matplotlib.colors.Normalize(range_colors[0], range_colors[1]) #this is how we will map our data

c1 = matplotlib.colors.LinearSegmentedColormap.from_list('mycm',['r','g'])
plt.pie(y, labels=x_label, shadow=False, radius=1.0, colors=c1(my_norm(undl_perf))); plt.show()



c2 = matplotlib.colors.LinearSegmentedColormap.from_list('mycm',['r','w','g'])
plt.pie(y, labels=x_label, shadow=False, radius=1.0, colors=c2(my_norm(undl_perf))); plt.show()
