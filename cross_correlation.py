
# coding: utf-8

# In[5]:

import math
import numpy as np
import pandas as pd
from scipy import stats
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
from bokeh.models.widgets import Panel, Tabs, Select
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox

# Data source in Excel
data_path = "D:/Python Projects/cross_correlation_of_shift_on_the_time_domain/cross_co-data.xlsx"

# Read the data from Excel with Pandas
df = pd.read_excel(data_path)

# Ankle data
x = np.array(df[[2]])
y = np.array(df[[3]])
x_ankle = x[~np.isnan(x)]
y_ankle = y[~np.isnan(y)]

# Knee data
x = np.array(df[[6]])
y = np.array(df[[7]])
x_knee = x[~np.isnan(x)]
y_knee = y[~np.isnan(y)]

# Hip data
x = np.array(df[[10]])
y = np.array(df[[11]])
x_hip = x[~np.isnan(x)]
y_hip = y[~np.isnan(y)]

# Crank data
x = np.array(df[[14]])
y = np.array(df[[15]])
x_crank = x[~np.isnan(x)]
y_crank = y[~np.isnan(y)]

# Define a function to find the K for R max
def get_K_for_maxR(x, y):
    # the mean of x, y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # N data points in X
    n = x.size

    # Calculate Cxx(0), Cyy(0)
    c_xx_0 = np.sum([(x[t] - x_mean)**2 for t in range(n)])
    c_yy_0 = np.sum([(y[t] - y_mean)**2 for t in range(n)])

    # Create a dict to store K & Rxy(K)
    k_r = {}

    # Calculate Cxy(K)
    for k in range(n+1):
        if k == 0:
            c_xy_k = np.sum([(x[t] - x_mean) * (y[t] - y_mean) for t in range(n)])
        elif k > 0:
            c_xy_k = np.sum([(x[t1] - x_mean) * (y[t1+k] - y_mean) for t1 in range(n-k)]) + np.sum( [(x[t2] - x_mean) * (y[t2-n+k] - y_mean ) for t2 in range(n)[n-k:]] )
        # Calculate Rxy(k)
        r_xy_k = round(c_xy_k / (c_xx_0 * c_yy_0)**0.5, 4)
        k_r[k] = r_xy_k

    # Get the max K and the max Rxy
    max_k, max_r = max(k_r.items(), key=lambda a: a[1])

    # Get 95% Confidence Interval
    h1 = 0.5 * np.log((1 + max_r) / (1 - max_r)) - 1.96 / np.sqrt(n - 3)
    h2 = 0.5 * np.log((1 + max_r) / (1 - max_r)) + 1.96 / np.sqrt(n - 3)
    ci_low = round((math.e**(2 * h1) - 1) / (math.e**(2 * h1) + 1), 4)
    ci_high = round((math.e**(2 * h2) - 1) / (math.e**(2 * h2) + 1), 4)

    # Find nearest Rxy and the corresponding K
    r_list = []
    k_list = []
    for ks, rs in k_r.items():
        k_list.append(ks)
        r_list.append(rs)

    # Define a function to find the two lowest K in a given CI
    def find_nearest_2(array, value):
        idx = (np.abs(array-value)).argmin()
        array2 = [a for a in array if a != array[idx]]
        idx2 = (np.abs(array2-value)).argmin()
        return (array[idx], array2[idx2])

    #r_ci_low = min(enumerate(r_list), key=lambda x: abs(x[1]-ci_low))[1]
    #r_ci_high = min(enumerate(r_list), key=lambda x: abs(x[1]-ci_high))[1]

    r_ci_low_1 = find_nearest_2(r_list, ci_low)[0]
    r_ci_low_2 = find_nearest_2(r_list, ci_low)[1]

    k_ci_left = [k for k, v in k_r.items() if v == r_ci_low_1 or v == -r_ci_low_1][0]
    k_ci_right = [k for k, v in k_r.items() if v == r_ci_low_2 or v == -r_ci_low_2][0]

    k_ci = (k_ci_left, k_ci_right)

    print("\tWhen K = " + str(max_k) + " , Rxy(K)max = " + str(max_r))
    print("\tCI = (" + str(ci_low) + ", " + str(ci_high) + ")")
    print("\tThe Corresponding Ks are: " + str(min(k_ci)) + " & " + str(max(k_ci)))

    output_notebook()

    x_lc_data = [idx for idx, mmt in enumerate(x)]
    y_lc_data = [mmt for idx, mmt in enumerate(x)]

    x_gc_data = [idx for idx, mmt in enumerate(y)]
    y_gc_data = [mmt for idx, mmt in enumerate(y)]

    p1 = figure(
        tools="pan,box_zoom,reset,save", # 显示工具栏
        x_range=[0, 360], # X轴坐标
        y_range=[min(min(y_gc_data), min(y_lc_data))*1.4, max(max(y_gc_data), max(y_lc_data))*1.2], # Y轴坐标
        title="Moment (NM) vs. Angle (degree)", # 图表标题
        x_axis_label="Angle (degree)", y_axis_label="Moment (NM)" # X和Y轴的标签
        )
    p1.line(x_gc_data, y_gc_data, legend="GC - Greater Cadence", line_color="blue", line_width=2)
    p1.line(x_lc_data, y_lc_data, legend="LC - Less Cadence", line_color="red", line_dash="4 4", line_width=2)
    p1.legend.location = "top_right"
    p1.title.align = 'center'
    p1.title.text_font_size = "14px"
    tab1 = Panel(child=p1, title="Moment (NM) vs. Angle (degree)")
    #show(p1)

    p2 = figure(
        tools="pan,box_zoom,reset,save", # 显示工具栏
        x_range=[min(k_ci), max(k_ci)],#[min(k_list), max(k_list)], # X轴坐标
        y_range=[r_ci_low_1, max_r*1.0005], # Y轴坐标
        title="Cross Correlation & K Shift", # 图表标题
        x_axis_label="k (degree)", y_axis_label="Cross Correlation (r)" # X和Y轴的标签
        )
    p2.line(k_list, r_list, line_color="blue", line_width=2)
    p2.title.align = 'center'
    p2.title.text_font_size = "14px"
    tab2 = Panel(child=p2, title="Cross Correlation & K Shift")
    #show(p2)
    tabs = Tabs(tabs=[tab1, tab2])
    show(tabs)


print("Ankle:")
get_K_for_maxR(x_ankle, y_ankle)
print("Knee:")
get_K_for_maxR(x_knee, y_knee)
print("Hip:")
get_K_for_maxR(x_hip, y_hip)
print("Crank:")
get_K_for_maxR(x_crank, y_crank)
