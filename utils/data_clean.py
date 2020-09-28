
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from dateutil.parser import parse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
# from lightgbm import  LGBMRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from scipy import stats
import gc
from mlxtend.regressor import StackingCVRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import svm
from sklearn.datasets import make_moons,make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE,ADASYN
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def num_to_time(a):
    b = str(a)
    c = parse(b)
    return c


def asd(tra):
    tr = tra.copy()
    tr.time = tr.time.transform(lambda x: num_to_time(x))
    return tr.time


def rate_with_last(data, target):
    tr = data.copy()
    tr["rate"] = 0.0
    for i in range(1, len(target)):
        tr.rate[i] = round(target[i] - target[i - 1], 2)
    return tr.rate


def rate_with_next(data, target):
    tr = data.copy()
    tr["rate"] = 0.0
    for i in range(0, len(target) - 1):
        tr.rate[i] = round(target[i + 1] - target[i], 2)
    return tr.rate


def relative_value_zeng(target):
    tar = target.copy()
    inital = tar[0]
    tar = tar.transform(lambda x: x - inital)
    return tar


def relative_value_jian(target):
    tar = target.copy()
    inital = tar[0]
    tar = tar.transform(lambda x: inital - x)
    return tar


def initial_mileage_day(df):
    tr = df.copy()
    mileage_min_day = tr.groupby(["year", "month", "day"])["mileage"].min()
    mileage_max_day = tr.groupby(["year", "month", "day"])["mileage"].max()
    tr.mileage = tr.mileage.transform(lambda x: x.min())
    return tr


def trans_mileage(df):
    tr = df.copy()
    mile_day = []
    for i, m_d in zip(tr.groupby(["year", "month", "day"])["mileage"].min().index,
                      tr.groupby(["year", "month", "day"])["mileage"].min()):
        for j, m in zip(tr.datetime, tr.mileage):
            if i == (j.year, j.month, j.day):
                mile_day.append(m_d)
    return mile_day


def soc_max(df):  # 这求的是每天soc的最大值
    tr = df.copy()
    soc_day = []
    for i, m_d in zip(tr.groupby(["year", "month", "day"])["soc"].max().index,
                      tr.groupby(["year", "month", "day"])["soc"].max()):
        for j, m in zip(tr.datetime, tr.soc):
            if i == (j.year, j.month, j.day):
                soc_day.append(m_d)
    return soc_day


def soc_min(df):  # 这求的是每天soc的最小值
    tr = df.copy()
    soc_day_min = []
    for i, m_d in zip(tr.groupby(["year", "month", "day"])["soc"].min().index,
                      tr.groupby(["year", "month", "day"])["soc"].min()):
        for j, m in zip(tr.datetime, tr.soc):
            if i == (j.year, j.month, j.day):
                soc_day_min.append(m_d)
    return soc_day_min


def bian_soc(x):
    if 0 < x < 15:
        return "soc:0-14%"
    elif 15 <= x < 30:
        return "soc:15%-30%"
    elif 30 <= x < 40:
        return "soc:30%-40%"
    elif 40 <= x < 50:
        return "soc:40%-50%"
    elif 50 <= x < 60:
        return "soc:50%-60%"
    elif 60 <= x < 70:
        return "soc:60%-70%"
    elif 70 <= x < 80:
        return "soc:70%-80%"
    elif 80 <= x < 90:
        return "soc:80%-90%"
    elif 90 <= x <= 100:
        return "soc:90%-100%"


def test_1(df):
    df_ = df.copy()
    return df_.soc.transform(lambda x: bian_soc(x))


def wash(data):
    temp_index = data.query(
        "speed==0&total_voltage==0&total_current==0&soc==0&temp_max==0&temp_min==0&motor_voltage==0&motor_current==0").index
    data = data.drop(index=temp_index, axis=0)  # 停车数据
    return data


def time_interval(df):
    tr = df.copy()
    tr["seconds_now"] = 0
    for i in range(1, len(tr.datetime)):
        tr.seconds_now.values[i - 1] = (tr.year[i] * 365 * 24 * 60 * 60 +
                                        tr.month[i] * 30 * 24 * 60 * 60 +
                                        tr.day[i] * 24 * 60 * 60 +
                                        tr.hour[i] * 60 * 60 +
                                        tr.minute[i] * 60 +
                                        tr.second[i]) - (tr.year[i - 1] * 365 * 24 * 60 * 60 +
                                                         tr.month[i - 1] * 30 * 24 * 60 * 60 +
                                                         tr.day[i - 1] * 24 * 60 * 60 +
                                                         tr.hour[i - 1] * 60 * 60 +
                                                         tr.minute[i - 1] * 60 +
                                                         tr.second[i - 1])
    return tr.seconds_now


def seconds_leijia(df):
    tr = df.copy()
    tr["seconds_now"] = 0
    for i in range(0, len(tr.datetime)):
        tr.seconds_now.values[i] = (tr.year[i] * 365 * 24 * 60 * 60 +
                                    tr.month[i] * 30 * 24 * 60 * 60 +
                                    tr.day[i] * 24 * 60 * 60 +
                                    tr.hour[i] * 60 * 60 +
                                    tr.minute[i] * 60 +
                                    tr.second[i])
    return tr.seconds_now


def cut_point(mowei_index):
    left = []
    right = []
    mowei_index.insert(0, 0)
    for index in range(0, len(mowei_index) - 1):
        left.append(mowei_index[index])
        right.append(mowei_index[index + 1])
    return left, right


def reset_index(df_list):
    for i in range(len(df_list)):
        df_list[i].reset_index(inplace=True)
        df_list[i].drop(["index"], axis=1, inplace=True)

    return df_list


def mile_trans(df):
    tr = df.copy()
    for i in range(len(tr)):
        initial = tr[i].mile[0]

        tr[i].mile = tr[i].mile.transform(lambda x: x - initial)
    return tr


def time_trans(df):
    tr = df.copy()
    for i in range(len(tr)):
        initial = tr[i].driving_time[0]

        tr[i].driving_time = tr[i].driving_time.transform(lambda x: x - initial)
    return tr


def norm_plot(data, color, name):
    mean = data.mean()
    sigma = data.std()
    df = pd.DataFrame()
    df["data"] = data
    y_probility = stats.norm.pdf(data, mean, sigma)
    df["y_probility"] = y_probility
    df = df.sort_values(by=["data"])
    plt.grid()
    ax = plt.plot(df.data, df.y_probility, linewidth=2, color=color, label=name)
    return ax


def cut_silce_by_soc(data):
    mo_wei_index = data.query("soc_rate>5|soc_rate<-10").index
    mo_wei_index = list(mo_wei_index)
    if len(mo_wei_index) == 0:
        print("error")
    if mo_wei_index[-1] != data.index[-1]:
        mo_wei_index.append(data.index[-1])
    if mo_wei_index[0] != 0:
        mo_wei_index.insert(0, 0)
    left = []
    right = []
    for index in range(0, len(mo_wei_index) - 1):
        left.append(mo_wei_index[index])
        right.append(mo_wei_index[index + 1])
    data_list = []
    for z, y in zip(left, right):
        if y == right[-1]:
            data_list.append(data.loc[z:y - 1, :])
        else:
            data_list.append(data.loc[z:y - 1, :])
    data_list = reset_index(data_list)
    return data_list


def cut_silce_by_time(data):
    mo_wei_index = data.query("time_interval>100").index
    mo_wei_index = list(mo_wei_index)
    if len(mo_wei_index) == 0:
        print("error")
    if mo_wei_index[-1] != data.index[-1]:
        mo_wei_index.append(data.index[-1])
    if mo_wei_index[0] != 0:
        mo_wei_index.insert(0, 0)
    left = []
    right = []
    for index in range(0, len(mo_wei_index) - 1):
        left.append(mo_wei_index[index])
        right.append(mo_wei_index[index + 1])
    data_list = []
    for z, y in zip(left, right):
        if y == right[-1]:
            data_list.append(data.loc[z:y, :])
        else:
            data_list.append(data.loc[z:y - 1, :])
    data_list = reset_index(data_list)
    return data_list


def cut_silce_by_mile(data):
    mo_wei_index = data.query("mile_rate>1").index
    mo_wei_index = list(mo_wei_index)
    if len(mo_wei_index) == 0:
        print("error")
    if mo_wei_index[-1] != data.index[-1]:
        mo_wei_index.append(data.index[-1])
    if mo_wei_index[0] != 0:
        mo_wei_index.insert(0, 0)
    left = []
    right = []
    for index in range(0, len(mo_wei_index) - 1):
        left.append(mo_wei_index[index])
        right.append(mo_wei_index[index + 1])
    data_list = []
    for z, y in zip(left, right):
        if y == right[-1]:
            data_list.append(data.loc[z:y, :])
        else:
            data_list.append(data.loc[z:y - 1, :])
    data_list = reset_index(data_list)
    return data_list


def cut_silce_by_mile_soc_time(data):
    mo_wei_index_mile = list(data.query("mile_rate>1").index)
    mo_wei_index_time = list(data.query("time_interval>100").index)
    mo_wei_index_soc = list(data.query("soc_rate>5|soc_rate<-10").index)
    mo_wei_index = mo_wei_index_time + mo_wei_index_mile + mo_wei_index_soc
    mo_wei_index = list(set(mo_wei_index))
    mo_wei_index.sort()
    if len(mo_wei_index) == 0:
        print("error")
    if mo_wei_index[-1] != data.index[-1]:
        mo_wei_index.append(data.index[-1])
    if mo_wei_index[0] != 0:
        mo_wei_index.insert(0, 0)
    left = []
    right = []
    for index in range(0, len(mo_wei_index) - 1):
        left.append(mo_wei_index[index])
        right.append(mo_wei_index[index + 1])
    data_list = []
    for z, y in zip(left, right):
        if z == left[0]:
            data_list.append(data.loc[z:y, :])
        else:
            data_list.append(data.loc[z + 1:y, :])
    data_list = reset_index(data_list)
    return data_list


def main_fast_1(train_0):
    train_0 = train_0.drop_duplicates(subset=["time"], keep="first")  # 去掉重复时间的数据
    train_0 = train_0.sort_values(by=["time"])  # 首先按照时间排序
    train_0["datetime"] = asd(train_0)  # 创建时间格式的时间
    train_0 = train_0.reset_index().drop(["index"], axis=1)  # 对于index重新排列
    train_0["year"] = [t.year for t in pd.DatetimeIndex(train_0.datetime)]  # 创建年月日
    train_0["month"] = [t.month for t in pd.DatetimeIndex(train_0.datetime)]
    train_0["day"] = [t.day for t in pd.DatetimeIndex(train_0.datetime)]
    train_0["hour"] = [t.hour for t in pd.DatetimeIndex(train_0.datetime)]
    train_0["minute"] = [t.minute for t in pd.DatetimeIndex(train_0.datetime)]
    train_0["second"] = [t.second for t in pd.DatetimeIndex(train_0.datetime)]
    train_0.time = seconds_leijia(train_0)
    train_0 = wash(train_0)
    train_0.reset_index(inplace=True)
    train_0.drop(["index"], axis=1, inplace=True)
    train_0["time_interval"] = time_interval(train_0)

    train_0["soc_rate"] = rate_with_next(train_0, train_0.soc)

    train_0["mile_rate"] = rate_with_next(train_0, train_0.mileage)

    return train_0


def remain_fast_1(train_0):
    train_0.reset_index(inplace=True)
    train_0.drop(["index"], axis=1, inplace=True)
    train_0["time_interval"] = time_interval(train_0)

    train_0["soc_rate"] = rate_with_next(train_0, train_0.soc)

    train_0["mile_rate"] = rate_with_next(train_0, train_0.mileage)

    return train_0


def main_fast_2(train_0):
    train_list_0 = cut_silce_by_mile_soc_time(train_0)

    return train_list_0


def trans(x):
    if x < 0:
        return x * 0.5
    else:
        return x


def trans_2(x):
    if x > 0:
        return 0
    else:
        return x


def main_fast_3(data_list):
    for i in range(len(data_list)):
        data_list[i]["soc_rate"] = rate_with_next(data_list[i], data_list[i].soc)

        temp_values = data_list[i].soc_rate.transform(lambda x: trans_2(x))
        data_list[i]["used_soc"] = np.abs(temp_values.cumsum())

        data_list[i]["mile"] = relative_value_zeng(data_list[i].mileage)
        data_list[i]["driving_time"] = relative_value_zeng(data_list[i].time)
        data_list[i]["motor_voltage_rate"] = rate_with_next(data_list[i], data_list[i].motor_voltage)
        data_list[i]["motor_current_rate"] = rate_with_next(data_list[i], data_list[i].motor_current)
        data_list[i]["total_voltage_rate"] = rate_with_next(data_list[i], data_list[i].total_voltage)
        data_list[i]["total_current_rate"] = rate_with_next(data_list[i], data_list[i].total_current)
        data_list[i]["mile_rate"] = rate_with_next(data_list[i], data_list[i].mileage)

        data_list[i].time_interval[0] = 10

        data_list[i]["total_power_transient"] = (data_list[i].total_current * data_list[i].total_voltage)
        data_list[i]["motor_power_transient"] = data_list[i].motor_current * data_list[i].motor_voltage
        data_list[i]["total_power"] = data_list[i].total_power_transient.transform(lambda x: trans(x))
        data_list[i]["motor_power"] = data_list[i].motor_power_transient.transform(lambda x: trans(x))

        data_list[i].total_power = data_list[i].total_power.cumsum()
        data_list[i].motor_power = data_list[i].motor_power.cumsum()

        data_list[i]["temp_diff"] = data_list[i].temp_max - data_list[i].temp_min

        # 删除片段中的停车数据

        data_list[i]["soc_start"] = data_list[i].soc[0]

    return data_list


def main_fast_4(data_list):
    for i in range(len(data_list)):
        data_list[i]["motor_power_per_mile"] = data_list[i].motor_power.max() / float(data_list[i].mile.max())
        data_list[i]["total_power_per_mile"] = data_list[i].total_power.max() / float(data_list[i].mile.max())
        data_list[i]["motor_power_per_soc"] = data_list[i].motor_power.max() / float(data_list[i].used_soc.max())
        data_list[i]["total_power_per_soc"] = data_list[i].total_power.max() / float(data_list[i].used_soc.max())

    return data_list


def wash_2(data):
    # 充电数据
    temp_index = data.query("mile_rate==0&motor_voltage==0&motor_current==0&total_current<0&speed==0.0").index
    data.drop(index=temp_index, axis=0, inplace=True)
    data.reset_index(inplace=True)
    data.drop(["index"], axis=1, inplace=True)
    # 删除的以下数据是充满电之后停车数据
    temp_index = data.query("motor_voltage==0&mile_rate<=0.0&speed==0&total_current==0&motor_current==0").index
    data.drop(index=temp_index, axis=0, inplace=True)
    data.reset_index(inplace=True)
    data.drop(["index"], axis=1, inplace=True)
    # 开车之前或者是用车结束关机时有些数据已经为0，有些则没有
    temp_index = data.query(
        "mile_rate<=0.0&speed==0&total_voltage==0&total_current==0&motor_current==0&soc==0&temp_max==0&temp_min==0").index
    data.drop(index=temp_index, axis=0, inplace=True)
    data.reset_index(inplace=True)
    data.drop(["index"], axis=1, inplace=True)
    return data


"""Outlier Detect and correction"""


def outlier_justify(data):
    temp_index = data.query("soc==0").index
    downs = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i + j not in temp_index:
                downs.append(i + j)
                break
    ups = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i - j not in temp_index:
                ups.append(i - j)
                break
    for up, down, now in zip(ups, downs, temp_index):
        if np.abs(data.time[up] - data.time[now]) >= np.abs(data.time[down] - data.time[now]):
            data.soc[now] = data.soc[down]
        else:
            data.soc[now] = data.soc[up]

    temp_index = data.query("total_voltage==0").index
    downs = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i + j not in temp_index:
                downs.append(i + j)
                break
    ups = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i - j not in temp_index:
                ups.append(i - j)
                break
    for up, down, now in zip(ups, downs, temp_index):
        if np.abs(data.time[up] - data.time[now]) >= np.abs(data.time[down] - data.time[now]):
            data.total_voltage[now] = data.total_voltage[down]
        else:
            data.total_voltage[now] = data.total_voltage[up]

    temp_index = data.query("temp_max==0").index
    downs = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i + j not in temp_index:
                downs.append(i + j)
                break
    ups = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i - j not in temp_index:
                ups.append(i - j)
                break
    for up, down, now in zip(ups, downs, temp_index):
        if np.abs(data.time[up] - data.time[now]) >= np.abs(data.time[down] - data.time[now]):
            data.temp_max[now] = data.temp_max[down]
        else:
            data.temp_max[now] = data.temp_max[up]

    temp_index = data.query("temp_min==0").index
    downs = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i + j not in temp_index:
                downs.append(i + j)
                break
    ups = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i - j not in temp_index:
                ups.append(i - j)
                break
    for up, down, now in zip(ups, downs, temp_index):
        if np.abs(data.time[up] - data.time[now]) >= np.abs(data.time[down] - data.time[now]):
            data.temp_min[now] = data.temp_min[down]
        else:
            data.temp_min[now] = data.temp_min[up]

    temp_index = data.query("motor_voltage==0").index
    downs = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i + j not in temp_index:
                downs.append(i + j)
                break
    ups = []
    for i in temp_index:
        for j in range(1, 10000, 1):
            if i - j not in temp_index:
                ups.append(i - j)
                break
    for up, down, now in zip(ups, downs, temp_index):
        if np.abs(data.time[up] - data.time[now]) >= np.abs(data.time[down] - data.time[now]):
            data.motor_voltage[now] = data.motor_voltage[down]
        else:
            data.motor_voltage[now] = data.motor_voltage[up]

    return data


def cut_silce_by_mile_for_take_silce(data):
    mo_wei_index = data.query("mile_rate<0").index
    mo_wei_index = list(mo_wei_index)
    if len(mo_wei_index) == 0:
        print("error")
    if mo_wei_index[-1] != data.index[-1]:
        mo_wei_index.append(data.index[-1])
    if mo_wei_index[0] != 0:
        mo_wei_index.insert(0, 0)
    left = []
    right = []
    for index in range(0, len(mo_wei_index) - 1):
        left.append(mo_wei_index[index])
        right.append(mo_wei_index[index + 1])
    data_list = []
    for z, y in zip(left, right):
        if z == left[0]:
            data_list.append(data.loc[z:y, :])
        else:
            data_list.append(data.loc[z + 1:y, :])
    temp_index = []
    for i, x in enumerate(data_list):
        if len(x) <= 10:
            temp_index.append(i)
    data_list = [data_list[i] for i in range(len(data_list)) if (i not in temp_index)]
    data_list = reset_index(data_list)
    return data_list


def time_change(data_list):
    for i, x in enumerate(data_list):
        x["time_interval"] = rate_with_next(x, x.time)
    return data_list


def intercept_missing_data(data):
    interval = 10
    for i, x in enumerate(data):
        input_index = []
        for j in range(len(x)):
            if np.round(x.time_interval[j] / interval, 0) > 1:
                num = int(np.round(x.time_interval[j] / interval, 0) - 1)
                input_index.append((j, num))
        iters = 0
        for k in input_index:
            input_data = pd.DataFrame(np.zeros((k[1], len(x.columns))), columns=x.columns)
            data[i] = data[i].iloc[:k[0] + 1 + iters].append(input_data, ignore_index=True).append(
                data[i].iloc[k[0] + 1 + iters:], ignore_index=True)
            iters += k[1]
    return data


def to_nan(data):
    for i in range(len(data)):
        temp_index = data[i].query("mileage==0").index
        for j in temp_index:
            data[i].loc[j] = np.nan
    return data


def linear_difference(data):
    for i in range(len(data)):
        data[i].time = data[i].time.interpolate()
        data[i].speed = data[i].speed.interpolate()
        data[i].total_voltage = data[i].total_voltage.interpolate()
        data[i].total_current = data[i].total_current.interpolate()
        data[i].soc = data[i].soc.interpolate()
        data[i].temp_max = data[i].temp_max.interpolate()
        data[i].temp_min = data[i].temp_min.interpolate()
        data[i].motor_voltage = data[i].motor_voltage.interpolate()
        data[i].motor_current = data[i].motor_current.interpolate()
        data[i].mileage = data[i].mileage.interpolate()
    return data


def round_float(data):
    for i in range(len(data)):
        data[i].time_interval = np.round(data[i].time_interval, 0)
    return data


def to_brakes(x):
    if x < 0:
        return 1
    else:
        return 0


def to_stops(x):
    if x == 0:
        return 1
    else:
        return 0


def to_accs(x):
    if x > 0:
        return 1
    else:
        return 0


def work_condition_feature(data):
    for i, x in enumerate(data):
        total_nums = np.array(list(range(1, len(data[i]) + 1, 1)))
        data[i]["brake_ratio"] = data[i].motor_current
        data[i]["stop_ratio"] = data[i].motor_current
        data[i]["accelerate_ratio"] = data[i].motor_current
        data[i].brake_ratio = data[i].brake_ratio.transform(lambda x: to_brakes(x))
        data[i].stop_ratio = data[i].stop_ratio.transform(lambda x: to_stops(x))
        data[i].accelerate_ratio = data[i].accelerate_ratio.transform(lambda x: to_accs(x))
        data[i].brake_ratio = data[i].brake_ratio.cumsum() / total_nums
        data[i].stop_ratio = data[i].stop_ratio.cumsum() / total_nums
        data[i].accelerate_ratio = data[i].accelerate_ratio.cumsum() / total_nums
    return data


def cut_silce_by_point(data, point):
    mo_wei_index = point
    mo_wei_index = list(mo_wei_index)
    if len(mo_wei_index) == 0:
        print("error")
        return 0
    if mo_wei_index[-1] != data.index[-1]:
        mo_wei_index.append(data.index[-1])
    if mo_wei_index[0] != 0:
        mo_wei_index.insert(0, 0)
    left = []
    right = []
    for index in range(0, len(mo_wei_index) - 1):
        left.append(mo_wei_index[index])
        right.append(mo_wei_index[index + 1])
    data_list = []
    for z, y in zip(left, right):
        if z == left[0]:
            data_list.append(data.loc[z:y, :])
        else:
            data_list.append(data.loc[z + 1:y, :])
    data_list = reset_index(data_list)
    return data_list


def create_work_condition_percentage(data):
    for i in range(len(data)):
        nums = np.array(list(range(1, len(data[i]) + 1)))
        data[i].work_condition_0 = (data[i].work_condition_0.cumsum()) / nums
        data[i].work_condition_1 = data[i].work_condition_1.cumsum() / nums
        data[i].work_condition_2 = data[i].work_condition_2.cumsum() / nums
        data[i].work_condition_3 = data[i].work_condition_3.cumsum() / nums
    return data


def delet_stopping_trips(train_list_recover):
    temp_index = []
    for i in range(len(train_list_recover)):
        if train_list_recover[i].speed.max() == 0:
            temp_index.append(i)
    train_list_recover = [train_list_recover[i] for i in range(len(train_list_recover)) if (i not in temp_index)]
    return train_list_recover

def train_test_perpare(train_list_recover,scaler_model_path,using_anchor_based = True):
    test_index = np.random.randint(0,2371,size=1000)

    train_list = [train_list_recover[i] for i in range(len(train_list_recover)) if (i not in test_index)]


    train = pd.concat(train_list, axis=0)
    train.reset_index(inplace=True)
    train.drop(["index"], axis=1, inplace=True)
    train.reset_index(inplace=True)
    train.drop(["index"], axis=1, inplace=True)
    train.drop(['time', 'speed', 'total_voltage', 'total_current', 'soc', 'soc_rate',
                            'motor_voltage', 'motor_current', 'mileage', 'datetime',
                            'year', 'month', 'day', 'hour', 'minute', 'second', 'time_interval', 'mile_rate',
                            'motor_voltage_rate', 'motor_current_rate', 'total_voltage_rate',
                            'total_current_rate', 'total_power_transient', 'motor_power_transient',
                            'cut_point', 'work_condition',
                            'work_condition_color', ], axis=1, inplace=True)
    temp_index = train.query("temp_diff<-10").index
    train.drop(index=temp_index, axis=1, inplace=True)
    train.reset_index(inplace=True)
    train.drop(["index"], axis=1, inplace=True)

    train_X, train_y = train.drop(['brake_ratio', 'stop_ratio', 'accelerate_ratio',
                                               'work_condition_0', 'work_condition_1', 'work_condition_2',
                                               'work_condition_3', 'mile', ], axis=1), train.mile

    if using_anchor_based:
        anchor_train = 1.6775 * train_X.used_soc


    scaler = StandardScaler()
    scaler.fit(train_X)
    Train_X = scaler.transform(train_X)
    joblib.dump(scaler, scaler_model_path)

    Train_X = pd.DataFrame(Train_X, columns=train_X.columns)
    Train_X = pd.concat([Train_X, train[['brake_ratio', 'stop_ratio', 'accelerate_ratio',
                                                     'work_condition_0', 'work_condition_1', 'work_condition_2',
                                                     'work_condition_3']]], axis=1)

    # 测试集
    test_list = [train_list_recover[i] for i in range(len(train_list_recover)) if (i in test_index)]
    # temp_index = []
    # for i in range(len(test_list)):
    #     if (test_list[i].shape[0] == 0) | (test_list[i].loc[len(test_list[i]) - 1:].mile.values[0] <= 5):
    #         temp_index.append(i)
    # test_list = [test_list[i] for i in range(len(test_list)) if (i not in temp_index)]
    #
    # test_features_1_list = []
    # for i, x in enumerate(test_list):
    #     if test_list[i].loc[len(test_list[i]) - 1:].mile.values[0] > 5:
    #         input_data = test_list[i].loc[len(test_list[i]) - 1:].drop(
    #             ['time', 'speed', 'total_voltage', 'total_current', 'soc', 'soc_rate',
    #              'motor_voltage', 'motor_current', 'mileage', 'datetime',
    #              'year', 'month', 'day', 'hour', 'minute', 'second', 'time_interval', 'mile_rate',
    #              'motor_voltage_rate', 'motor_current_rate', 'total_voltage_rate',
    #              'total_current_rate', 'total_power_transient', 'motor_power_transient',
    #              'cut_point', 'work_condition',
    #              'work_condition_color', "mile", 'brake_ratio', 'stop_ratio', 'accelerate_ratio',
    #              'work_condition_0', 'work_condition_1', 'work_condition_2',
    #              'work_condition_3'], axis=1)
    #         test_features_1_list.append(scaler.transform(input_data))
    #
    # test_features_1 = np.concatenate(test_features_1_list, axis=0)
    # test_features_1 = pd.DataFrame(test_features_1, columns=train_X.columns)
    #
    # test_features_2_list = []
    # for i, x in enumerate(test_list):
    #     if test_list[i].loc[len(test_list[i]) - 1:].mile.values[0] > 5:
    #         input_data = test_list[i].loc[len(test_list[i]) - 1:][['brake_ratio', 'stop_ratio', 'accelerate_ratio',
    #                                                      'work_condition_0', 'work_condition_1', 'work_condition_2',
    #                                                      'work_condition_3']]
    #         test_features_2_list.append(input_data)
    # test_features_2 = pd.concat(test_features_2_list, axis=0)
    # test_features_2.reset_index(inplace=True)
    # test_features_2.drop(["index"], axis=1, inplace=True)
    #
    #
    # test_features = pd.concat([test_features_1, test_features_2], axis=1)
    #
    # #测试集合标签
    # test_label_final_data = []
    # for i, x in enumerate(test_list):
    #     if test_list[i].loc[len(test_list[i]) - 1:].mile.values[0] > 5:
    #         input_label = test_list[i].loc[len(test_list[i]) - 1:].mile.values[0]
    #         test_label_final_data.append(input_label)

    #测试集计算2
    test_input = []
    anchor_test_list = []
    for i, x in enumerate(test_list):
        if test_list[i].loc[len(test_list[i]) - 1:].mile.values[0] > 5:
            input_data = test_list[i].drop(['time', 'speed', 'total_voltage', 'total_current', 'soc', 'soc_rate',
                                       'motor_voltage', 'motor_current', 'mileage', 'datetime',
                                       'year', 'month', 'day', 'hour', 'minute', 'second', 'time_interval', 'mile_rate',
                                       'motor_voltage_rate', 'motor_current_rate', 'total_voltage_rate',
                                       'total_current_rate', 'total_power_transient', 'motor_power_transient',
                                       'cut_point', 'work_condition',
                                       'work_condition_color', "mile", 'brake_ratio', 'stop_ratio', 'accelerate_ratio',
                                       'work_condition_0', 'work_condition_1', 'work_condition_2',
                                       'work_condition_3'], axis=1)
            feature_name = input_data.columns
            if using_anchor_based:
                anchor_test_list.append(input_data.used_soc*1.6775)
            input_data = pd.DataFrame(scaler.transform(input_data), columns=feature_name)
            input_data = pd.concat([input_data, test_list[i][['brake_ratio', 'stop_ratio', 'accelerate_ratio',
                                                         'work_condition_0', 'work_condition_1', 'work_condition_2',
                                                         'work_condition_3']]], axis=1)
            test_input.append(input_data)
    anchor_test = np.concatenate(anchor_test_list,axis=0)
    test_features = pd.concat(test_input, axis=0)
    test_features.reset_index(inplace=True)
    test_features.drop(["index"], axis=1, inplace=True)
    test_label_list = []
    for i, x in enumerate(test_list):
        if test_list[i].loc[len(test_list[i]) - 1:].mile.values[0] > 5:
            input_label = test_list[i].mile
            test_label_list.append(input_label)
    test_label = pd.concat(test_label_list, axis=0)


    if using_anchor_based:
        return Train_X,train_y,test_features,test_label,anchor_train,anchor_test
    else:
        return Train_X, train_y, test_features, test_label

if __name__ == '__main__':
    pass



