import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def evaluate(train_list_recover,xgboost_regressor,lgbm_regressor,scaler,i=32):
    # 2018.1.8,17：00-19：00
    # 2018.1.8,17：00-19：00

    trips = train_list_recover[32].copy()
    predict_xgb = []
    predict_lgb = []
    dt = []
    for i in range(int(np.round(len(trips) * 0.8, 0)), len(trips)):
        vsn = len(trips) * 0.8
        f_motor_power = np.polyfit(trips.loc[:i].used_soc, trips.loc[:i].motor_power, 1)
        f_total_power = np.polyfit(trips.loc[:i].used_soc, trips.loc[:i].total_power, 1)
        temp_max = trips.temp_max[i]
        temp_min = trips.temp_min[i]
        work_condition_0 = trips.work_condition_0[i]
        work_condition_1 = trips.work_condition_1[i]
        work_condition_2 = trips.work_condition_2[i]
        work_condition_3 = trips.work_condition_3[i]
        accelerate_ratio = trips.accelerate_ratio[i]
        f_driving_time = LinearRegression()
        f_driving_time.fit(trips.loc[:i][["used_soc", "brake_ratio", "stop_ratio"]], trips.loc[:i].driving_time)
        used_soc = trips.used_soc.max()
        val_motor_power = np.polyval(f_motor_power, used_soc)
        val_total_power = np.polyval(f_total_power, used_soc)
        brake_ratio = trips.brake_ratio[i]
        stop_ratio = trips.stop_ratio[i]
        time_features = pd.DataFrame({"used_soc": used_soc, "brake_ratio": brake_ratio, "stop_ratio": stop_ratio},
                                     index=range(1))
        driving_time = f_driving_time.predict(time_features)
        dt.append(f_driving_time.predict(time_features))
        temp_max = trips.temp_max[i]
        temp_min = trips.temp_min[i]
        temp_diff = trips.temp_diff[i]
        features = pd.DataFrame({"temp_max": temp_max,
                                 "temp_min": temp_min,
                                 "used_soc": used_soc,
                                 "driving_time": driving_time,
                                 "total_power": val_total_power,
                                 "motor_power": val_motor_power,
                                 "temp_diff": temp_diff,
                                 "soc_start": used_soc,
                                 "brake_ratio": brake_ratio,
                                 "stop_ratio": stop_ratio,
                                 "accelerate_ratio": accelerate_ratio,
                                 "work_condition_0": work_condition_0,
                                 "work_condition_1": work_condition_1,
                                 "work_condition_2": work_condition_2,
                                 "work_condition_3": work_condition_3}, index=range(1))
        train_X = features.drop(['brake_ratio', 'stop_ratio', 'accelerate_ratio',
                                 'work_condition_0', 'work_condition_1', 'work_condition_2',
                                 'work_condition_3'], axis=1)
        cat = features[['brake_ratio', 'stop_ratio', 'accelerate_ratio',
                        'work_condition_0', 'work_condition_1', 'work_condition_2',
                        'work_condition_3']]
        Feature_vector_normal = scaler.transform(train_X)
        Feature_vector_normal = pd.DataFrame(Feature_vector_normal, columns=train_X.columns)
        Feature_vector_normals = pd.concat([Feature_vector_normal, cat], axis=1)
        Feature_vector_xgb = xgb.DMatrix(Feature_vector_normals)
        predict_xgb.append(xgboost_regressor.predict(Feature_vector_xgb)[0] - trips.loc[:i].mile.max())
        predict_lgb.append(lgbm_regressor.predict(Feature_vector_normals)[0] - trips.loc[:i].mile.max())
    True_value = sorted(trips.loc[:len(trips) - int(np.round(len(trips) * 0.8, 0)) - 1].mile, reverse=True)
    error_xgb = predict_xgb - np.array(True_value)
    error_lgb = predict_lgb - np.array(True_value)
    blend_error = np.array(np.array(predict_xgb) + np.array(predict_lgb)) / 2 - np.array(True_value)
    return error_xgb,error_lgb,blend_error

if __name__ == '__main__':
    evaluate()

