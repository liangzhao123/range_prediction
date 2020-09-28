import numpy as np
import pandas as pd
import os
import utils.vis_function as vis_function
import utils.data_clean as  data_clean
import xgboost as xgb
import model.ml as ml
trips_path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/mile_estimator/tijiaocode/origin_silce_plot/train_list_recover"
model_save_path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/models"
import joblib
from sklearn.linear_model import ElasticNet
from utils.common_function import rmse,mae,mape

if __name__ == '__main__':
    ElesticNet_model_save_path = os.path.join(model_save_path, "ElasticNet.m")
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=2371)
    for num, i in enumerate(silces):
        trip_i_path = os.path.join(trips_path, "{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        if (len(temp_values) < 15):
            continue
        train_list_recover.append(temp_values)
        print("load %s" % str(i), num)
    train_list_recover = data_clean.delet_stopping_trips(train_list_recover)
    scaler_path = os.path.join(model_save_path, "scaler.m")
    train_x, train_y, test_x, test_y = data_clean.train_test_perpare(train_list_recover, scaler_model_path=scaler_path)
    x = pd.concat((train_x, test_x), axis=0)
    y = pd.concat((train_y, test_y), axis=0)
    data = (x, y)
    ElasticNet_model = ElasticNet(
        alpha= 1.0,
        l1_ratio=0.8,
        fit_intercept=True,
        max_iter=4000,
        tol=1e-6,
        positive=False,
    )
    split = "training"
    if split=="training":
        ElasticNet_model.fit(train_x,train_y)
        joblib.dump(ElasticNet_model,ElesticNet_model_save_path)
    else:
        ElasticNet_model = joblib.load(ElesticNet_model_save_path)
    preds = ElasticNet_model.predict(test_x)
    endl_array = np.array([10]).repeat(test_y.size, axis=0)
    test_y = np.array(test_y)
    end_index = np.argwhere(np.array(test_y) > endl_array)
    test_y = test_y[end_index]
    preds = preds[end_index]
    score_rmse = rmse(preds, test_y)
    score_mae = mae(preds, test_y)
    score_mape = mape(preds, test_y)
    print("rmse-score:", score_rmse)
    print("mae-score:", score_mae)
    print("mape:", score_mape)
    print("done")