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
from utils.common_function import rmse,mae,mape


if __name__ == '__main__':
    xgboost_model_save_path = os.path.join(model_save_path, "xgboost.m")
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=2371)
    for num,i in enumerate(silces):
        trip_i_path = os.path.join(trips_path,"{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        if(len(temp_values)<15):
            continue
        train_list_recover.append(temp_values)
        print("load %s" % str(i),num)
    train_list_recover = data_clean.delet_stopping_trips(train_list_recover)
    scaler_path = os.path.join(model_save_path,"scaler.m")
    train_x,train_y,test_x,test_y = data_clean.train_test_perpare(train_list_recover,scaler_model_path=scaler_path)
    Train = xgb.DMatrix(train_x, label=train_y)
    Test = xgb.DMatrix(test_x, label=np.array(test_y))
    params = ml.get_parameters_xgb()
    split = "training"
    if split == "training":
        xgbregressor = xgb.train(params, dtrain=Train, num_boost_round=50000,
                             evals=[(Train, "train"), (Test, "test")], early_stopping_rounds=1000)
        joblib.dump(xgbregressor, xgboost_model_save_path)
    else:
        xgbregressor = joblib.load(xgboost_model_save_path)
    preds = xgbregressor.predict(Test)
    endl_array = np.array([1]).repeat(test_y.size, axis=0)
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
    print("done")