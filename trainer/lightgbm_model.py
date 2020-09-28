import numpy as np
import pandas as pd
import os
import utils.vis_function as vis_function
from utils import  data_clean
import lightgbm as lgb
import model.ml as ml
trips_path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/mile_estimator/tijiaocode/origin_silce_plot/train_list_recover"
model_save_path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/models"
import joblib
from utils.common_function import rmse,mae,mape

def light_params():
    params = {}
    params["tast"] = "train"
    params["boosting_type"] = "gbdt"
    params["objective"] = "regression"
    params["metric"] = {"mae", "rmse"}
    params["num_leaves"] = 6
    params["eta"] = 0.05

    params["min_child_weight"] = 0.5
    params["bagging_fraction"] = 0.5
    params["bagging_freq"] = 1
    params['feature_fraction'] = 0.66
    params["max_bin"] = 200
    params["lambda_l2"] = 0.6571
    params["lambda_l1"] = 0.4640
    params["gamma"] = 0.0468
    params["verbose"] = 1
    return params

if __name__ == '__main__':
    lightboost_model_save_path = os.path.join(model_save_path, "lightboost.m")

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

    Train = lgb.Dataset(train_x, label=np.array(list(train_y)))
    Test = lgb.Dataset(test_x, label=np.array(test_y))

    params = light_params()
    split = "training"
    if split == "training":
        lgbm_regressor = lgb.train(params=params, train_set=Train,
                                    num_boost_round=50000, valid_sets=[Train,Test],
                                    early_stopping_rounds=1000)

        joblib.dump(lgbm_regressor, lightboost_model_save_path)
    else:
        lgbm_regressor = joblib.load(lightboost_model_save_path)
    preds = lgbm_regressor.predict(test_x)
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