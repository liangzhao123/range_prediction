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
from trainer.lightgbm_model import light_params
import lightgbm as lgb


if __name__ == '__main__':
    xgboost_1_model_save_path = os.path.join(model_save_path, "xgboost_1.m")
    xgboost_2_model_save_path = os.path.join(model_save_path, "xgboost_2.m")
    lightboost_1_model_save_path = os.path.join(model_save_path, "lightboost_1.m")
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=14)
    for num,i in enumerate(silces):
        trip_i_path = os.path.join(trips_path,"{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        if(len(temp_values)<15):
            continue
        train_list_recover.append(temp_values)
        print("load %s" % str(i),num)
    train_list_recover = data_clean.delet_stopping_trips(train_list_recover)
    scaler_path = os.path.join(model_save_path,"scaler.m")
    anchor_based = False
    if anchor_based ==  True:
        train_x,train_y,test_x,test_y,anchor_train,anchor_test = data_clean.train_test_perpare(train_list_recover,
                                                                                               scaler_model_path=scaler_path,
                                                                                               using_anchor_based=anchor_based)
        train_y = train_y - anchor_train
        test_y = test_y-anchor_test
    else:
        train_x, train_y, test_x, test_y= data_clean.train_test_perpare(train_list_recover,scaler_model_path=scaler_path)
    #xgboost data
    Train = xgb.DMatrix(train_x, label=train_y)
    Test = xgb.DMatrix(test_x, label=np.array(test_y))
    params = ml.get_parameters_xgb()
    #lightgbm data
    Train_lgb = lgb.Dataset(train_x, label=np.array(list(train_y)))
    Test_lgb = lgb.Dataset(test_x, label=np.array(test_y))
    light_parameters = light_params()
    split = "training"
    if split == "training":

        xgb_1 = xgb.train(params, dtrain=Train, num_boost_round=50000,
                             evals=[(Train, "train"), (Test, "test")], early_stopping_rounds=1000)
        joblib.dump(xgb_1, xgboost_1_model_save_path)
        lgbm_1 = lgb.train(params=light_parameters, train_set=Train_lgb,
                                   num_boost_round=50000, valid_sets=[Train_lgb, Test_lgb],
                                   early_stopping_rounds=1000)
        joblib.dump(lgbm_1, lightboost_1_model_save_path)
        output_1 = xgb_1.predict(Train)
        output_2 = lgbm_1.predict(train_x)
        features_for_second_layer = np.concatenate([output_1[...,np.newaxis],output_2[...,np.newaxis]],axis=1)
        xgb_2 = xgb.train(params, dtrain=Train, num_boost_round=50000,
                  evals=[(Train, "train"), (Test, "test")], early_stopping_rounds=1000)
        joblib.dump(lgbm_1, xgboost_2_model_save_path)
    else:
        xgb_1 = joblib.load(xgboost_1_model_save_path)
        xgb_2 = joblib.load(xgboost_2_model_save_path)
        lgbm_1 = joblib.load(lightboost_1_model_save_path)
    output_1 = xgb_1.predict(Test)
    output_2 = lgbm_1.predict(Test_lgb)
    features_concat = np.concatenate([output_1[...,np.newaxis],output_2[...,np.newaxis]],axis=1)
    features_concat = xgb.DMatrix(features_concat, label=test_y)
    preds = xgb_2.predict(features_concat)
    if anchor_based:
        preds = preds+ anchor_test
        test_y = test_y + anchor_test
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
