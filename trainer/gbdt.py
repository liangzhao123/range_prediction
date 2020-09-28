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
from sklearn.ensemble import GradientBoostingRegressor

def get_model():
    gradientbr = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                           max_depth=3, max_features=0.5,
                                           min_samples_leaf=1, min_samples_split=2,
                                           min_weight_fraction_leaf=0,
                                           loss='huber', random_state=42, verbose=1, n_iter_no_change=200,
                                           subsample=0.5)
    return gradientbr



if __name__ == '__main__':
    gbdt_model_save_path = os.path.join(model_save_path, "gbdt.m")
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
    gbdt_model = get_model()

    split = "training"
    if split == "training":
        gbdt_model.fit(train_x,train_y)

        joblib.dump(gbdt_model, gbdt_model_save_path)
    else:
        gbdt_model = joblib.load(gbdt_model_save_path)
    preds = gbdt_model.predict(test_x)
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