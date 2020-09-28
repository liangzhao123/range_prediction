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
from sklearn.neural_network import MLPRegressor
from utils.common_function import rmse,mae,mape

class ANNs(object):
    def __init__(self,hide_layer_size=850,
                 actvation = "relu",
                 solver = "adam",
                 alpha = 0.0003,
                 learning_rate = "invscaling",
                 learning_rate_init = 0.001,
                 max_iter = 2000,
                 shuffle = True,
                 tol = 1e-4,
                 epsilon = 1e-8,
                 ):
        super().__init__()
        self.paramters = None
        self.model = MLPRegressor(hidden_layer_sizes=hide_layer_size,
                                  activation=actvation,
                                  solver=solver,
                                  alpha=alpha,
                                  batch_size="auto",
                                  learning_rate=learning_rate,
                                  learning_rate_init=learning_rate_init,
                                  max_iter=max_iter,
                                  shuffle=shuffle,
                                  tol=tol,
                                  verbose=True,
                                  early_stopping=True,
                                  validation_fraction=0.5,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon= epsilon,
                                  n_iter_no_change=100
                                  )
    def forward(self,data):
        x,y = data[0],data[1]
        self.model.fit(x,y)


if __name__ == '__main__':
    train_list_recover = []
    ANN_model_save_path = os.path.join(model_save_path, "ANNs.m")
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
    data = (train_x,train_y)
    split = "testing"    #changing to "training" can retraining the model
    if split == "training":
        ANN_model = ANNs()
        ANN_model.forward(data)
        joblib.dump(ANN_model.model, ANN_model_save_path)
    else:
        ANN_model = ANNs()
        ANN_model.model = joblib.load(ANN_model_save_path)
    preds = ANN_model.model.predict(test_x)
    endl_array = np.array([1]).repeat(test_y.size,axis=0)
    test_y= np.array(test_y)
    end_index = np.argwhere(np.array(test_y)>endl_array)
    test_y = test_y[end_index]
    preds = preds[end_index]
    score_rmse = rmse(preds,test_y)
    score_mae = mae(preds,test_y)
    score_mape = mape(preds,test_y)
    print("rmse-score:",score_rmse)
    print("mae-score:",score_mae)
    print("mape:",score_mape)
    print("done")
