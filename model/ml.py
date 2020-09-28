import xgboost
import joblib
import os
root_path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/mile_estimator/tijiaocode"
def load_model():
    path = os.path.join(root_path,"origin_silce_plot/car_4/before_sample/xgboost.m")
    xgboost_regressor = joblib.load(path)
    path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/lgbm.m")
    lgbm_regressor = joblib.load(path)
    # path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/gradientboost.m")
    # gbdt_regressor = joblib.load("./origin_silce_plot/car_4/before_sample/gradientboost.m")
    scaler = joblib.load("./origin_silce_plot/car_4_scaler.m")
    return xgboost_regressor,lgbm_regressor,scaler
def get_parameters_xgb():
    params = {}
    params["objective"] = "reg:squarederror"
    params["eta"] = 0.05
    params["min_child_weight"] = 1.7817
    params["subsample"] = 0.5213
    params["max_depth"] = 3
    params["gamma"] = 0.0468
    params["colsample_bytree"] = 0.4603
    params["colsample_bylevel"] = 1
    params["colsample_bynode"] = 1
    params["lambda"] = 0.8571
    params["alpha"] = 0.4640
    params["tree_method"] = "exact"

    params["base_score"] = 0.5
    params["eval_metric"] = ["mae", "rmse"]
    return params
