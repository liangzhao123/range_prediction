import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

subplot_fig_size=(20,30)
subplot_xticks_size=20
subplot_yticks_size=20
subplot_xlabel_size=20
subplot_ylabel_size=20
subplot_title_size=20
subplot_legend_size=20

fig_size = (8, 6)
xticks_size = 20
yticks_size = 20
xlabel_size = 20
ylabel_size = 20
title_size = 20
legend_size = 20

def plot_features_vs_range_a_trip(train_list_recover,save_path):
    for i in range(len(train_list_recover)):
        train_list_recover[i].drop(['time', 'speed', 'total_voltage', 'total_current', 'soc', 'soc_rate',
                                    'motor_voltage', 'motor_current', 'mileage', 'datetime',
                                    'year', 'month', 'day', 'hour', 'minute', 'second', 'time_interval', 'mile_rate',
                                    'motor_voltage_rate', 'motor_current_rate', 'total_voltage_rate',
                                    'total_current_rate', 'total_power_transient', 'motor_power_transient',
                                    'cut_point', 'work_condition',
                                    'work_condition_color'], axis=1, inplace=True)
    subplot_fig_size = (20, 30)
    alphabets = range(len(train_list_recover[0].columns))
    plt.figure(figsize=subplot_fig_size)
    # plt.suptitle("Add more than 120 trips", y=1.01, fontsize=30)
    for j in range(1, len(train_list_recover[0].columns) + 1):
        ax = plt.subplot(7, 4, j)
        for i, x in enumerate(train_list_recover):
            ax.plot(x.mile, x.iloc[:, j - 1])
        feature_name = change_features_name(x.columns[j - 1])
        plt.ylabel("{}".format(feature_name), fontsize=subplot_ylabel_size)
        plt.xlabel("Driving distance [km] \n   (%d)" % j, fontsize=subplot_xlabel_size)
        plt.xticks(fontsize=subplot_xticks_size)
        plt.yticks(fontsize=subplot_yticks_size)
        plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return 0

def plot_features_vs_range(train_list_recover,save_path):
    for i in range(len(train_list_recover)):
        train_list_recover[i].drop(['time', 'speed', 'total_voltage', 'total_current', 'soc', 'soc_rate',
                                    'motor_voltage', 'motor_current', 'mileage', 'datetime',
                                    'year', 'month', 'day', 'hour', 'minute', 'second', 'time_interval', 'mile_rate',
                                    'motor_voltage_rate', 'motor_current_rate', 'total_voltage_rate',
                                    'total_current_rate', 'total_power_transient', 'motor_power_transient',
                                    'cut_point', 'work_condition',
                                    'work_condition_color'], axis=1, inplace=True)
    subplot_fig_size = (20, 30)
    alphabets = range(len(train_list_recover[0].columns))
    plt.figure(figsize=subplot_fig_size)
    # plt.suptitle("Add more than 120 trips", y=1.01, fontsize=30)
    for j in range(1, len(train_list_recover[0].columns) + 1):
        ax = plt.subplot(7, 4, j)
        for i, x in enumerate(train_list_recover):
            ax.plot(x.mile, x.iloc[:, j - 1])
        feature_name = change_features_name(x.columns[j - 1])
        plt.ylabel("{}".format(feature_name), fontsize=subplot_ylabel_size)
        plt.xlabel("Driving distance [km] \n   (%d)" % j, fontsize=subplot_xlabel_size)
        plt.xticks(range(0, 150, 30), fontsize=subplot_xticks_size)
        plt.yticks(fontsize=subplot_yticks_size)
        plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return 0

def change_features_name(feature_name):
    if feature_name=="total_power":
        return "COEB  [J]"
    elif feature_name=="temp_max":
        return "Temp_max  [Celsius]"
    elif feature_name=="temp_min":
        return "Temp_min  [Celsius]"
    elif feature_name=="temp_diff":
        return "Temp_diff  [Celsius]"
    elif feature_name=="motor_power":
        return "COEM  [J]"
    elif feature_name== "mile":
        return "Driving distance  [km]"
    elif feature_name=="soc_start":
        return "SOC_start  [%]"
    elif feature_name=="brake_ratio":
        return "BR  [%]"
    elif feature_name=="stop_ratio":
        return "SR  [%]"
    elif feature_name=="accelerate_ratio":
        return "AR  [%]"
    elif feature_name=="work_condition_0":
        return "RDP_1  [%]"
    elif feature_name=="work_condition_1":
        return "RDP_2  [%]"
    elif feature_name=="work_condition_2":
        return "RDP_3  [%]"
    elif feature_name=="work_condition_3":
        return "RDP_4  [%]"
    elif feature_name=="used_soc":
        return "Used_SOC  [%]"
    else:
        return feature_name

def plot_COEM_vs_Used_soc(train_list_recover):
    # 画出电机输出能量和SOC使用量之间的关系

    plt.figure(figsize=fig_size)
    for i in range(len(train_list_recover)):
        plt.plot(train_list_recover[i].used_soc, train_list_recover[i].motor_power)
        plt.xticks(fontsize=xticks_size)
        plt.yticks(fontsize=yticks_size)
        # plt.title("Used_SOC & Motor power", fontsize=title_size)
        plt.xlabel("Used_SOC [%]", fontsize=xlabel_size)
        plt.ylabel("COEM  [J]", fontsize=ylabel_size)
    plt.savefig("/home/liang/range_prediction/output/figures/COEM_vs_Used_SOC.jpg", dpi=600, bbox_inches="tight")

def plot_COMB_VS_Used_SOC(train_list_recover):
    # 画出电机输出能量和SOC使用量之间的关系
    plt.figure(figsize=fig_size)
    for i in range(len(train_list_recover)):
        plt.plot(train_list_recover[i].used_soc, train_list_recover[i].total_power)
        plt.xticks(fontsize=xticks_size)
        plt.yticks(fontsize=yticks_size)
        # plt.title("Used_soc & Total_power", fontsize=title_size)
        plt.xlabel("Used_SOC  [%]", fontsize=xlabel_size)
        plt.ylabel("COMB  [J]", fontsize=ylabel_size)
    plt.savefig("/home/liang/range_prediction/output/figures/COEB_vs_Used_SOC.jpg", dpi=600, bbox_inches="tight")

def plot_silce_COME_vs_SOC(train_list_recover):

    # 特征预测的举例说明：电机能量预测
    trips = train_list_recover[2]
    plt.figure(figsize=fig_size)
    f_motor_power = np.polyfit(trips.used_soc, trips.motor_power, 1)
    val_motor_power = np.polyval(f_motor_power, range(0, int(trips.soc[0])))
    plt.plot(range(int(trips.soc[0]), 0, -1), val_motor_power, label="Linear Model")
    plt.plot(trips.soc, trips.motor_power, label="True value")
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.title("SOC & Motor_power", fontsize=title_size)
    plt.xlabel("SOC  [%]", fontsize=xlabel_size)
    plt.ylabel("COEM  [J]", fontsize=ylabel_size)
    plt.legend(fontsize=legend_size)
    plt.savefig("/home/liang/range_prediction/output/figures/COME_prediction.jpg", dpi=600, bbox_inches="tight")

def plot_silce_COMB_vs_SOC(train_list_recover):
    # 特征预测的举例说明：电池能量预测
    trips = train_list_recover[2]
    plt.figure(figsize=fig_size)
    f_total_power = np.polyfit(trips.used_soc, trips.total_power, 1)
    val_total_power = np.polyval(f_total_power, range(0, int(trips.soc[0])))
    plt.plot(range(int(trips.soc[0]), 0, -1), val_total_power, label="Linear Model")
    plt.plot(trips.soc, trips.total_power, label="True Value")
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    # plt.title("SOC & Total_power", fontsize=title_size)
    plt.xlabel("SOC  [%]", fontsize=xlabel_size)
    plt.ylabel("COEB  [J]", fontsize=ylabel_size)
    plt.legend(fontsize=legend_size)
    plt.savefig("/home/liang/range_prediction/output/figures/COEB_prediction.jpg", dpi=600, bbox_inches="tight")

def plot_model_comparision(save_path):
    score = [3.63, 3.74, 3.97, 11.52, 9.44, 10.65, 11.95, 8.55]#obtain from experiments
    model_name = ["XGBoost", "LightGBM", "GBRT", "Lasso", "Elastic_net", "RandomForest", "Bagging", "Neural_NetWork"]
    plt.figure(figsize=(15, 8))
    plt.bar(model_name, score)
    plt.title("Model Selection", fontsize=title_size)
    plt.xticks(rotation=30, fontsize=xticks_size)
    plt.yticks(np.arange(0, 16, 3), fontsize=yticks_size)
    plt.ylabel('MAPE [%]', fontsize=ylabel_size)
    for a, b in zip(model_name, score):
        plt.text(a, b + 0.25, str(b) + "%", ha="center", va="center", fontsize=xticks_size)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")

def plot_train_process():
    # 迭代图
    root_path = "/home/liang/mile_estimator/tijiaocode"
    path = os.path.join(root_path,"origin_silce_plot/car_1/before_sample/gradientboost_print.csv")
    gradientboost_print_1 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_1/before_sample/xgboost_print.csv")
    xgboost_print_1 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_1/before_sample/light_print.csv")
    light_print_1 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_2/before_sample/gradientboost_print.csv")
    gradientboost_print_2 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_2/before_sample/xgboost_print.csv")
    xgboost_print_2 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_2/before_sample/light_print.csv")
    light_print_2 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_3/before_sample/gradientboost_print.csv")
    gradientboost_print_3 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_3/before_sample/xgboost_print.csv")
    xgboost_print_3 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_3/before_sample/light_print.csv")
    light_print_3 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/gradientboost_print.csv")
    gradientboost_print_4 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/xgboost_print.csv")
    xgboost_print_4 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/light_print.csv")
    light_print_4 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/gradientboost_print.csv")
    gradientboost_print_5 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/xgboost_print.csv")
    xgboost_print_5 = pd.read_csv(path)

    path = os.path.join(root_path, "origin_silce_plot/car_4/before_sample/light_print.csv")
    light_print_5 = pd.read_csv(path)

    model_print = [gradientboost_print_1, xgboost_print_1, light_print_1,
                   gradientboost_print_2, xgboost_print_2, light_print_2,
                   gradientboost_print_3, xgboost_print_3, light_print_3,
                   gradientboost_print_4, xgboost_print_4, light_print_4,
                   gradientboost_print_5, xgboost_print_5, light_print_5, ]
    gradient_index = [1, 4, 7, 10, 13]
    xgboost_index = [2, 5, 8, 11, 14]
    light_index = [3, 6, 9, 12, 15]
    xlable_index = [13, 14, 15]
    title_index = [1, 2, 3]
    plt.figure(figsize=subplot_fig_size)
    for i in range(1, 16):
        ax = plt.subplot(6, 3, i)
        if i in gradient_index:
            ax = plt.plot(model_print[i - 1].loc[16:].iloc[:, 0], model_print[i - 1].loc[16:].train_rmse,
                          label="train_rmse")

            plt.xticks(rotation=20, fontsize=subplot_xticks_size)

            plt.ylabel("Score", fontsize=subplot_ylabel_size)
            plt.yticks(np.arange(0, 3.5, 0.5), fontsize=subplot_ylabel_size)
            plt.plot(model_print[i - 1].loc[16:].iloc[:, 0], model_print[i - 1].loc[16:].out_of_bag_rmse_imporve,
                     color="r", label="out_of_bag_rmse_imporve")

            if i in xlable_index:
                plt.xlabel("Iteration", fontsize=xlabel_size)
            if i in title_index:
                plt.title("GBRT", fontsize=subplot_title_size)
            plt.legend(fontsize=legend_size)
            if i == 1:
                plt.text(model_print[i - 1].iloc[:, 0].max() * 0.2,
                         model_print[i - 1].loc[16:].train_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              16:].train_rmse.max(),
                         "{}st fold".format(int(i / 3) + 1), fontsize=ylabel_size)
            elif i == 4:
                plt.text(model_print[i - 1].iloc[:, 0].max() * 0.2,
                         model_print[i - 1].loc[16:].train_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              16:].train_rmse.max(),
                         "{}nd fold".format(int(i / 3) + 1), fontsize=ylabel_size)
            elif i == 7:
                plt.text(model_print[i - 1].iloc[:, 0].max() * 0.2,
                         model_print[i - 1].loc[16:].train_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              16:].train_rmse.max(),
                         "{}rd fold".format(int(i / 3) + 1), fontsize=ylabel_size)
            else:
                plt.text(model_print[i - 1].iloc[:, 0].max() * 0.2,
                         model_print[i - 1].loc[16:].train_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              16:].train_rmse.max(),
                         "{}th fold".format(int(i / 3) + 1), fontsize=ylabel_size)

            plt.tight_layout()
        elif i in xgboost_index:
            model_print[i - 1].loc[100:].train_mae.plot(label="train_mae")
            model_print[i - 1].loc[100:].train_rmse.plot(label="train_rmse")
            model_print[i - 1].loc[100:].test_mae.plot(label="test_mae")
            model_print[i - 1].loc[100:].test_rmse.plot(label="test_rmse")

            plt.xticks(rotation=20, fontsize=subplot_xticks_size)

            plt.ylabel("Score", fontsize=subplot_ylabel_size)
            if i in xlable_index:
                plt.xlabel("Iteration", fontsize=xlabel_size)
            if i in title_index:
                plt.title("XGBoost", fontsize=subplot_title_size)
            plt.legend(fontsize=legend_size)
            if i == 2:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}st fold".format(int(i / 3) + 1), fontsize=ylabel_size)
            elif i == 5:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}nd fold".format(int(i / 3) + 1), fontsize=ylabel_size)
            elif i == 8:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}rd fold".format(int(i / 3) + 1), fontsize=ylabel_size)
            else:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}th fold".format(int(i / 3) + 1), fontsize=ylabel_size)
            plt.yticks(np.arange(0.2, 3.5, 0.5), fontsize=subplot_ylabel_size)
            plt.legend(fontsize=legend_size)
            plt.tight_layout()
        else:
            model_print[i - 1].loc[100:].train_mae.plot(label="train_mae")
            model_print[i - 1].loc[100:].train_rmse.plot(label="train_rmse")
            model_print[i - 1].loc[100:].test_mae.plot(label="test_mae")
            model_print[i - 1].loc[100:].test_rmse.plot(label="test_rmse")
            plt.legend(fontsize=legend_size)

            plt.xticks(rotation=20, fontsize=subplot_xticks_size)
            if i in xlable_index:
                plt.xlabel("Iteration", fontsize=xlabel_size)
            if i in title_index:
                plt.title("LightGBM", fontsize=subplot_title_size)
            if i == 3:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}st fold".format(int(i / 3)), fontsize=ylabel_size)
            elif i == 6:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}nd fold".format(int(i / 3)), fontsize=ylabel_size)
            elif i == 9:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}rd fold".format(int(i / 3)), fontsize=ylabel_size)
            else:
                plt.text(len(model_print[i - 1]) * 0.2,
                         model_print[i - 1].loc[100:].test_rmse.max() - 0.4 * model_print[i - 1].loc[
                                                                              100:].test_rmse.max(),
                         "{}th fold".format(int(i / 3)), fontsize=ylabel_size)
            plt.ylabel("Score", fontsize=subplot_ylabel_size)
            plt.yticks(np.arange(0.2, 3.5, 0.5), fontsize=subplot_ylabel_size)
            plt.tight_layout()

    plt.savefig("/home/liang/range_prediction/output/figures/training_process.jpg", dpi=600, bbox_inches="tight")

def plot_feature_importance_XGB():
    xgb_feature_name = ["COEM", "COEB", "driving_time",
                        "Used_SOC", "Temp_min", "RDP_2", "BR",
                        "Temp_max", "SOC_start", "RDP_4", "RDP_3",
                        "Temp_diff", "AR", "RDP_1", "SR"]
    xgb_score = [43097, 38253, 35359, 31152, 27191, 26527, 24639, 24604, 24439, 23202, 22767, 20423, 19682, 17662,
                 15326]
    xgb_feature_imprtance = pd.DataFrame({"feature_name": xgb_feature_name, "score": xgb_score})
    plt.figure(figsize=fig_size)
    plt.barh(y=xgb_feature_imprtance.feature_name, width=xgb_feature_imprtance.score)
    for a, b in zip(xgb_score, xgb_feature_name):
        plt.text(a + 3300, b, str(a), ha="center", va="center", fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xticks(range(0, 52000, 10000), fontsize=xticks_size)
    plt.xlabel("F score", fontsize=xlabel_size)
    plt.title("XGBoost Feature Importance", fontsize=title_size)
    plt.savefig("/home/liang/range_prediction/output/figures/xgb_feature_importance.jpg", dpi=600, bbox_inches="tight")


def plot_LGB_feature_importance():
    lgbm_feature_name = ["COEM", "driving_time", "COEB",
                         "Used_SOC", "Temp_min", "RDP_2",
                         "RDP_3", "BR", "Temp_max",
                         "RDP_4", "SOC_start", "Temp_diff",
                         "AR", "RDP_1", "SR"]
    lgb_score=[34588,33451,29920,28579,24130,23597,23079,23029,22297,21660,20765,19380,17122,15775,12503]

    lgb_feature_imprtance = pd.DataFrame({"feature_name": lgbm_feature_name, "score": lgb_score})

    plt.figure(figsize=fig_size)
    plt.barh(y=lgb_feature_imprtance.feature_name, width=lgb_feature_imprtance.score)
    for a, b in zip(lgb_score, lgbm_feature_name):
        plt.text(a + 2600, b, str(a), ha="center", va="center", fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xticks(range(0, 50000, 10000), fontsize=xticks_size)
    plt.xlabel("F score", fontsize=xlabel_size)
    plt.title("LightGBM Feature Importance", fontsize=title_size)
    plt.savefig("/home/liang/range_prediction/output/figures/LGB_feature_importance.jpg", dpi=500, bbox_inches="tight")
if __name__ == '__main__':
    pass
    # data = pd.read_csv("./vehicle_data/4.csv")
    # print(data["mileage"].max()-data["mileage"].min())
    # vis_function.plot_feature_importance_XGB()
    # vis_function.plot_LGB_feature_importance()
    # vis_function.plot_train_process()
    # vis_function.plot_model_comparision()
    # train_list_recover = delet_stopping_trips(train_list_recover)
    # vis_function.plot_COEM_vs_Used_soc(train_list_recover)
    # vis_function.plot_COMB_VS_Used_SOC(train_list_recover)
    # vis_function.plot_silce_COME_vs_SOC(train_list_recover)
    # vis_function.plot_silce_COMB_vs_SOC(train_list_recover)

