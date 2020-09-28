from utils.data_clean import *


if __name__ == '__main__':
    train_0 = pd.read_csv("./TrainData/Vehicle No.0.csv")
    train_1 = pd.read_csv("./TrainData/Vehicle No.1.csv")
    train_2 = pd.read_csv("./TrainData/Vehicle No.2.csv")
    train_3 = pd.read_csv("./TrainData/Vehicle No.3.csv")
    train_4 = pd.read_csv("./TrainData/Vehicle No.4.csv")

    train_0 = main_fast_1(train_0)
    train_1 = main_fast_1(train_1)
    train_2 = main_fast_1(train_2)
    train_3 = main_fast_1(train_3)
    train_4 = main_fast_1(train_4)

    train_0 = wash_2(train_0)
    train_1 = wash_2(train_1)
    train_2 = wash_2(train_2)
    train_3 = wash_2(train_3)
    train_4 = wash_2(train_4)

    train_0 = remain_fast_1(train_0)
    train_1 = remain_fast_1(train_1)
    train_2 = remain_fast_1(train_2)
    train_3 = remain_fast_1(train_3)
    train_4 = remain_fast_1(train_4)

    train_0 = outlier_justify(train_0)
    train_1 = outlier_justify(train_1)
    train_2 = outlier_justify(train_2)
    train_3 = outlier_justify(train_3)
    train_4 = outlier_justify(train_4)

    train_list_0 = main_fast_2(train_0)
    train_list_1 = main_fast_2(train_1)
    train_list_2 = main_fast_2(train_2)
    train_list_3 = main_fast_2(train_3)
    train_list_4 = main_fast_2(train_4)

    train_silce = []
    train_silce = train_list_0 + train_list_1 + train_list_2 + train_list_3 + train_list_4

    temp_index = []
    for i, x in enumerate(train_silce):
        if len(x) <= 2:
            temp_index.append(i)
    train_silce = [train_silce[i] for i in range(len(train_silce)) if (i not in temp_index)]

    train_silce = main_fast_3(train_silce)

    for i, x in enumerate(train_silce):
        x["cut_point"] = 0
        x.cut_point[0] = -1
    temp_index = []
    for i, x in enumerate(train_silce):
        if x.mile_rate.min() < 0:
            temp_index.append(i)
    outlier_silce = []
    for i in temp_index:
        outlier_silce.append(train_silce[i])
    train_silce = [train_silce[i] for i in range(len(train_silce)) if (i not in temp_index)]

    normal_silce = []
    for i, x in enumerate(outlier_silce):
        normal_silce += cut_silce_by_mile_for_take_silce(x)

    normal_silce = main_fast_3(normal_silce)

    train_silce = normal_silce + train_silce

    train_silce = time_change(train_silce)

    train_silce = intercept_missing_data(train_silce)

    train_silce = to_nan(train_silce)

    train_silce = linear_difference(train_silce)

    train_silce = main_fast_3(train_silce)

    train_silce = time_change(train_silce)

    train_silce = round_float(train_silce)

    train_silce = work_condition_feature(train_silce)

    train = pd.concat(train_silce, axis=0)
    train.reset_index(inplace=True)
    train.drop(["index"], axis=1, inplace=True)

    CutPoint = train.query("time_interval==0").index

    # 显然根据手肘法，发现是在4左右应该使用4作为聚类的数量
    train_wc = train[["speed"
        , "motor_current", "motor_current_rate"]]
    scaler_for_cluster = StandardScaler()
    scaler_for_cluster.fit(train_wc)
    train_wc = scaler_for_cluster.transform(train_wc)
    kmeans = KMeans(n_clusters=4, random_state=0, n_jobs=-1).fit(train_wc)
    train["work_condition"] = kmeans.labels_
    train.work_condition = train.work_condition.astype("int")
    train.work_condition = train.work_condition.astype("category")

    work_condition_cat = pd.get_dummies(train.work_condition)
    work_condition_cat = work_condition_cat.rename(columns={0: "work_condition_0",
                                                            1: "work_condition_1",
                                                            2: "work_condition_2",
                                                            3: "work_condition_3"})

    train = pd.concat([train, work_condition_cat], axis=1)

    train_list_recover = cut_silce_by_point(train, CutPoint)

    train_list_recover = create_work_condition_percentage(train_list_recover)

    train = pd.concat(train_list_recover[:1800], axis=0)
    train.reset_index(inplace=True)
    train.drop(["index"], axis=1, inplace=True)

    for i in range(len(train_list_recover)):
        train_list_recover[i].to_csv("./data/{}.csv".format(i))