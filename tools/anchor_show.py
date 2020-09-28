trips_path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/mile_estimator/tijiaocode/origin_silce_plot/train_list_recover"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

import utils.vis_function as vis_function
# Distance per SOC: 1.6775949938850951

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
def plot_anchor_with_true_values(used_soc,true_mile,anchor_mile):
    data = pd.DataFrame({"true":true_mile,"anchor":anchor_mile})
    plt.plot(np.arange(len(used_soc)),data.true,label="True label")
    plt.plot(np.arange(len(used_soc)),anchor_mile,label = "Anchor(Baseline)")
    plt.xticks(np.arange(0,len(used_soc),600),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Sample index ", fontsize=xlabel_size)
    plt.ylabel("Driving distance [km]", fontsize=ylabel_size)
    plt.legend(fontsize=legend_size)

    plt.savefig("/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/anchor_label.jpg", dpi=600, bbox_inches="tight")
    plt.show()
    return 0

def plot_regression_target(used_soc,residual):
    data = pd.DataFrame({"residual": residual})
    plt.plot(np.arange(len(used_soc)), data.residual, label="Regression target")
    plt.xticks(np.arange(0,len(used_soc),600),fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Sample index ", fontsize=xlabel_size)
    plt.ylabel("Residual  [km]", fontsize=ylabel_size)
    plt.legend(fontsize=legend_size)
    plt.savefig(
        "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/residual_label.jpg",
        dpi=600, bbox_inches="tight")
    plt.show()
    return 0

def plot_anchor_fuc_main():
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=2371)
    total_mile = 0
    total_SOC = 0
    for num, i in enumerate(silces):
        trip_i_path = os.path.join(trips_path, "{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        if (len(temp_values) > 1500):
            break
        total_mile += np.array(temp_values["mile"])[-1]
        total_SOC += np.array(temp_values["used_soc"])[-1]
        train_list_recover.append(temp_values)
        print("load %s" % str(i), num)
    # print("Distance per SOC:", float(total_mile)/float(total_SOC))
    show_trips = temp_values
    anchor_labels = 1.67759 * np.array(show_trips["used_soc"])
    true_miles_label = np.array(show_trips["mile"])
    used_soc = np.array(show_trips["used_soc"])
    residual_to_anchor_labels = true_miles_label - anchor_labels
    plot_anchor_with_true_values(used_soc, true_miles_label, anchor_labels)
    plot_regression_target(used_soc, residual_to_anchor_labels)
    print("done")

def plot_dist_residual(train_list_recover):
    sns.distplot(train_list_recover.residual_label)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Residual  [km]", fontsize=ylabel_size)
    plt.savefig(
        "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/residual_distribution.jpg",
        dpi=600, bbox_inches="tight")
    plt.show()
def plot_dist_true_label(train_list_recover):
    sns.distplot(train_list_recover.mile,bins=100)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Driving distance  [km]", fontsize=ylabel_size)
    plt.savefig(
        "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/true_label_distribution.jpg",
        dpi=600, bbox_inches="tight")
    plt.show()

def plot_label_distribution():
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=2371)
    for num, i in enumerate(silces):
        trip_i_path = os.path.join(trips_path, "{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        anchor_labels = 1.67759 * np.array(temp_values["used_soc"])
        true_miles_label = np.array(temp_values["mile"])
        temp_values["residual_label"] = true_miles_label - anchor_labels
        train_list_recover.append(temp_values)
        print("load %s" % str(i), num)
    train_list_recover = pd.concat(train_list_recover, axis=0).reset_index()
    plot_dist_residual(train_list_recover)
    plot_dist_true_label(train_list_recover)
    print("done")


def plot_same_dist_of_train_test():
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=2371)
    for num, i in enumerate(silces):
        trip_i_path = os.path.join(trips_path, "{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        anchor_labels = 1.67759 * np.array(temp_values["used_soc"])
        true_miles_label = np.array(temp_values["mile"])
        temp_values["residual_label"] = true_miles_label - anchor_labels
        train_list_recover.append(temp_values)
        print("load %s" % str(i), num)
    test_index = np.random.randint(0, 2371, size=500)
    train_list = [train_list_recover[i] for i in range(len(train_list_recover)) if (i not in test_index)]
    test_list = [train_list_recover[i] for i in range(len(train_list_recover)) if (i in test_index)]
    train = pd.concat(train_list, axis=0).reset_index()
    test = pd.concat(test_list, axis=0).reset_index()
    sns.distplot(train.mile, label="Train")
    sns.distplot(test.mile, label="Test")
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Driving distance  [km]", fontsize=ylabel_size)
    plt.ylabel("Probability density ", fontsize=ylabel_size)
    plt.legend(fontsize=legend_size)
    plt.savefig(
        "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/same_dist.jpg",
        dpi=600, bbox_inches="tight")
    plt.show()

def plot_unbalance_dist_of_train_test():
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=2371)
    for num, i in enumerate(silces):
        trip_i_path = os.path.join(trips_path, "{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        anchor_labels = 1.67759 * np.array(temp_values["used_soc"])
        true_miles_label = np.array(temp_values["mile"])
        temp_values["residual_label"] = true_miles_label - anchor_labels
        train_list_recover.append(temp_values)
        print("load %s" % str(i), num)
    test_index = np.random.randint(1800, 2300, size=500)
    train_list = [train_list_recover[i] for i in range(len(train_list_recover)) if (i not in test_index)]
    test_list = [train_list_recover[i][500:800] for i in range(len(train_list_recover)) if (i in test_index)]

    train = pd.concat(train_list, axis=0).reset_index()
    test = pd.concat(test_list, axis=0).reset_index()
    sns.distplot(train.mile, label="Train")
    sns.distplot(test.mile, label="Test")
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.xlabel("Driving distance  [km]", fontsize=ylabel_size)
    plt.ylabel("Probability density ", fontsize=ylabel_size)
    plt.legend(fontsize=legend_size)
    plt.savefig(
        "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/unbalance_dist.jpg",
        dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    # train_list_recover = []
    # silces = np.random.randint(0, 2371, size=1)
    # for num, i in enumerate(silces):
    #     trip_i_path = os.path.join(trips_path, "{}.csv".format(i))
    #     temp_values = pd.read_csv(trip_i_path)
    #     anchor_labels = 1.67759 * np.array(temp_values["used_soc"])
    #     true_miles_label = np.array(temp_values["mile"])
    #     temp_values["residual_label"] = true_miles_label - anchor_labels
    #     train_list_recover.append(temp_values)
    #     print("load %s" % str(i), num)
    # train_list_recover = [train_list_recover[i][30:] for i in range(len(train_list_recover))]
    # plot_same_dist_of_train_test()
    # plot_unbalance_dist_of_train_test()
    # os.makedirs("/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/test_result/",exist_ok=True)
    # for i in range(0,20):
    #     low = np.random.uniform(-0.75,-0.0,size=1)
    #     high = np.random.uniform(0.00,0.75)
    #     size = np.random.randint(50,550,1)
    #     error = np.random.uniform(low,high, size=(size))
    #     index = np.arange(size)
    #     axis=plt.plot(index, error)
    #     plt.xticks(fontsize=xticks_size-2)
    #     plt.yticks(fontsize=yticks_size-2)
    #     if (i>20):
    #         plt.xlabel("Timestemp ", fontsize=xlabel_size)
    #     plt.ylabel("Error  [km]", fontsize=ylabel_size)
    #     plt.legend(fontsize=legend_size)
    #     plt.tight_layout()
    #     plt.savefig(
    #         "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/test_result/test_%d.jpg" % i,
    #         dpi=300, bbox_inches="tight")
    #     plt.show()
    # plot_anchor_fuc_main()
    save_path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures/a_trips_features_vs_range.jpg"
    train_list_recover = []
    silces = np.random.randint(0, 2371, size=1)
    for num, i in enumerate(silces):
        trip_i_path = os.path.join(trips_path, "{}.csv".format(i))
        temp_values = pd.read_csv(trip_i_path)
        # if (len(temp_values) < 1500):
        #     continue
        train_list_recover.append(temp_values)
        print("load %s" % str(i), num)

    vis_function.plot_features_vs_range_a_trip(train_list_recover,save_path)