# Machine learning based driving range prediction

**Authors**: [liang zhao](https://github.com/liangzhao123),

## Updates
2020-09-26: Create this project. Though there are some code not released, we will soon released them in this project

## Demo

# Introduction
![model](https://github.com/liangzhao123/range_prediction/blob/master/doc/results_ture_vs_pred.jpg)
Limited range is one of the major obstacles to the widespread application of electric vehicles (EVs). Accurately predicting the driving range of EVs can effectively reduce the range anxiety of drivers and maximize the driving range of EVs. In this paper, we propose a blended machine learning model to predict the driving range of EVs based on real-world historical driving data. The blended model consists of two advanced machine learning algorithms, the Extreme Gradient Boosting Regression Tree (XGBoost) and the Light Gradient Boosting Regression Tree (LightGBM). The blended model is trained to “learn” the relationship between the driving distance and the features (e.g. the battery state of charge (SOC), the cumulative output energy of the motor and the battery, different driving patterns, and temperature of the battery). Besides, this study first proposes an “anchor (baseline)-based” strategy, which eliminates the unbalance distribution of dataset. The results of experiments suggest that our proposed anchor-based blended model has more robust performances with a small prediction error range of [-0.80, 0.80] km as compared with previous methods.

# Dependencies
- `python 3.6`
- `xgboost` 
- `lightgbm`
- `sklearn`
- `imblearn`
- `matplotlib` 

# Installation
1. Clone this repository.
2. install the dependencies, e.g.
```bash
$ pip install xgboost
$ pip install lightgbm
$ pip install matplotlib
```

# Data Preparation
1. Download the  dataset from [here](https://pan.baidu.com/s/1fG6bC6tqb2nWABSlQE93lw 
                                     passward：sa9p). Data to download include:
    * Raw data, including five vehicles' driving data
    * After silce, cut the raw dataset into segments (individual trips)

2. data preprocessing

```bash
$ python utils/preprocessing.py
```
or you can use the data in “after silce” fold to train


3. The data fold
```plain
└── vehicle_data
       ├── after silce  
       |   ├── 0.csv
       |   ├── 1.csv
       |   └── ...
       └── raw  
       |   ├── 0.csv
       |   ├── 1.csv
       |   ├── ....
       |   └── 4.csv
```

# Train
To train the xgboost model , run the following command:
```
cd trainer
python xgboost_light.py
```
To train the lightgbm model , run the following command:
```
cd trainer
python lightgbm_model.py
```
To train the ANNs model , run the following command:
```
cd trainer
python ANNs.py
```
# Eval
The evaluation of these models are conducted in train process
```
The evaluation of these models are conducted in train process
```
## Citation
If you find this work useful in your research, please consider cite:
```
@journals{IEEE access,
title={Machine learning based driving range prediction for electric vehicles},
author={Liang Zhao, Yao Wei, Yu Wang, Jie He},
year={2020}
}
```

## Acknowledgement
The data collected from NDANEV.
* [NDANEV](http://www.ndanev.com/) 
* [NCBDC](http://www.ncbdc.top/)


