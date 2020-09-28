import numpy as np
import pandas as pd

def rmse(preds,y):
    error = np.sqrt(np.sum((preds-y)*(preds-y))/len(preds))
    return error
def mape(preds,y):
    y = np.clip(y,a_max=np.inf,a_min=1e-1)
    error = np.sum(np.abs(preds-y)/y)/len(preds)
    return error
def mae(preds,y):
    error = np.sum(np.abs(preds-y))/len(preds)
    return error


if __name__ == '__main__':
    a = 3.2
    a = np.clip(a,a_max=np.inf,a_min=1e-2,)
    print(a)