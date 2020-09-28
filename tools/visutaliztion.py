import os
from utils.vis_function import plot_model_comparision
if __name__ == '__main__':
    path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/range_prediction/output/figures"
    model_compar = os.path.join(path,"model_comparsion.jpg")
    plot_model_comparision(model_compar)