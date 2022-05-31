from utils.Dataset import Dataset
from utils.model_2views import model
import os
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
Demo for Handwritten
'''
if __name__ == '__main__':

    data = Dataset('handwritten_2view')

    lr_pre = 1.0e-3
    lr_vcl = 1.0e-3
    lr_url = 1.0e-3
    lr_h = 1.0e-1
    # --------------------------------------------------------------------
    epochs_pre = 1000
    epochs_total = 15
    epochs_h = 50
    # --------------------------------------------------------------------
    para_lambda = 10
    dim_mid = 200
    dim_desired = 60
    n_data = 2000

    views, view_shape,gt = data.load_data()
    view_num = len(views)
    for i in range(view_num):
        views[i] = data.normalize(views[i], 0)

    act_vcl = list()
    act_url = list()
    dims_vcl = list()
    dims_url = list()
    for i in range(view_num):
        act_vcl.append('sigmoid')
        act_url.append('sigmoid')
        dims_vcl.append([view_shape[i], dim_mid, ])
        dims_url.append([dim_desired, dim_mid])
    batch_size = views[0].shape[0]

    lr = [lr_pre, lr_vcl, lr_url, lr_h]
    epochs = [epochs_pre, epochs_total, epochs_h]

    H, gt = model(views, gt, dim_mid, n_data, para_lambda, dims_vcl, dims_url, act_vcl, act_url, lr, epochs, batch_size)

    savename = './result/CUMRL_Handwritten' + '_dm' + str(dim_mid) + '_dd' + str(dim_desired) + '_lambda' + str(
        para_lambda) + '.mat'
    sio.savemat(savename, mdict={'H': H, 'gt': gt})
