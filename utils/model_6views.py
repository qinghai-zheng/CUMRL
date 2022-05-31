import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.VCL import URL
from utils.URL import VCL
from utils.Tensor_RPCA_6views import TRPCA
from utils.next_batch import next_batch
import math
from sklearn.utils import shuffle
import timeit
from utils.print_result import print_result
import scipy.io as sio


def model(views, gt, dim_mid, n_data, para_lambda, dims_vcl, dims_url, act_vcl, act_url, lr, epochs, batch_size):

    start = timeit.default_timer()
    err_pre = list()
    err_total = list()
    view_num = len(views)

    net_ae=[]
    net_dg = []
    for i in range(view_num):
        net_ae.append(URL(i + 1, dims_vcl[i], para_lambda, act_vcl[i]))
    for i in range(view_num):
        net_dg.append(VCL(i + 1, dims_url[i], act_url[i]))

    H = np.random.uniform(0, 1, [views[0].shape[0], dims_url[0][0]])
    x_input = []
    for i in range(view_num):
        x_input.append(tf.placeholder(np.float32, [None, dims_vcl[i][0]]))

    with tf.variable_scope("H"):
        h_input = tf.Variable(xavier_init(batch_size, dims_url[0][0]), name='LatentSpaceData')
        h_list = tf.trainable_variables()
    fea_latent = []
    for i in range(view_num):
        fea_latent.append(tf.placeholder(np.float32, [None, dims_vcl[i][-1]]))

    loss_pre = net_ae[0].loss_reconstruct(x_input[0])
    for i in range(1,view_num):
        loss_pre += net_ae[i].loss_reconstruct(x_input[i])
    pre_train = tf.train.AdamOptimizer(lr[0]).minimize(loss_pre)

    loss_ae = net_ae[0].loss_total(x_input[0],fea_latent[0])
    for i in range(1,view_num):
        loss_ae += net_ae[i].loss_total(x_input[i],fea_latent[i])

    net_ae_par = []
    for i in range(view_num):
        net_ae_par.extend(net_ae[i].netpara)
    update_ae = tf.train.AdamOptimizer(lr[1]).minimize(loss_ae, var_list=net_ae_par)
    z_half = []
    for i in range(view_num):
        z_half.append(net_ae[i].get_z_half(x_input[i]))

    loss_dg = net_dg[0].loss_degradation(h_input, fea_latent[0])
    for i in range(1,view_num):
        loss_dg += net_dg[i].loss_degradation(h_input, fea_latent[i])

    loss_dg = para_lambda *loss_dg
    net_de_par = []
    for i in range(view_num):
        net_de_par.extend(net_dg[i].netpara)
    update_dg = tf.train.AdamOptimizer(lr[2]).minimize(loss_dg, var_list=net_de_par)

    update_h = tf.train.AdamOptimizer(lr[3]).minimize(loss_dg, var_list=h_list)

    g = []
    for i in range(view_num):
        g.append(net_dg[i].get_g(h_input))


    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    for k in range(epochs[0]):
        for i in range(view_num):
            views[i] = shuffle(views[i],random_state=1)
        gt = shuffle(gt,random_state=1)
        for batch_x, batch_No in next_batch(views, batch_size):
            _, val_pre = sess.run([pre_train, loss_pre], feed_dict={i: d for i, d in zip(x_input, batch_x)})
            err_pre.append(val_pre)
            output = "Pre_epoch : {:.0f}, Batch : {:.0f}  ===> Reconstruction loss = {:.8f} ".format((k + 1), batch_No,
                                                                                                     val_pre)
            print(output)

    num_samples = views[0].shape[0]
    num_batchs = math.ceil(num_samples / batch_size)  # fix the last batch
    score = 0
    nmi_b = 0
    acc_b = 0
    ar_b = 0
    f_b =0
    for j in range(epochs[1]):
        for i in range(view_num):
            views[i] = shuffle(views[i],random_state=1)
        H, gt = shuffle(H, gt,random_state=1)
        for num_batch_i in range(int(num_batchs)):
            start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
            end_idx = min(num_samples, end_idx)
            batch_x = []
            for i in range(view_num):
                batch_x.append(views[i][start_idx: end_idx, ...])

            batch_h = H[start_idx: end_idx, ...]
            batch_g = []
            for i in range(view_num):
                batch_g.append(sess.run(g[i], feed_dict={h_input: batch_h}))

            d1 = {i:d for i,d in zip(x_input, batch_x)}
            d2 = {i:d for i,d in zip(fea_latent, batch_g)}
            d1.update(d2)
            _, val_ae = sess.run([update_ae, loss_ae], feed_dict=d1)

            batch_tensor_1 = np.empty(shape=[n_data, dim_mid, 0])
            for i in range(view_num):
                tmp = sess.run(z_half[i], feed_dict={x_input[i]: batch_x[i]})
                a_tmp,b_tmp = np.shape(tmp)
                tmp_3d = np.reshape(tmp,(a_tmp,b_tmp,1))
                # batch_z_half.append(tmp)
                batch_tensor_1 = np.concatenate((batch_tensor_1,tmp_3d),axis=2)

            sess.run(tf.assign(h_input, batch_h))

            TRPCA_1 = TRPCA()
            batch_tensor_new_1, _ = TRPCA_1.ADMM(batch_tensor_1)
            batch_z_half = []
            for i in range(view_num):
                batch_z_half.append(batch_tensor_new_1[:,:,i])

            _, val_dg = sess.run([update_dg, loss_dg], feed_dict={i:d for i,d in zip(fea_latent, batch_z_half)})

            for k in range(epochs[2]):
                sess.run(update_h, feed_dict={i:d for i,d in zip(fea_latent, batch_z_half)})
            batch_h_new = sess.run(h_input)
            H[start_idx: end_idx, ...] = batch_h_new

            sess.run(tf.assign(h_input, batch_h_new))


            batch_g_new = []
            for i in range(view_num):
                tmp = sess.run(g[i], feed_dict={h_input: batch_h})
                batch_g_new.append(tmp)

            d1 = {i: d for i, d in zip(x_input, batch_x)}
            d2 = {i: d for i, d in zip(fea_latent, batch_g_new)}
            d1.update(d2)

            val_total = sess.run(loss_ae, feed_dict=d1)
            err_total.append(val_total)
            output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.8f} ".format((j + 1),
                                                                                                  (num_batch_i + 1),
                                                                                                  val_total)
            print(output)

    elapsed = (timeit.default_timer() - start)
    print("Time used: ", elapsed)
    return H, gt


def save_model(saver,sess,model_path):
    save_path = saver.save(sess, model_path)
    print("model saved in ", save_path)
    return save_path


def restore(saver,sess,restore_path):
    saver.restore(sess, restore_path)
    print("mode restored successed.")


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)