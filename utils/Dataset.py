import h5py
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio

class Dataset():
    def __init__(self, name):
        self.path = './dataset/'
        self.name = name

    def load_data(self):
        data_path = self.path + self.name + '.mat'
        if '2view_25dB' in data_path or '2views_25dB' in data_path or '2views' in data_path:
            dataset = sio.loadmat(data_path)
            x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
            # x1, x2, y = x1.value, x2.value, y.value
            # x1, x2, y = x1.transpose(), x2.transpose(), y.transpose()
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
            views = [x1.T, x2.T]
            view_shape = []
            view_shape.append(x1.shape[1])
            view_shape.append(x2.shape[1])
        elif '3views' in data_path or '3view' in data_path:
            dataset = sio.loadmat(data_path)
            x1, x2, x3, y = dataset['x1'], dataset['x2'], dataset['x3'], dataset['gt']
            # x1, x2, y = x1.value, x2.value, y.value
            # x1, x2, y = x1.transpose(), x2.transpose(), y.transpose()
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
            views = [x1.T, x2.T, x3.T]
            view_shape = []
            view_shape.append(x1.shape[1])
            view_shape.append(x2.shape[1])
            view_shape.append(x3.shape[1])
        elif '6views_25dB' in data_path or '6views_50dB' in data_path:
            dataset = sio.loadmat(data_path)
            x1, x2, x3, x4, x5, x6, y = dataset['x1'], dataset['x2'], dataset['x3'], dataset['x4'], dataset['x5'], dataset['x6'], dataset['gt']
            # x1, x2, y = x1.value, x2.value, y.value
            # x1, x2, y = x1.transpose(), x2.transpose(), y.transpose()
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
            views = [x1.T, x2.T, x3.T,x4.T, x5.T, x6.T]
            view_shape = []
            view_shape.append(x1.shape[1])
            view_shape.append(x2.shape[1])
            view_shape.append(x3.shape[1])
            view_shape.append(x4.shape[1])
            view_shape.append(x5.shape[1])
            view_shape.append(x6.shape[1])
        elif 'yale' in data_path or 'tweet' in data_path:
            data = sio.loadmat(data_path)
            features = data['X']
            y = data['gt']
            views = []
            view_shape = []
            for v in features[0]:
                view_shape.append(max(v.shape[0], v.shape[1]))
                views.append(v)
        elif 'ORL' in data_path:
            data = sio.loadmat(data_path)
            features = data['X']
            y = data['gt']
            views = []
            view_shape = []
            for v in features[0]:
                view_shape.append(max(v.shape[0], v.shape[1]))
                views.append(v)
        elif 'MSRA' in data_path:
            data = sio.loadmat(data_path)
            features = data['X']
            y = data['gt']
            views = []
            view_shape = []
            for v in features[0]:
                view_shape.append(v.shape[0])
                views.append(v)
        elif 'bbc' in data_path:
            data = sio.loadmat(data_path)
            features = data['data']
            y = data['truth']
            views = []
            view_shape = []
            for v in features[0]:
                view_shape.append(max(v.shape[0], v.shape[1]))
                views.append(v.A.T)

        elif 'movie617_2view' in data_path:
            data = sio.loadmat(data_path)
            features = data['X']
            y = data['gt']
            views = []
            view_shape = []
            for v in features[0]:
                view_shape.append(v.shape[0])
                views.append(v)
        elif '3_sources' in data_path:
            data = sio.loadmat(data_path)
            y = data['truth']
            views = []
            view_shape = []
            v1 = data['bbc'].toarray()
            v2 = data['guardian'].toarray()
            v3 = data['reuters'].toarray()
            view_shape.append(v1.shape[1])
            views.append(v1.T)
            view_shape.append(v2.shape[1])
            views.append(v2.T)
            view_shape.append(v3.shape[1])
            views.append(v3.T)
        elif 'NGs' in data_path:
            data = sio.loadmat(data_path)
            label = data['truelabel']
            views = []
            view_shape = []
            features = data['data']
            for v in features[0]:
                view_shape.append(v.shape[0])
                views.append(v)
            y = label[0, 0]
        elif '100leaves' in data_path:
            data = sio.loadmat(data_path)
            label = data['truelabel']
            views = []
            view_shape = []
            features = data['data']
            for v in features[0]:
                view_shape.append(v.shape[0])
                views.append(v)
            y = label[0, 0]
        else:
            dataset = h5py.File(data_path, mode='r')
            x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
            x1, x2, y = x1.value, x2.value, y.value
            x1, x2, y = x1.transpose(), x2.transpose(), y.transpose()
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
            views = [x1.T,x2.T]
            view_shape = []
            view_shape.append(x1.shape[1])
            view_shape.append(x2.shape[1])
        for i in range(len(views)):
            views[i] = views[i].T
        return views,view_shape, np.squeeze(y)

    def normalize(self, x, min=0):
        # min_val = np.min(x)
        # max_val = np.max(x)
        # x = (x - min_val) / (max_val - min_val)
        # return x

        if min == 0:
            scaler = MinMaxScaler([0, 1])
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x
