import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import h5py

tf_type=tf.float64

class Gaussian_Kernel():
    def __init__(self, jitter=0):
        self.jitter = tf.constant(jitter, dtype=tf_type)

    def matrix(self, X, t):
       K = self.cross(X,X,t)
       K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
       return K

    def cross(self, X1, X2, t):
        norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
        K = norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2
        K = tf.exp(-1.0 * K / t)
        return K

class Linear_Kernel():
    def __init__(self, jitter):
        self.jitter = tf.constant(jitter, dtype=tf_type)

    def matrix(self, X):
        K = self.cross(X, X)
        K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
        return K

    def cross(self, X1, X2):
        K = tf.matmul(X1, tf.transpose(X2))
        return K

class Normalization():
    def normalize(self, X, mean=None, std=None):
        """Normalize data along each column"""
        if mean is None or std is None:
            mean = np.mean(X, axis=0)
            # var = np.sum(np.square(X - mean), axis=0) / (np.shape(X)[0] - 1)
            # std = np.sqrt(var)
            std = np.std(X, axis=0)
        idx = std > 0
        X_n = X - mean
        X_n[:, idx] = X_n[:, idx] / std[idx]
        return X_n, mean, std

    def denormalize(self, X, mean, std):
        """Denormalize data along each column"""
        return X * std + mean

class Helper():
    def __init__(self, k, n_fold):
        self.k = k
        self.n_fold = n_fold
        self.file_name = '../k'+ str(self.k) + 'f' + str(self.n_fold)+'.mat'
        self.Y_file_name = '../k'+ str(self.k) + 'f' + str(self.n_fold)+'Y.mat'
        

    def get_cfg(self, layer0=-1, layer1=-1):
        data = loadmat(self.file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)

        X_train = data['X'].train
        X_test = data['X'].test
        Y_train = data['Y'].train
        Y_test = data['Y'].test
        Y_mean = data['Y'].mean
        Y_std = data['Y'].std

        M = data['M']
        # Alpha = data['Alpha']
        # A = data['A']

        C = data['C']
        U = data['U']
        n_bases = data['n_bases']
        n_samples = data['n_samples']

        # L = data['L']
        # R = data['R']
        if isinstance(n_samples, float):
            n_bases = [int(n_bases)]
            n_samples = [int(n_samples)]
            X_train = [data['X'].train]
            # X_test = [data['X'].test]
            Y_train = [data['Y'].train]
            # Y_test = [data['Y'].test]

            M = [data['M']]
            # Alpha = [data['Alpha']]
            # A = [data['A']]

            C = [data['C']]
            U = [data['U']]
            # R = [R]
            # L = [L]
        else:
            n_bases = n_bases.astype(np.int32)
            n_samples = n_samples.astype(np.int32)
        
            

        if layer0 != -1:
            X_train = [X_train[layer0]]
            Y_train = [Y_train[layer0]]
            # M = [Alpha[layer0]]
            # Alpha = [Alpha[layer0]]
            # A = [[]]
            C = [C[layer0]]
            U = [U[layer0]]
            n_bases = [C[0].shape[0]]
            n_samples = [n_samples[layer0]]


        feed_data = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test' : X_test,
            'Y_test0': Y_test[layer0],
            'Y_test1': Y_test[layer1],
            'Y_mean' : Y_mean,
            'Y_std' : Y_std,
            'M' : M,
            # 'Alpha' : Alpha,
            # 'A' : A,
            'C' : C,
            'U' : U,
        }

        cfg = {
            'in_d': 1,
            'out_d':128 * 128 ,
            'n_bases': n_bases,
            'n_samples': n_samples,
            'dim': [128, 128],
            'feed_data': feed_data,
            'a0': 1.5,
            'b0': 1,
            'epoch': 5000,
            'N_sampling': 1,
        }

        return cfg
    
    def get_cfg1(self, layer0=-1, layer1=-1):
        data = loadmat(self.file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)

        X_train = data['X'].train
        X_test = data['X'].test

        M = data['M']
        C = data['C']
        U = data['U']
        n_bases = data['n_bases'].astype(np.int32)
        n_samples = data['n_samples'].astype(np.int32)

        # load Y from v7.3 mat
        data = h5py.File(self.Y_file_name)
        n_layers = len(n_samples)
        Y_train = []
        Y_test = []
        for i in range(n_layers):
            train_ref = data['Y']['train'][i, 0]
            test_ref = data['Y']['test'][i, 0]    
            Y_train.append(np.transpose(data[train_ref][()]))
            Y_test.append(np.transpose(data[test_ref][()]))
        i = n_layers - 1
        test_ref = data['Y']['test'][i, 0]  
        Y_test.append(np.transpose(data[test_ref][()]))

        Y_mean = data['Y']['mean'][0,0]
        Y_std = data['Y']['std'][0,0]




        feed_data = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test' : X_test,
            'Y_test0': Y_test[layer0],
            'Y_test1': Y_test[layer1],
            'Y_mean' : Y_mean,
            'Y_std' : Y_std,
            'M' : M,
            'C' : C,
            'U' : U
        }
        
        cfg = {
            'in_d': 1,
            'out_d': 100 * 100 * 100,
            'n_bases': n_bases,
            'n_samples': n_samples,
            'dim': [100, 100, 100],
            'feed_data': feed_data,
            'a0': 1e-3,
            'b0': 1e-3,
            'epoch': 10000,
            'N_sampling': 1,
        }

        return cfg
    
    



    def save_rmse(self, s0r0, s0r1, s1r0, s1r1, s0ll, s1ll):
        of = './k'+ str(self.k) + 'f' + str(self.n_fold)+'_s0r0.csv'
        np.savetxt(of, s0r0, delimiter=',')
        of = './k'+ str(self.k) + 'f' + str(self.n_fold)+'_s0r1.csv'
        np.savetxt(of, s0r1, delimiter=',')
        of = './k'+ str(self.k) + 'f' + str(self.n_fold)+'_s1r0.csv'
        np.savetxt(of, s1r0, delimiter=',')
        of = './k'+ str(self.k) + 'f' + str(self.n_fold)+'_s1r1.csv'
        np.savetxt(of, s1r1, delimiter=',')
        of = './k'+ str(self.k) + 'f' + str(self.n_fold)+'_s0ll.csv'
        np.savetxt(of, s0ll, delimiter=',')
        of = './k'+ str(self.k) + 'f' + str(self.n_fold)+'_s1ll.csv'
        np.savetxt(of, s1ll, delimiter=',')

