import numpy as np
from matplotlib import pylab as plt
from numpy.linalg import svd
from PIL import Image
import scipy.io as sio

class TRPCA:

    def converged(self, L, E, X, L_new, E_new):
        eps = 1e-6
        condition1 = np.max(L_new - L) < eps
        condition2 = np.max(E_new - E) < eps
        condition3 = np.max(L_new + E_new - X) < eps
        return condition1 and condition2 and condition3

    def SoftShrink(self, X, tau):
        z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)

        return z

    def SVDShrink(self, X, tau):
        W_bar = np.empty((X.shape[0], X.shape[1], 0), complex)
        D = np.fft.fft(X)
        for i in range (6):
            if i < 6:
                U, S, V = svd(D[:, :, i], full_matrices = False)
                S = self.SoftShrink(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
            if i == 6:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))
        return np.fft.ifft(W_bar).real


    def ADMM(self, X):
        m, n, l = X.shape
        rho = 1.1
        mu = 1e-3
        mu_max = 1e10
        max_iters = 1000
        lamb = (max(m, n) * l) ** -0.5
        L = np.zeros((m, n, l), float)
        E = np.zeros((m, n, l), float)
        Y = np.zeros((m, n, l), float)
        iters = 0
        while True:
            iters += 1
            # update L(recovered image)
            L_new = self.SVDShrink(X - E + (1/mu) * Y, 1/mu)
            # update E(noise)
            E_new = self.SoftShrink(X - L_new + (1/mu) * Y, lamb/mu)
            Y += mu * (X - L_new - E_new)
            mu = min(rho * mu, mu_max)
            if (self.converged(L, E, X, L_new, E_new) and iters >=10) or iters >= max_iters:
                return L_new, E_new
            else:
                L, E = L_new, E_new
                print(np.max(X - L - E))