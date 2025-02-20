import tensorflow as tf
import numpy as np
import EXPUtil as util

class DGP_hd:

    def __init__(self, cfg, jitter):
        """Constructor"""
        ## set structure param
        self.in_d = cfg['in_d']                    # input dimension
        self.out_d = cfg['out_d']                  # ouput dimension
        self.n_bases = cfg['n_bases']              # number of bases in first layer
        self.n_samples = cfg['n_samples']          # list of number of samples in each layer
        self.n_layers = len(self.n_samples) # number of layers
        # self.s_idx = cfg['s_idx']                  # subset index
        self.dim = cfg['dim']                      # prod(dim) == out_d
        self.a0 = cfg['a0']
        self.b0= cfg['b0']
        self.epoch = cfg['epoch']
        self.tstidx = [np.random.choice(128, 50, replace=False) for _ in range(5)]
        self.S0R0 = [[] for _ in range(6)]
        self.S0R1 = [[] for _ in range(6)]
        self.S1R0 = [[] for _ in range(6)]
        self.S1R1 = [[] for _ in range(6)]

        

        ## kernel
        self.gaussian_kernel = util.Gaussian_Kernel(jitter)
        ## set compute graph nodes in tensorflow
        # input
        self.X = [tf.placeholder(tf.float64, [self.n_samples[i], self.in_d]) for i in range(self.n_layers)]
        # output
        self.Y = [tf.placeholder(tf.float64, [self.n_samples[i], self.out_d]) for i in range(self.n_layers)]
        # phi
        self.phi = [tf.placeholder(tf.float64, [self.n_samples[i], self.n_bases[i]]) for i in range(self.n_layers)]
        # U
        # M, L, R
        self.feed_data = cfg['feed_data']
        ID = self.feed_data
        self.M = []
        self.L = []
        self.R = []
        self.U = []
        self.alpha = [None for i in range(self.n_layers)]
        
        for i in range(self.n_layers):
            self.M.append(tf.Variable(ID['M'][i], dtype=tf.float64))
            # self.M.append(tf.Variable(np.zeros([self.n_samples[i], self.n_bases[i]]), dtype=tf.float64))
            #self.L.append(tf.Variable(ID['L'][i], dtype=tf.float64))
            #self.R.append(tf.Variable(ID['R'][i], dtype=tf.float64))
            self.L.append(tf.Variable(tf.eye(self.n_samples[i], dtype=tf.float64), dtype=tf.float64))
            self.R.append(tf.Variable(tf.eye(self.n_bases[i], dtype=tf.float64), dtype=tf.float64))
            U_i = []
            if i == 0:
                n_bases = self.n_bases[i]
            else:
                n_bases = self.n_bases[i] - self.n_bases[i - 1]
            for j in range(n_bases):
                # print(j)
                U_ij = [tf.Variable(ID['U'][i][j][k].reshape([1,-1]), dtype=tf.float64) for k in range(len(self.dim))]
                # U_ij = [tf.Variable(np.random.normal(0,1, k), dtype=tf.float64) for k in self.dim]
                U_i.append(U_ij)
            self.U.append(U_i)

        # variance
        self.log_tau = [tf.Variable(0, dtype=tf.float64) for i in range(self.n_layers)]
        self.log_beta = [tf.Variable(0, dtype=tf.float64) for i in range(self.n_layers)]
        # Gaussian Kernel : log length scale
        self.log_t1 = [tf.Variable(0, dtype=tf.float64) for i in range(self.n_layers)]
        self.log_t2 = [tf.Variable(0, dtype=tf.float64) for i in range(self.n_layers)]

        # test
        self.X_test = tf.placeholder(tf.float64, [None, self.in_d])
        # self.Y_test = tf.placeholder(tf.float64, [None, self.out_d])

        self.Y_pre_s0 = None
        self.Y_pre_s1 = None

        # Loss function
        self.Loss = self.get_loss()
        self.opt = tf.train.AdamOptimizer().minimize(self.Loss)
        # prediction
        self.N_sampling = cfg['N_sampling']
        # self.sample_flag = cfg['sample_flag']
        self.Y_pre_s0, self.Y_pre_s1 = self.get_prediction()


        ## session
        self.sess = tf.Session()

        # init variables
        data_dict = {Y_i: data.reshape([-1, self.out_d]) for Y_i, data in zip(self.Y, self.feed_data['Y_train'])}
        data_dict.update({X_i: data.reshape([-1, self.in_d]) for X_i, data in zip(self.X, self.feed_data['X_train'])})
        self.sess.run(tf.global_variables_initializer(), feed_dict=data_dict)


    def get_prediction(self):
        N = self.N_sampling
        flag = np.array([0 for _ in range(self.n_layers)])
        Y_pre_s0 = self.sample(flag)
        flag = np.array([1 for _ in range(self.n_layers)])
        for i in range(N):
            sample = self.sample(flag)
            if i == 0:
                Y_pre_s1 = sample
            else:
                Y_pre_s1 = Y_pre_s1 + sample

        Y_pre_s1 = Y_pre_s1 / N
        return Y_pre_s0, Y_pre_s1

    #def train(self, train_x, train_y, test_x=None, test_y=None, mean=None, std=None):
    def train(self):
        """Train"""
        data_dict = {X_i: data.reshape([-1, self.in_d]) for X_i, data in zip(self.X, self.feed_data['X_train'])}
        data_dict.update({Y_i: data.reshape([-1, self.out_d]) for Y_i, data in zip(self.Y, self.feed_data['Y_train'])})

        for i in range(self.epoch):
            data_dict.update({self.phi[i]: np.random.normal(size=[self.n_samples[i], self.n_bases[i]])  for i in range(self.n_layers)})
            _, L_eval = self.sess.run([self.opt, self.Loss], feed_dict=data_dict)
            X_test = self.feed_data['X_test'].reshape([-1, self.in_d])
            N_Y_pre_s0, N_Y_pre_s1 = self.fit(X_test)
            Y_test = self.feed_data['Y_test1']
            Y_mean = self.feed_data['Y_mean']
            Y_std = self.feed_data['Y_std']
            N_Y_test = (Y_test - Y_mean) / Y_std 
            Y_pre_s0 = N_Y_pre_s0 * Y_std + Y_mean
            Y_pre_s1 = N_Y_pre_s1 * Y_std + Y_mean
            # sampling 0
            s0r0 = np.sqrt(np.mean(np.square(N_Y_pre_s0 - N_Y_test)))
            self.S0R0[0].append(s0r0)
            s0r1 = np.sqrt(np.mean(np.square(Y_pre_s0 - Y_test)))
            self.S0R1[0].append(s0r1)
            # sampling 1
            s1r0 = np.sqrt(np.mean(np.square(N_Y_pre_s1 - N_Y_test)))
            self.S1R0[0].append(s1r0)
            s1r1 = np.sqrt(np.mean(np.square(Y_pre_s1 - Y_test)))
            self.S1R1[0].append(s1r1)

            if i % 10 == 0:
                print('iter: ', i, 'Loss: ', L_eval, 's0r0', s0r0, 's0r1: ', s0r1, 's1r0', s1r0, 's1r1: ', s1r1)

            for i in range(1,6):
                N_Y_pre_s0, N_Y_pre_s1 = self.fit(X_test[self.tstidx[i - 1], :])                   
                Y_pre_s0 = N_Y_pre_s0 * Y_std + Y_mean
                Y_pre_s1 = N_Y_pre_s1 * Y_std + Y_mean
                # sampling 0
                s0r0 = np.sqrt(np.mean(np.square(N_Y_pre_s0 - N_Y_test[self.tstidx[i - 1], :])))
                self.S0R0[i].append(s0r0)
                s0r1 = np.sqrt(np.mean(np.square(Y_pre_s0 - Y_test[self.tstidx[i - 1], :])))
                self.S0R1[i].append(s0r1)
                # sampling 1
                s1r0 = np.sqrt(np.mean(np.square(N_Y_pre_s1 - N_Y_test[self.tstidx[i - 1], :])))
                self.S1R0[i].append(s1r0)
                s1r1 = np.sqrt(np.mean(np.square(Y_pre_s1 - Y_test[self.tstidx[i - 1], :])))
                self.S1R1[i].append(s1r1)
            
        return self.S0R0, self.S0R1, self.S1R0, self.S1R1



    def sample(self, flag):
        """Sample alpha according to flag 0: mean 1: sample"""
        alpha_sample = [None for _ in range(self.n_layers)]
        alpha_test = [None for _ in range(self.n_layers)]
        # log_tau_i = 0
        for i in range(self.n_layers):
            # determine alpha_i
            if flag[i] == 0:
                alpha_sample[i] = self.M[i]
            else:
                Ltril = tf.matrix_band_part(self.L[i], -1, 0)
                Rtril = tf.matrix_band_part(self.R[i], 0, -1)
                phi = tf.random_normal([self.n_samples[i], self.n_bases[i]], dtype=tf.float64)
                alpha_sample[i] = self.M[i] + tf.matmul(tf.matmul(Ltril, phi), Rtril)

            C_i = self.get_C(i)
            
            if i == 0:
                B_i = C_i
                X_test = self.X_test
                X_i = self.X[i]
                n_bases = self.n_bases[i]
            else:
                B_i = tf.concat([B_i, C_i], 0)
                X_test = tf.concat([alpha_test[i - 1], self.X_test], 1)
                subset = tf.gather(alpha_sample[i - 1], np.arange(self.n_samples[i]))                
                X_i = tf.concat([subset, self.X[i]], 1)
                n_bases = self.n_bases[i] - self.n_bases[i - 1]
            
            for j in range(n_bases):
                tmp = tf.reshape(tf.concat(self.U[i][j], 1), [1,-1])
                if j == 0:
                    b_j = tmp
                else:
                    b_j = tf.concat([b_j, tmp], 0)

            if i == 0:
                bases = b_j
            else:
                bases = tf.concat([bases, b_j], 0)

            k1 = self.gaussian_kernel.matrix(X_i, tf.exp(self.log_t1[i]))
            k2 = self.gaussian_kernel.matrix(X_test, tf.exp(self.log_t1[i]))
            k3 = self.gaussian_kernel.cross(X_test, X_i, tf.exp(self.log_t1[i]))

            # K product
            sig11 = k1
            sig22 = k2
            sig21 = k3
            sig12 = tf.transpose(sig21)

            m = tf.matmul(sig21, tf.linalg.solve(sig11, alpha_sample[i]))
            alpha_test[i] = m
            if flag[i] == 1:
                # column variance
                kBB = self.gaussian_kernel.matrix(bases, tf.exp(self.log_t2[i]))
                # row variance
                sig = sig22 - tf.matmul(sig21, tf.linalg.solve(sig11, sig12))

                phi = tf.random_normal([tf.shape(self.X_test)[0], self.n_bases[i]], dtype=tf.float64)
                L = tf.linalg.cholesky(sig)
                R = tf.transpose(tf.linalg.cholesky(kBB))

                alpha_test[i] = m + tf.matmul(tf.matmul(L, phi), R)


        i = self.n_layers - 1
        Y_pre = tf.matmul(alpha_test[i], B_i)


        return Y_pre
    
    def trace(self, A, B):
        """Trace of Matrix Multiplication AB"""
        return tf.reduce_sum(A * tf.transpose(B))

    def fit(self, X_test):
        """Fit test data to trained model"""
        data_dict = {X_i: data.reshape([-1, self.in_d]) for X_i, data in zip(self.X, self.feed_data['X_train'])}
        data_dict.update({self.X_test: X_test.reshape([-1, self.in_d])})
        # data_dict.update({self.phi[i]: np.random.normal(size=[self.n_samples[i], self.n_bases[i]])  for i in range(self.n_layers)})
        Y_pre_s0, Y_pre_s1 = self.sess.run([self.Y_pre_s0, self.Y_pre_s1], feed_dict=data_dict)
        return Y_pre_s0, Y_pre_s1

    def outer_product(self, dim, U):
        n = len(dim)
        idx = ['k','l','m','n']
        left = 'i,j'
        right = 'ji'
        for i in range(2, n):
            left = left +',' +idx[i-2]
            right = idx[i-2] + right
        exp = left+'->'+right
        T = tf.einsum(exp, *U)
        return T
            

    def get_C(self, i):
        if i == 0:
            n_bases = self.n_bases[i]
        else:
            n_bases = self.n_bases[i] - self.n_bases[i - 1]

        for j in range(n_bases):
            U = []
            for k in range(len(self.dim)):
                U.append(tf.reshape(self.U[i][j][k], ([-1,])))
            c = self.outer_product(self.dim, U)
            c = tf.reshape(c, [1, -1])
            if j == 0:
                C = c
            else:
                C = tf.concat([C,c], 0)
        return C

    def get_loss(self):
        """Calculate Loss Function"""
        L = 0
        log_tau_i = 0 # self.log_tau[0]
        
        for i in range(self.n_layers):
            """Layer i"""
            C_i = self.get_C(i)
            # get Ai
            if i == 0:
                B_i = C_i
                n_bases = self.n_bases[i]
            else:
                B_i = tf.concat([B_i, C_i], 0)
                n_bases = self.n_bases[i] - self.n_bases[i - 1]

            for j in range(n_bases):
                tmp = tf.reshape(tf.concat(self.U[i][j], 1), [1,-1])
                if j == 0:
                    b_j = tmp
                else:
                    b_j = tf.concat([b_j, tmp], 0)

            if i == 0:
                bases = b_j
            else:
                bases = tf.concat([bases, b_j], 0)

            Ltril = tf.matrix_band_part(self.L[i], -1, 0)
            Rtril = tf.matrix_band_part(self.R[i], 0, -1)
            phi = tf.random_normal([self.n_samples[i], self.n_bases[i]], dtype=tf.float64)
            self.alpha[i] = self.M[i] + tf.matmul(tf.matmul(Ltril, phi), Rtril)
            if i > 0:
                subset = tf.gather(self.alpha[i - 1], np.arange(self.n_samples[i])) 
                X_i = tf.concat([subset, self.X[i]], 1)
            else:
                X_i = self.X[i]
            k1 = self.gaussian_kernel.matrix(X_i, tf.exp(self.log_t1[i]))
            k2 = self.gaussian_kernel.matrix(bases, tf.exp(self.log_t2[i]))

            ## calculate each term in log-likelihood
            U_ld= tf.reduce_sum(tf.log(tf.diag_part(Ltril)**2))
            V_ld = tf.reduce_sum(tf.log(tf.diag_part(Rtril)**2))
            U = tf.matmul(Ltril, tf.transpose(Ltril))
            V = tf.matmul(tf.transpose(Rtril), Rtril)

            log_tau_i = log_tau_i + self.log_tau[i]

            # lpY = 0.5 * self.n_samples[i] * self.out_d * log_tau_i - 0.5 * tf.exp(log_tau_i) * tf.reduce_sum(tf.square(self.Y[i] - tf.matmul(self.alpha[i], B_i)))
            lpa = -0.5 * self.n_bases[i] * tf.linalg.logdet(k1) -  0.5 * self.n_samples[i] * tf.linalg.logdet(k2) - 0.5 * self.trace(tf.linalg.solve(k2, tf.transpose(self.alpha[i])), tf.linalg.solve(k1, self.alpha[i]))
            BB = tf.matmul(B_i, tf.transpose(B_i))
            lpY = 0.5 * self.n_samples[i] * self.out_d * log_tau_i - 0.5 * tf.exp(log_tau_i) * (tf.reduce_sum(tf.square(self.Y[i] - tf.matmul(self.M[i], B_i))) + tf.linalg.trace(U) * self.trace(V, BB))
            lptau = (self.a0 - 1) * self.log_tau[i] - self.b0 * tf.exp(self.log_tau[i])
            # lq = -0.5 *  self.n_bases[i] * U_ld -  0.5 *  self.n_samples[i] *  V_ld - 0.5 * self.trace(tf.linalg.solve(Rtril, tf.linalg.solve(tf.transpose(Rtril), tf.transpose(self.alpha[i] - self.M[i]))), tf.linalg.solve(tf.transpose(Ltril), tf.linalg.solve(Ltril, self.alpha[i] - self.M[i])))
            h = 0.5 * self.n_bases[i] * U_ld + 0.5* self.n_samples[i] * V_ld
            L = L - (lpY + lpa + lptau - h)
        lu = -0.5 * tf.reduce_sum(bases**2)
        L = L - lu

        return L

