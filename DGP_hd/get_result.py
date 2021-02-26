#%%
import numpy as np
import matplotlib.pyplot as plt

base_list = [5]

s0r0 = np.zeros([4, 5])
s0r1 = np.zeros([4, 5]) 
s1r0 = np.zeros([4, 5])
s1r1 = np.zeros([4, 5]) 
# fig, ax = plt.subplots(nrows=1, ncols=4, subplot_kw=dict(polar=True))
fig = plt.figure()
for k, base in enumerate(base_list):
    S0R0 = None
    S0R1 = None
    S1R0 = None
    S1R1 = None
    for i in range(5):
        # s0r0
        fn = './k'+str(base)+'f'+str(i+1)+'_s0r0.csv'
        tmp00 = np.genfromtxt(fn, delimiter=',').reshape([1,-1])
        s0r0[k][i] = np.min(tmp00[0])
        # s0r1
        fn = './k'+str(base)+'f'+str(i+1)+'_s0r1.csv'
        tmp01 = np.genfromtxt(fn, delimiter=',').reshape([1,-1])
        s0r1[k][i] = np.min(tmp01[0])
        # s1r0
        fn = './k'+str(base)+'f'+str(i+1)+'_s1r0.csv'
        tmp10 = np.genfromtxt(fn, delimiter=',').reshape([1,-1])
        s1r0[k][i] = np.min(tmp10[0])
        # s1r1
        fn = './k'+str(base)+'f'+str(i+1)+'_s1r1.csv'
        tmp11 = np.genfromtxt(fn, delimiter=',').reshape([1,-1])
        s1r1[k][i] = np.min(tmp11[0])

        # sum
        if S0R0 is None:
            S0R0 = tmp00
            S0R1 = tmp01
            S1R0 = tmp10
            S1R1 = tmp11
        l0 = tmp00.shape[1]
        l1 = S0R0.shape[1]
        ex = np.abs(l0 - l1)
        if l0 < l1:
            tmp00 = np.append(tmp00, np.array([tmp00[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
            tmp01 = np.append(tmp01, np.array([tmp01[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
            tmp10 = np.append(tmp10, np.array([tmp10[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
            tmp11 = np.append(tmp11, np.array([tmp11[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
        elif l0 > l1:
            S0R0 = np.append(S0R0, np.array([S0R0[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
            S0R1 = np.append(S0R1, np.array([S0R1[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
            S1R0 = np.append(S1R0, np.array([S1R0[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
            S1R1 = np.append(S1R1, np.array([S1R1[0][-1] for _ in range(ex)]).reshape([1,-1]), axis=1)
        S0R0 = S0R0 + tmp00
        S0R1 = S0R1 + tmp01
        S1R0 = S1R0 + tmp10
        S1R1 = S1R1 + tmp11
    S0R0 = S0R0 / 5
    S0R1 = S0R1 / 5
    S1R0 = S1R0 / 5
    S1R1 = S1R1 / 5

    x = np.arange(0, S0R0.shape[1], 1)
    ax = plt.subplot(2, 4, k+1)
    plt.plot(x, tmp00[0], 'b')
    plt.plot(x, tmp10[0], 'r')
    plt.ylabel('NRMSE')
    
    ax.set_title("k=" + str(base))
    plt.subplot(2, 4, 4 + k + 1)
    plt.plot(x, tmp01[0], 'b')
    plt.plot(x, tmp11[0], 'r')
    plt.ylabel('RMSE')
    plt.xlabel('ITERATION')
print('s0r0:', s0r0)
print('s0r1:', s0r1)
print('s1r0:', s1r0)
print('s1r1:', s1r1)
# plt.show()
fig.savefig('result.png')
            
