import EXPUtil as util
import DGP_hd

if __name__ == '__main__':
    folds = 1
    base_list = [5]
    for n_bases in base_list:
        for n_fold in range(folds, folds + 1):
            helper = util.Helper(n_bases, n_fold)

            model = DGP_hd.DGP_hd(helper.get_cfg(), 1e-2)
            s0r0, s0r1, s1r0, s1r1 = model.train()
            helper.save_rmse(s0r0, s0r1, s1r0, s1r1)
            model.sess.close()
