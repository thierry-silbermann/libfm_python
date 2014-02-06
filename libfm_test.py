import numpy as np
import random
import sys
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from libfm_sparse_v2 import DataMetaInfo 
from libfm_sparse_v2 import Data 
from libfm_sparse_v2 import libFM
from libfm_sparse_v2 import MCMC_learn
import unittest

class Initialisation():
    def __init__(self):
        train_file = 'data/small_train.libfm' 
        test_file = 'data/small_test.libfm'

        def get_num_attribute(filename):
            has_feature = False
            num_feature = 0
            with open(filename, 'r') as f:
                for line in f:
                    spl = line.split()
                    for i in range(1,len(spl)):
                        _feature, _value = map(float, spl[i].split(':'))
                        num_feature = max(_feature, num_feature)
                        has_feature = True
            if has_feature:    
                num_feature += 1 # number of feature is bigger (by one) than the largest value
            return num_feature

        self.num_all_attribute = max(get_num_attribute(train_file), get_num_attribute(test_file))

        self.train = Data(train_file, False, True, self.num_all_attribute)
        self.test = Data(test_file, False, True, self.num_all_attribute)
        

# Here's our "unit tests".
class DataTests(unittest.TestCase):

    #'''
    def testData(self):
        init = Initialisation()
        self.assertTrue((init.train.data.col == [0,5,1,5,2,5,3,5,4,5,0,6,1,6,2,6,3,6,4,6,1,7,3,7,0,8,2,8,4,8]).all())
        self.assertTrue((init.train.data.row == [ 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14]).all() )
        self.assertTrue((init.train.data.data == [1.]*30 ).all() )
    
    def test_w0_ALS(self):
        init = Initialisation()
        train, test, num_all_attribute = init.train, init.test, init.num_all_attribute
        meta = DataMetaInfo(num_all_attribute)
        fm = libFM(num_all_attribute, seed=1, method='als', num_iter=1, dim='1,0,0',
                param_regular='0,0,0.1', init_stdev=0.1)
        mcmc = MCMC_learn(fm, meta, train, test, 0)
        mcmc.learn()
        self.assertAlmostEqual(fm.w0, 3.26666666667, places=7, msg=None, delta=None)
        np.testing.assert_array_almost_equal(mcmc.predict(), [ 3.26666667, 3.26666667, 3.26666667, 3.26666667], decimal=6, err_msg='', verbose=True)
        
    def test_w0_MCMC(self): 
        init = Initialisation()
        train, test, num_all_attribute = init.train, init.test, init.num_all_attribute
        meta = DataMetaInfo(num_all_attribute)
        fm = libFM(num_all_attribute, seed=3, method='mcmc', num_iter=1000, dim='1,0,0',
                param_regular='0,0,0.1', init_stdev=0.1)
        mcmc = MCMC_learn(fm, meta, train, test, 0)
        mcmc.learn()
        np.testing.assert_array_almost_equal(mcmc.predict(), [ 3.266, 3.266, 3.266, 3.266], decimal=1, err_msg='', verbose=True)
    
    def test_w_ALS(self): 
        init = Initialisation()
        train, test, num_all_attribute = init.train, init.test, init.num_all_attribute
        meta = DataMetaInfo(num_all_attribute)
        fm = libFM(num_all_attribute, seed=1, method='als', num_iter=2, dim='0,1,0',
                param_regular='0,0,0.1', init_stdev=0.1)
        fm.w = np.array( [ 0.16243454, -0.06117564, -0.05281718, -0.10729686, 0.08654076, -0.23015387,
                            0.17448118, -0.07612069, 0.03190391] )
        mcmc = MCMC_learn(fm, meta, train, test, 0)
        mcmc.learn()
        np.testing.assert_array_almost_equal(mcmc.predict(), [ 2.24725653, 3.91392319, 2.64163236, 3.9749657 ], decimal=5, err_msg='', verbose=True)
    
    def test_v_ALS(self):
        init = Initialisation()
        train, test, num_all_attribute = init.train, init.test, init.num_all_attribute
        meta = DataMetaInfo(num_all_attribute)
        fm = libFM(num_all_attribute, seed=1, method='als', num_iter=2, dim='0,0,3',
                param_regular='0,0,0.1', init_stdev=0.1)

        fm.v = np.asarray([[-0.02493704,0.14621079,-0.20601407,-0.03224172,-0.03840544,0.11337694,-0.10998913,-0.01724282,-0.08778584],
                [0.00422137,0.05828152,-0.11006192,0.11447237,0.09015907,0.05024943,0.09008559,-0.06837279,-0.01228902],
                [-0.09357694,-0.02678881,0.05303555,-0.06916608,-0.03967535,-0.06871727,-0.08452056,-0.06712461,-0.00126646]])

        mcmc = MCMC_learn(fm, meta, train, test, 0)
        mcmc.learn()
        np.testing.assert_array_almost_equal(mcmc.predict(), [3.71472717, 5.0, 1.0, 5.0], decimal=5, err_msg='', verbose=True)
     
    #'''   
    def test_ALS(self):
        init = Initialisation()
        train, test, num_all_attribute = init.train, init.test, init.num_all_attribute
        meta = DataMetaInfo(num_all_attribute)
        fm = libFM(num_all_attribute, seed=1, method='als', num_iter=20, dim='1,1,3',
                param_regular='0,0,0.1', init_stdev=0.1)
        fm.w = np.array( [ 0.16243454, -0.06117564, -0.05281718, -0.10729686, 0.08654076, -0.23015387,
                            0.17448118, -0.07612069, 0.03190391] )
        fm.v = np.asarray([[-0.02493704,0.14621079,-0.20601407,-0.03224172,-0.03840544,0.11337694,-0.10998913,-0.01724282,-0.08778584],
                [0.00422137,0.05828152,-0.11006192,0.11447237,0.09015907,0.05024943,0.09008559,-0.06837279,-0.01228902],
                [-0.09357694,-0.02678881,0.05303555,-0.06916608,-0.03967535,-0.06871727,-0.08452056,-0.06712461,-0.00126646]])

        mcmc = MCMC_learn(fm, meta, train, test, 0)
        mcmc.learn()
        
        np.testing.assert_array_almost_equal(mcmc.predict(), [1.0, 4.73393, 1.05236, 5.0], decimal=4, err_msg='', verbose=True)      
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
