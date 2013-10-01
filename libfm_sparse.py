import numpy as np
import random
import sys
import scipy.sparse as sps
from scipy.sparse import coo_matrix


####################################
####################################
####################################

class libFM:
    """
    Parameters
    ----------

    learn_rate:
        learn_rate for SGD; default=0.1
    num_iter : int
        Number of iterations
    dim : 'k0,k1,k2': k0=use bias, k1=use 1-way interactions,
            k2=dim of 2-way interactions; default=1,1,8
    param_regular
        'r0,r1,r2' for SGD and ALS: r0=bias regularization,
        r1=1-way regularization, r2=2-way regularization
    task : string
        regression: Labels are real values.
        classification: Labels are either positive or negative.
    verbose : bool
        Whether or not to print current iteration, training error
    init_stdev : double, optional
        Standard deviation for initialization of 2-way factors.
        Defaults to 0.01.
    method:
        learning method (SGD, SGDA, ALS, MCMC); default=MCMC
    seed : int
        The seed of the pseudo random number generator
    """
    
    def __init__(self, num_attribute, learn_rate=0.01, num_iter=2, dim=(1,0,1),
                param_regular=(0,0,0.1), init_stdev=0.1, task='regression', 
                method='mcmc', verbose=True, seed=None, output_file='output.csv'):
        
        self.num_attribute = num_attribute
        self.num_iter = num_iter
        #self.learn_rate = learn_rate
        self.init_stdev = init_stdev
        self.dim = dim
        self.task = task
        self.method = method
        
        self.k0 = dim[0] != 0
        self.k1 = dim[1] != 0
        self.num_factor = dim[2]    
        
        # Regularization Parameters
        if method == 'mcmc':
            self.reg0, self.regw, self.regv = 0.0, 0.0, 0.0
        else:
            self.reg0 = param_regular[0]
            self.regw = param_regular[1]
            self.regv = param_regular[2]
        
        self.verbose = verbose
        self.seed = seed
        
        if self.seed > -1:
            np.random.seed(seed=self.seed)
        
        init_mean = 0
        self.w0 = 0
        self.w = np.random.normal(init_mean, init_stdev, self.num_attribute)
        self.v = np.random.normal(init_mean, init_stdev, (self.num_factor, self.num_attribute))

        #m_sum = np.zeros(self.num_factor)
        #m_sum_sqr = np.zeros(self.num_factor)
    
        self.do_sample = True
        self.do_multilevel = True
        self.save = True
        self.output_file = output_file

####################################
####################################
####################################
   
class MCMC_learn:

    def __init__(self, fm, meta, train, test):
        self.fm = fm
        self.meta = meta
        self.num_iter = fm.num_iter
        self.num_eval_cases = test.num_cases
        
        self.min_target = train.min_target
        self.max_target = train.max_target
        
        self.train = train
        self.test = test
        
        self.cache_for_group_values = np.zeros(meta.num_attr_groups)

        self.alpha_0, self.gamma_0, self.beta_0, self.mu_0  = 1.0, 1.0, 1.0, 0.0 
        self.alpha = 1.0
        
        self.w0_mean_0 = 0.0 
 
        self.w_mu = np.zeros(meta.num_attr_groups, dtype=float)
        self.w_lambda = fm.regw * np.ones(meta.num_attr_groups, dtype=float) 
        
        self.v_mu = np.zeros((meta.num_attr_groups, fm.num_factor), dtype=float)
        self.v_lambda = fm.regv * np.ones((meta.num_attr_groups, fm.num_factor), dtype=float) 
        
        self.pred_sum_all = np.zeros(test.num_cases, dtype=float) 
        self.pred_this = np.zeros(test.num_cases, dtype=float) 

        self.cache  = np.zeros((2, train.num_cases),dtype=float) #e_q_term 
        self.cache_test = np.zeros((2, test.num_cases), dtype=float) #e_q_term 
        
    def learn(self):

        self.fm.reg0, self.fm.regw, self.fm.regv = 0.0, 0.0, 0.0
        self.predict_data_and_write_to_eterms()
        
        if self.fm.task == 'regression':
            # remove the target from each prediction, because: e(c) := \hat{y}(c) - target(c)
            self.cache[0] -= self.train.target_value
        else:
            raise Exception("Unknown task")
        
        for i in xrange(self.num_iter):
            self.draw_all()
            self.predict_data_and_write_to_eterms()

            acc_train = 0.0
            rmse_train = 0.0
            if self.fm.task == 'regression':
                # evaluate test and store it
                tmp = np.copy(self.cache_test[0])
                #print 'tmp', tmp
                pred_this = np.copy(tmp)
                tmp = np.clip(tmp, self.min_target, self.max_target)
                self.pred_sum_all += tmp
                
                # Evaluate the training dataset and update the e-terms 
                tmp = np.copy(self.cache[0])
                tmp = np.clip(tmp, self.min_target, self.max_target)
                err = tmp - self.train.target_value
                rmse_train = np.sum(err*err)
                self.cache[0] -= self.train.target_value
                rmse_train = np.sqrt(rmse_train/self.train.num_cases)
            elif self.fm.task == 'classification':
                continue
            else:
                raise Exception('Unknown task')
            #Evaluate the test data set
            if self.fm.task == 'regression':
                rmse_test_this, mae_test_this = self.evaluate(self.pred_this, self.test.target_value, 1.0, 0, self.num_eval_cases)
                rmse_test_all, mae_test_all = self.evaluate(self.pred_sum_all, self.test.target_value, 1.0/(i+1), 0, self.num_eval_cases)
                print "#Iter=", i, "\tTrain=", rmse_train, "\tTest=", rmse_test_all

            else:
                raise Exception('Unknown task')
        
        print 'w0:', self.fm.w0
        print 'w:', self.fm.w
        print 'v:', self.fm.v
        
        if self.fm.save:
            print self.test.target_value, self.test.num_feature, self.test.num_values, self.test.num_cases
            print np.mean(self.test.target_value)
            pred = self.predict()
            np.savetxt(self.fm.output_file, pred, delimiter=",", fmt='%.10f') #default fmt='%.18e'
    
    def predict(self): 

        if self.fm.do_sample:
            assert(self.test.num_cases == self.pred_sum_all.shape[0])
            out = self.pred_sum_all / self.num_iter
        else:
            assert(self.test.num_cases == self.pred_this.shape[0])
            out = np.copy(self.pred_this)
        
        print out
        if self.fm.task == 'regression':
            out = np.clip(out, self.min_target, self.max_target)
        elif self.fm.task == 'classification':
            out = np.clip(out, 0.0, 1.0)
        else:
            raise Exception('Unknown task')    
        
        return out

    
    def predict_data_and_write_to_eterms(self): #Ok

        self.cache = np.zeros_like(self.cache) 
        self.cache_test = np.zeros_like(self.cache_test) 
        
        #print '############ (1)'
        #print 'v', self.fm.v
        
        # (1) do the 1/2 sum_f (sum_i v_if x_i)^2 and store it in the e/y-term
        for f in xrange(self.fm.num_factor):
            v = self.fm.v[f] 
        
            # calculate cache[i].q = sum_i v_if x_i (== q_f-term)
            # Complexity: O(N_z(X^M))
            self.cache[1] += v * self.train.data_t
            self.cache_test[1] += v * self.test.data_t
            
            #print 'q cache', self.cache[1]
            #print 'q cache_t', self.cache_test[1]
      
            # add 0.5*q^2 to e and set q to zero.
            # O(n*|B|)
            self.cache[0] += 0.5 * self.cache[1] * self.cache[1]
            self.cache[1] = np.zeros_like(self.cache[1])
            self.cache_test[0] += 0.5 * self.cache_test[1] * self.cache_test[1]
            self.cache_test[1] = np.zeros_like(self.cache_test[1])
     
            #print 'e cache', self.cache[0]
            #print 'e cache_t', self.cache_test[0]
            
     
        #print '\n (2) do -1/2 sum_f (sum_i v_if^2 x_i^2) and store it in the q-term'
     
        # (2) do -1/2 sum_f (sum_i v_if^2 x_i^2) and store it in the q-term
        for f in xrange(self.fm.num_factor):
            v = self.fm.v[f]

            # sum up the q^S_f terms in the main-q-cache: 0.5*sum_i (v_if x_i)^2 (== q^S_f-term)
            # Complexity: O(N_z(X^M))
            self.cache[1] -= 0.5 * (v * v) * self.train.data_t.multiply(self.train.data_t)
            self.cache_test[1] -= 0.5 * (v * v) * self.test.data_t.multiply(self.test.data_t)
            
            #print 'q cache', self.cache[1]
            #print 'q cache_t', self.cache_test[1]
        
        #print '\n (3) add the w to the q-term '
        #print 'w', self.fm.w
        # (3) add the w's to the q-term    
        if self.fm.k1:
            self.cache[1] += self.fm.w * self.train.data_t
            self.cache_test[1] += self.fm.w * self.test.data_t
        
            #print 'q cache', self.cache[1]
            #print 'q cache_t', self.cache_test[1]
        
        #print '\n (3) merge both for getting the prediction: w0+e(c)+q(c)'
        # (3) merge both for getting the prediction: w0+e(c)+q(c)
      
        self.cache[0] += self.cache[1]
        self.cache_test[0] += self.cache_test[1]
        if self.fm.k0:
            self.cache[0] += self.fm.w0
            self.cache_test[0] += self.fm.w0
        self.cache[1], self.cache_test[1] = np.zeros_like(self.cache[1]), np.zeros_like(self.cache_test[1]) 
        
        #print 'e cache', self.cache[0]
        #print 'e cache_t', self.cache_test[0]
       
        #print '(3)', self.cache_test[0]
       
    #rmse_test_all, mae_test_all = self.evaluate(self.pred_sum_all, test.target_value, 1.0/(i+1), 0, self.num_eval_cases)   
    def evaluate(self, pred, target, normalizer, from_case, to_case):
        assert(pred.shape[0] == target.shape[0])
        _rmse, _mae = 0, 0
        
        end = min(pred.shape[0], to_case)
        
        tmp = pred[:end] * normalizer
        tmp = np.clip(tmp, self.min_target, self.max_target)
        err = tmp - target[:end]
        _rmse, _mae = np.sum(err*err), np.sum(np.absolute(err))
        
        num_cases = end
        
        rmse = np.sqrt(_rmse/num_cases)
        mae = _mae/num_cases
        
        return rmse, mae
        
    def draw_all(self):
        
        self.draw_alpha()
        if self.fm.k0 :
            self.draw_w0()
            
        if self.fm.k1:
            self.draw_w_lambda()
            self.draw_w_mu()

            # draw the w from their posterior
            g = self.meta.attr_group
            self.draw_w(self.w_mu[g], self.w_lambda[g])
        
        if self.fm.num_factor > 0:
            self.draw_v_lambda()
            self.draw_v_mu()
            
        for f in xrange(self.fm.num_factor):

            self.cache[1] = np.zeros_like(self.cache[1])
            # add the q(f)-terms to the main relation q-cache (using only the transpose data)
            self.cache[1] += self.fm.v[f]  * self.train.data_t
            
            # draw the thetas from their posterior
            g = self.meta.attr_group
            self.draw_v(f, self.v_mu[g,f], self.v_lambda[g,f])
            
    # Find the optimal value for the global bias (0-way interaction)
    def draw_w0(self): #ok
        
        assert(self.train.num_cases == self.cache[0].shape[0])
        print 'alpha', self.alpha
        w0_mean = np.sum(self.cache[0] - self.fm.w0) 
        w0_sigma_sqr = 1.0 / (self.fm.reg0 + self.alpha * self.train.num_cases)
        w0_mean = - w0_sigma_sqr * (self.alpha * w0_mean - self.w0_mean_0 * self.fm.reg0)
        
        # update w0
        w0_old = self.fm.w0
        
        print 'draw_w0: mean, sigsqr', w0_mean, w0_sigma_sqr

        if self.fm.do_sample:
            self.fm.w0 = self.ran_gaussian(w0_mean, np.sqrt(w0_sigma_sqr))
        else:
            self.fm.w0 = w0_mean

        # update error
        self.cache[0] -= (w0_old - self.fm.w0)
        print 'draw_w0: e', self.cache[0]
   
    
    # Find the optimal value for the 1-way interaction w
    def draw_w(self, w_mu, w_lambda):
        x_li = self.train.data_t.tocsr()
        nnz_per_row = np.diff(x_li.indptr)
        Y = sps.csr_matrix((x_li.data * np.repeat(self.fm.w, nnz_per_row), x_li.indices, x_li.indptr), shape=x_li.shape)
        Y.data -= np.take(self.cache[1], Y.indices)
        h = x_li.multiply(-Y)
        h.data *= np.take(self.cache[0], h.indices)
        
        w_mean = np.asarray(h.sum(axis=1).transpose())[0]
        w_sigma_sqr = np.asarray(( x_li.multiply(x_li) ).sum(axis=1).transpose())[0]
        
        w_sigma_sqr = 1.0 / (w_lambda + self.alpha * w_sigma_sqr)
        w_mean = - w_sigma_sqr * (self.alpha * w_mean - w_mu * w_lambda)

        # update w:
        w_old = np.copy(self.fm.w)
        
        print 'draw_w: w_mu, w_lambda', w_mu, w_lambda
        print 'draw_w: w_mean, w_sigma_sqr', w_mean, w_sigma_sqr

        if ( np.isnan(np.sum(w_sigma_sqr)) or np.isinf(np.sum(w_sigma_sqr)) ) :
            self.fm.w = 0.0
            raise Exception('w NaN')
        else: 
            if self.fm.do_sample : 
                self.fm.w = self.ran_gaussian(w_mean, np.sqrt(w_sigma_sqr))
            else:
                self.fm.w = w_mean
            
        # update error:
        self.cache[0] -= (w_old - self.fm.w) * x_li
        
        print 'draw_w: e', self.cache[0]

        
    # Find the optimal value for the 2-way interaction parameter v
    def draw_v(self, f, v_mu, v_lambda): 
        x_li = self.train.data_t.tocsr()
        v = self.fm.v[f]
        cache_li = self.cache
        
        nnz_per_row = np.diff(x_li.indptr) # number of nonzero entries per row of X
        # multiply every row by the corresponding entry of v
        Y = sps.csr_matrix((x_li.data * np.repeat(v, nnz_per_row), x_li.indices, x_li.indptr), shape=x_li.shape)
        Y.data -= np.take(cache_li[1], Y.indices) # subtract from the non-zero entries the corresponding column value in c...
        h = x_li.multiply(-Y)
        v_sigma_sqr = np.asarray((h.multiply(h)).sum(axis=1).transpose())[0]
        h.data *= np.take(cache_li[0], h.indices)
        v_mean = np.asarray(h.sum(axis=1).transpose())[0]
        
        v_mean -= v * v_sigma_sqr
        v_sigma_sqr = 1.0 / (v_lambda + self.alpha * v_sigma_sqr)
        v_mean = - v_sigma_sqr * (self.alpha * v_mean - v_mu * v_lambda)
        
        # update v:
        v_old = np.copy(v)
        
        print 'draw_v: v_mean, v_sigma_sqr', v_mean, v_sigma_sqr

        if ( np.isnan(np.sum(v_sigma_sqr)) or np.isinf(np.sum(v_sigma_sqr)) ) :
            #v[np.isnan(v)] = 0
            #v[np.isinf(v)] = 0
            v = 0.0
            raise Exception('v NaN')
        else:
            if self.fm.do_sample:
                self.fm.v[f] = self.ran_gaussian(v_mean, np.sqrt(v_sigma_sqr))
            else:
                self.fm.v[f] = np.copy(v_mean)
    
        # check for out of bounds values
        #Check limit TODO

        # update error and q:
        nnz_per_row = np.diff(x_li.indptr)
        Y = sps.csr_matrix((x_li.data * np.repeat(v, nnz_per_row), x_li.indices, x_li.indptr), shape=x_li.shape)
        Y.data -= np.take(cache_li[1], Y.indices)
        h = x_li.multiply(-Y)
        
        self.cache[1] -= (v_old - v) * x_li
        self.cache[0] -= (v_old - v) * h
        
        print 'draw_v, q', self.cache[1]
        print 'draw_v, e', self.cache[0]

        
    def draw_alpha(self): #ok
        if not self.fm.do_multilevel:
            self.alpha = self.alpha_0
            return
        
        alpha_n = self.alpha_0 + self.train.num_cases
        gamma_n = self.gamma_0
        
        gamma_n = np.sum(self.cache[0] * self.cache[0])
        
        #alpha_old = self.alpha
        self.alpha = self.ran_gamma(alpha_n / 2.0, gamma_n / 2.0) #TODO ran_gamma
        
        #Check limit TODO
        
    def draw_w_mu(self):
        if not self.fm.do_multilevel:
            self.w_mu = self.mu_0 * np.ones(self.w_mu) 
            return

        w_mu_mean = self.cache_for_group_values
        w_mu_mean = np.zeros_like(w_mu_mean)
        g = self.meta.attr_group
        w_mu_mean = np.bincount(g, weights=self.fm.w)
        
        w_mu_mean = (w_mu_mean + self.beta_0 * self.mu_0) / (self.meta.num_attr_per_group + self.beta_0)
        w_mu_sigma_sqr = 1.0 / ((self.meta.num_attr_per_group + self.beta_0) * self.w_lambda)
        w_mu_old = self.w_mu
        
        if self.fm.do_sample:
            self.w_mu = self.ran_gaussian(w_mu_mean, np.sqrt(w_mu_sigma_sqr))
        else:
            self.w_mu = w_mu_mean

            # check for out of bounds values 
            #Check limit TODO

    def draw_w_lambda(self): #Ok
        if not self.fm.do_multilevel:
            return
        
        
        #w_lambda_gamma = self.cache_for_group_values
        w_lambda_gamma = self.beta_0 * (self.w_mu - self.mu_0) * (self.w_mu - self.mu_0) + self.gamma_0
        
        g = self.meta.attr_group
        w_lambda_gamma += np.bincount(g, weights=(self.fm.w - self.w_mu[g]) * (self.fm.w - self.w_mu[g]))
        
        g = np.unique(g)
        w_lambda_alpha = self.alpha_0 + self.meta.num_attr_per_group[g] + 1
        #w_lambda_old = self.w_lambda
        

        if self.fm.do_sample:
            self.w_lambda = self.ran_gamma(w_lambda_alpha / 2.0, w_lambda_gamma / 2.0)
        else:
            self.w_lambda = w_lambda_alpha/w_lambda_gamma
 
        # check for out of bounds values
        
    def draw_v_mu(self):
        if not self.fm.do_multilevel:
            self.v_mu = self.mu_0 * np.ones(self.v_mu) 
            return

        v_mu_mean = self.cache_for_group_values
        
        g = self.meta.attr_group
        for f in xrange(self.fm.num_factor):
            v_mu_mean = np.zeros_like(v_mu_mean)
            v_mu_mean = np.bincount(g, weights=self.fm.v[f])
 
            v_mu_mean = (v_mu_mean + self.beta_0 * self.mu_0) / (self.meta.num_attr_per_group + self.beta_0)
            v_mu_sigma_sqr = 1.0 / ((self.meta.num_attr_per_group + self.beta_0) * self.v_lambda[:, f])
            #v_mu_old = self.v_mu[:,f]
            
            if self.fm.do_sample:
                self.v_mu[:,f] = self.ran_gaussian(v_mu_mean, np.sqrt(v_mu_sigma_sqr))
            else:
                self.v_mu[:,f] = v_mu_mean
       
    def draw_v_lambda(self):
        if not self.fm.do_multilevel:
            return
            
        for f in xrange(self.fm.num_factor):
            print '# Factor:', f
            v_lambda_gamma = self.beta_0 * (self.v_mu[:,f] - self.mu_0) * (self.v_mu[:,f] - self.mu_0) + self.gamma_0
            print '1st loop', v_lambda_gamma
            
            g = self.meta.attr_group
            v_lambda_gamma += np.bincount(g, weights=((self.fm.v[f,:] - self.v_mu[g,f]) * (self.fm.v[f,:] - self.v_mu[g,f])) )
            print '2nd loop', v_lambda_gamma

            g = np.unique(g)
            v_lambda_alpha = self.alpha_0 + self.meta.num_attr_per_group[g] + 1
            print 'v_lambda_alpha'
            #v_lambda_old = self.v_lambda[:,f]
            if self.fm.do_sample:
                print 'Before v_lambda[:,f]', self.v_lambda[:,f]
                self.v_lambda[:,f] = self.ran_gamma(v_lambda_alpha / 2.0, v_lambda_gamma / 2.0)
                print 'After v_lambda[:,f]', self.v_lambda[:,f]
            else:
                self.v_lambda[:,f] = v_lambda_alpha / v_lambda_gamma
       

    ##################################
    ########### Random.h #############
    ##################################

    def ran_gaussian(self, mean, stdev):
        return mean + stdev * np.random.randn()
        
    def ran_gamma(self, alpha, beta):
        tmp = np.random.gamma(alpha, 1/beta)
        if isinstance(tmp, float):
            return np.asarray([tmp])
        else:
            return tmp
        


####################################
####################################
####################################
 
class Data:
       
    def __init__(self, filename, has_x, has_xt, max_feature):
    
        self.filename = filename
        self.has_x = has_x #False
        self.has_xt = has_xt #True
       
        num_rows = 0
        num_values = 0
        num_feature = 0
        has_feature = False
        self.min_target = float("inf")
        self.max_target = -float("inf")
    
        # (1) determine the number of rows and the maximum feature_id
        with open(filename, 'r') as f:
            for line in f:
                spl = line.split()
                _value = float(spl[0])
                self.min_target = min(_value, self.min_target)
                self.max_target = max(_value, self.max_target)    
                num_rows += 1
                #print spl
                for i in range(1,len(spl)):
                    _feature, _value = map(float, spl[i].split(':'))
                    num_feature = max(_feature, num_feature)
                    has_feature = True
                    num_values += 1
        if has_feature:    
            num_feature += 1 # number of feature is bigger (by one) than the largest value
        print "num_rows=", num_rows, "\tnum_values=" ,num_values, "\tnum_features=", num_feature, "\tmin_target=", self.min_target, "\tmax_target=", self.max_target
        
        assert(num_feature <= max_feature)
        
        rows    = np.zeros(num_values)
        cols    = np.zeros(num_values)
        values  = np.zeros(num_values)
        
        self.target_value = np.zeros(num_rows)
        self.num_feature = max_feature
        self.num_values  = num_values
        
        # (2) read the data   
        row_id = 0
        cacheID = 0
        with open(filename, 'r') as f:
            for line in f:
                spl = line.split()
                assert(row_id < num_rows)
                self.target_value[row_id] = float(spl[0])
                
                for i in range(1,len(spl)):
                    assert(cacheID < num_values)
                    _feature, _value = map(float, spl[i].split(':'))
                    rows[cacheID] = row_id
                    cols[cacheID] = _feature
                    values[cacheID] = _value
                    cacheID += 1

                row_id += 1
           
        assert(num_rows == row_id)
        assert(num_values == cacheID)  
        
        self.data = coo_matrix((values,(rows, cols)), shape=(num_rows, max_feature))
        self.num_cases = num_rows 

        if has_xt:
            self.data_t = self.data.transpose()

####################################
####################################
####################################

class DataMetaInfo:
    def __init__(self, num_attributes):
        self.attr_group = np.zeros(num_attributes, dtype=int)
        self.num_attr_groups = 1
        self.num_attr_per_group = np.zeros(self.num_attr_groups)
        self.num_attr_per_group[0] = num_attributes

####################################
####################################
####################################

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
    
             
def main():
    train_file = 'data/small_train.libfm' #small_
    test_file = 'data/small_test.libfm'
    
    num_all_attribute = max(get_num_attribute(train_file), get_num_attribute(test_file))
    
    train = Data(train_file, False, True, num_all_attribute)
    test = Data(test_file, False, True, num_all_attribute)
    
    assert(num_all_attribute == max(train.num_feature, test.num_feature))
    
    meta = DataMetaInfo(num_all_attribute)
    fm = libFM(num_all_attribute, seed=1)
    mcmc = MCMC_learn(fm, meta, train, test)
    mcmc.learn()
    
if __name__ == "__main__":
    main()

                    
