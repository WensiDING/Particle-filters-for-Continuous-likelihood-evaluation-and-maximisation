import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# For simplicity, I have just implemented the 1D case
class Kalman_Filter(object):

    def __init__(self, V_epi, mu0=0, V0=1, A=1, b=0, V=1):
        self.mu0 = mu0 # mu0 = mu_tile
        self.V0 = V0 # V0 = sigma2_tile
        self.A = A
        self.b = b
        self.V = V # V = sigma2_eta
        self.V_epi = V_epi
        self.T = None
        
        self.mu_alpha = None
        self.V_alpha = None
        
        self.mu_gamma = None
        self.V_gamma = None
        self.J = None
        
        self.Z1 = None # E[Z_n]
        self.Z2 = None # E[Z_n Z_n-1]
        self.Z3 = None # E[Z_n Z_n]
        
        self.ct = [] # stock p(y_{t} | y_{1:t-1})
        self.log_lokelihood = 0 # stock likelihood for current set of parameters
        self.log_likelihoods = [] # stock the likelihoods when traing a model
        
    def compute_alpha_t(self, y, t):
        '''All sigma are squared: sigma2 == variance'''
        if t == 0:
            denom = self.V0 + self.V_epi
            self.mu_alpha[t] = (y * self.V0 + self.V_epi * self.mu0) / denom
            self.V_alpha[t] = self.V0 * self.V_epi / denom
            self.ct[t] = norm.pdf(y, loc=self.mu0, scale=np.sqrt(denom))
        else:
            sigma2_tmp = self.V + self.A * self.A * self.V_alpha[t-1]
            denom = sigma2_tmp + self.V_epi
            nom = sigma2_tmp * y + self.V_epi * (self.A * self.mu_alpha[t-1] + self.b)
            self.mu_alpha[t] = nom / denom
            self.V_alpha[t] = self.V_epi * sigma2_tmp / denom
            self.ct[t] = norm.pdf(y, loc=self.A*self.mu_alpha[t-1]+self.b, scale=np.sqrt(denom))
        return

    def compute_alpha_hat(self, Y):
        '''This function compute alpha_hat = p(z_t|x_{1:t})'''
        #Y = np.array(Y)
        self.T = T
        if self.mu_alpha is None:
            self.mu_alpha = np.zeros(T)
            self.V_alpha = np.zeros(T)
            self.ct = np.zeros(T)
        for t in range(T):
            self.compute_alpha_t(Y[t], t)
        self.log_likelihood = np.sum(np.log(self.ct))
        self.log_likelihoods.append(self.log_likelihood)
        return
    
    def compute_gamma_t(self, y, t):
        if t == self.T-1:
            self.mu_gamma[t] = self.mu_alpha[t]
            self.V_gamma[t] = self.V_alpha[t]
        else:
            P = self.V + self.A * self.A * self.V_alpha[t]
            J = self.A * self.V_alpha[t] / P
            self.J[t] = J
            self.mu_gamma[t] = self.mu_alpha[t] + J * (self.mu_gamma[t+1] - self.b - self.A * self.mu_alpha[t])
            self.V_gamma[t] = self.V_alpha[t] + J * J * (self.V_gamma[t+1] - P)
        return
    
    def compute_gamma(self, Y):
        '''This function compute gamma(z_t) = p(z_t | x_{1:T})'''
        #Y = np.array(Y)
        if self.mu_gamma is None:
            self.mu_gamma = np.zeros(self.T)
            self.V_gamma = np.zeros(self.T)
            self.J = np.zeros(self.T)
        for t in reversed(range(self.T)):
            self.compute_gamma_t(Y[t], t)
        return
    
    def E_step(self, Y):
        '''This functiom implements the Expectation step for EM learning model parameters'''
        self.compute_alpha_hat(Y)
        self.compute_gamma(Y)
        
        self.Z1 = self.mu_gamma.copy()
        self.Z2 = self.J[:-1] * self.V_gamma[1:] + self.mu_gamma[1:] * self.mu_gamma[:-1]
        self.Z3 = self.V_gamma + self.mu_gamma * self.mu_gamma
        
    def M_step(self, Y):
        '''This function implements the Maximization step for EM learning model parameters'''
        self.mu0 = self.Z1[0]
        self.V0 = self.Z3[0] - self.Z1[0] ** 2
        
        a1, a2, a3, a4 = np.sum(self.Z2), np.mean(self.Z1[1:]), np.mean(self.Z1[:-1]), np.sum(self.Z3[:-1])
        self.b = (a2 * a4 - a1 * a3) / (a4 * (self.T-1) - a3 ** 2)
        self.A = (a1 - a3 * self.b) / a4
        a5 = np.sum(self.Z3[1:])
        self.V = a5 - 2*self.b*a2 + self.b**2 -2*self.A*(a1-self.b*a3) + self.A*self.A*a4
        self.V = self.V / (self.T-1)
        return
    
    def fit(self, Y, epsilon=1e-3, n_iter=1000, verbose=True):
        '''This function train a Kalman filter'''
        Y = np.array(Y)
        self.T = Y.shape[0]
        i = 0
        old_lkh = -np.inf
        while True:
            self.E_step(Y)
            if i%5 == 0 and verbose:
                print('iter ', i, ' log_likelihood = ', self.log_likelihood)
                print(' mu0:', '%0.3f' % self.mu0, 
                      ' V0:', '%0.3f' % self.V0,
                      ' A:', '%0.3f' % self.A,
                      ' b:', '%0.5f' % self.b,
                      ' V:', '%0.3f' % self.V)
            if np.abs(self.log_likelihood - old_lkh) < epsilon:
                break
            old_lkh = self.log_likelihood
            i += 1
            self.M_step(Y)
            if i > n_iter: break
        return

#     @classmethod
#     def compute_log_likelihood(Y, t, ct, mu_alpha, V_alpha, mu0, V0, V_epi, A):
#         Y = np.array(Y)
#         T = Y.shape[0]
#         mu_alpha = np.zeros(T)
#         V_alpha = np.zeros(T)
#         ct = np.zeros(T)
#         for t in range(T):
#             Kalman_Filter._compute_alpha_t(Y[t], t, ct, mu_alpha, V_alpha, mu0, V0, V_epi, A)
#         return np.sum(np.log(ct))
#     @classmethod
#     def _compute_alpha_t(y, t, ct, mu_alpha, V_alpha, mu0, V0, V_epi, A):
#         '''All sigma are squared: sigma2 == variance'''
#         if t == 0:
#             denom = V0 + V_epi
#             mu_alpha[0] = (y * V0 + V_epi * mu0) / denom
#             V_alpha[1] = V0 * V_epi / denom
#             ct[0] = norm.pdf(y, loc=mu0, scale=np.sqrt(denom))
#         else:
#             sigma2_tmp = V + A * A * V_alpha[t-1]
#             denom = sigma2_tmp + V_epi
#             nom = sigma2_tmp * y + V_epi * (A * mu_alpha[t-1] + b)
#             mu_alpha[t] = nom / denom
#             V_alpha[t] = V_epi * sigma2_tmp / denom
#             ct[t] = norm.pdf(y, loc=A*mu_alpha[t-1]+b, scale=np.sqrt(denom))
#         return    

################################################
def generator_ar_1(sigma_epsilon_square=2, sigma_eta_square=0.02, phi=0.975, mu=0.5, T=5000, seed=2345):
    np.random.seed(seed=seed)
    #x = np.random.randn(1)
    x = 0
    ys = np.zeros(T)
    for i in range(T):
        ys[i] = x + np.random.randn(1) * np.sqrt(sigma_epsilon_square)
        x = (x - mu) * phi + mu + np.random.randn(1) * \
            np.sqrt(sigma_eta_square)
    print('sample generated with success !')
    return ys

# parameters
sigma_epsilon_square_0 = 2
sigma_eta_square_0 = 0.02
phi_0 = 0.975
mu_0 = 0.5


# T = 5000
# N = 3500
T = 5000
N = 600
observations = generator_ar_1(sigma_epsilon_square=sigma_epsilon_square_0,
                              sigma_eta_square=sigma_eta_square_0, phi=phi_0, mu=mu_0, T=T)

# mus = [i*0.03 for i in range(10,30)]
# log_likelihoods = []
# for mu in mus:
#     A = phi_0
#     b = (1 - phi_0) * mu
#     kalman_filter = Kalman_Filter(V_epi=sigma_epsilon_square_0, 
#                                   mu0=0, 
#                                   V0=1, 
#                                   A=A, 
#                                   b=b, 
#                                   V=sigma_eta_square_0)
#     kalman_filter.compute_alpha_hat(Y=observations)
#     print('mu=', '%0.2f' % (mu), ' log-lokelihood=', kalman_filter.log_likelihood)
#     log_likelihoods.append(kalman_filter.log_likelihood)

kalman_filter = Kalman_Filter(V_epi=sigma_epsilon_square_0, 
                              mu0=0, 
                              V0=1, 
                              A=1, 
                              b=1, 
                              V=1)
kalman_filter.fit(Y=observations, epsilon=1e-4, n_iter=1000)
plt.figure()
plt.plot(kalman_filter.log_likelihoods)
plt.show()