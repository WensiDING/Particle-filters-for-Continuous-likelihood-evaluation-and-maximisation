import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
# return the mean value of non-normalized weights for loglikelihood estimation
# and the normalized weights for resampling step


def importance_ratio(likelihood_func, y, xs, eta):
    log_weights = np.zeros(len(xs))
    for i in range(len(xs)):
        log_weights[i] = likelihood_func(y, xs[i], eta[i])
    maximum = np.max(log_weights)
    weights_ratio = np.exp(log_weights - maximum)
    likelihood = np.mean(weights_ratio) * np.exp(maximum)
    normalized_weights = weights_ratio / sum(weights_ratio)
    return likelihood, normalized_weights

# resampling methods


def continuous_stratified_resample(weights, xs):
    n = len(weights)
    # generate n uniform rvs with stratified method
    u0 = np.random.uniform(size=1)
    u = [(u0 + i) / n for i in range(n)]
    pi = np.zeros(n + 1)
    pi[0] = weights[0] / 2
    pi[n] = weights[-1] / 2
    for i in range(1, n):
        pi[i] = (weights[i] + weights[i - 1]) / 2
    # pi = np.append(0, weights)
    # algo described in the paper A.3.

    r = np.zeros(n)
    u_new = np.zeros(n)
    s = 0
    j = 1

    for i in range(n + 1):
        s = s + pi[i]
        while(j <= n and u[j - 1] <= s):
            r[j - 1] = i
            u_new[j - 1] = (u[j - 1] - (s - pi[i])) / pi[i]
            j = j + 1
    r = r.astype(int)
    x_new = np.zeros(n)
    for k in range(n):
        if r[k] == 0:
            x_new[k] = xs[0]
        elif r[k] == n:
            x_new[k] = xs[-1]
        else:
            x_new[k] = (xs[r[k]] - xs[r[k] - 1]) * u_new[k] + xs[r[k] - 1]
    return x_new


def particle_filter(observations, initial_particles, likelihood_func, transition, N, seed=1234):
    np.random.seed(seed=seed)
    T = len(observations)
    likelihoods = np.zeros(T)
    eta_t = np.random.randn(N)
    for i in range(T):
        initial_particles = np.sort(initial_particles)
        likelihood, normalized_weights = importance_ratio(
            likelihood_func, observations[i], initial_particles, eta_t)
        likelihoods[i] = likelihood
        # print(likelihood)
        new_particles = continuous_stratified_resample(
            normalized_weights, initial_particles)
        eta_t = np.random.randn(N)
        for j in range(N):
            initial_particles[j] = transition(
                new_particles[j], eta_t[j])
        # print(len(np.unique(new_particles)), len(np.unique(initial_particles)))
        # print('time step {} finished with likelihood {}'.format(i, likelihood))
    return likelihoods



def generator_sv_with_leverage(mu=0.5, phi=0.975, sigma_eta_square=0.02, rho=-0.8, T=1000, seed=2345):
    np.random.seed(seed=seed)
    x = 0
    ys = np.zeros(T)
    ys[0] = np.random.randn(1) * np.exp(x / 2)
    for i in range(1,T):
        eta_t = np.random.randn(1)
        epsilon_t = rho * eta_t + np.sqrt(1 - rho**2) * np.random.randn(1)
        x = mu * (1 - phi) + phi * x + np.sqrt(sigma_eta_square) * eta_t
        ys[i] = epsilon_t * np.exp(x / 2)
        
        # ys[i] = rho * np.exp(x / 2) * eta_t + np.random.randn(1) * \
        #     np.sqrt((1 - rho ** 2) * np.exp(x))
        

    print('sample generated with success !')
    return ys

# generate initial particles with stationary distribution if it exists


def initial_particle(N):
    return np.random.randn(N)

# likelihood function


def likelihood_function(y, x, eta):
    return norm.logpdf(y, loc=rho*np.exp(x/2)*eta, scale=np.sqrt(1-rho**2)*np.exp(x / 2))

# transition function


def transition_sample(x, eta):
    return mu * (1 - phi) + phi * x + np.sqrt(sigma_eta_square) * eta


# simulated sample
# parameters
mu_0 = 0.5
phi_0 = 0.975
sigma_eta_square_0 = 0.02
rho_0 = -0.8


phi = 0.975
sigma_eta_square = 0.02
rho = -0.8

T = 1000
N = 600

observations = generator_sv_with_leverage(
    mu=mu_0, phi=phi_0, sigma_eta_square=sigma_eta_square_0, rho=rho_0, T=T)


# mus = [i * 0.01 for i in range(10, 38)]
# # S&P 500 Historical data
# # parameters
# phi = 0.98
# sigma_eta_square = 0.03
# rho = 0
# T = 2000
# N = 500
# sp = pd.read_csv('S&P 500 Historical Data.csv')
# sp = sp.set_index('Date')
# sp.index = pd.to_datetime(sp.index)
# close_price = np.asarray(sp['19950515':'20030424']['Close'])
# dr2000 = (close_price[1:] - close_price[:2000]) / close_price[:2000]

# observations = dr2000
# print(observations)
# plt.hist(observations)
# plt.show()



# likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                               likelihood_func=likelihood_function, transition=transition_sample, N=N)
# loglikelihood = sum(np.log(likelihoods))
# print(loglikelihood)
mus = [i * 0.05 for i in range(6, 16)]
cumulated_likelihoods = np.zeros((25,len(mus)))
estimations = np.zeros(25)
for seed in range(25,50):
    print('iteration {}'.format(seed))
    initial_particles = initial_particle(N=N)

    loglikelihoods = np.zeros(len(mus))

    for k in range(len(mus)):
        mu = mus[k]
        likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
                                      likelihood_func=likelihood_function, transition=transition_sample, N=N,seed=seed)
        loglikelihood = sum(np.log(likelihoods))
        loglikelihoods[k] = loglikelihood
        # print(
        #     'log-likelihood calculation finished for mu = {} : {}'.format(mu, loglikelihood))
    print(loglikelihoods)
    # plt.plot(mus, loglikelihoods)
    # plt.show()
    estimations[seed-25] = mus[np.argmax(loglikelihoods)]
    cumulated_likelihoods[seed-25,:] = loglikelihoods
print(estimations)
plt.hist(estimations)
plt.show()
means = np.mean(cumulated_likelihoods,axis=0) 
variances = np.var(cumulated_likelihoods,axis=0)
print(means)
print(variances)
plt.plot(mus,means,'r--',mus,means+np.sqrt(variances),'b--',mus,means-np.sqrt(variances),'b--')
plt.show()

