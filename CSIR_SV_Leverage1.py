import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
# return the mean value of non-normalized weights for loglikelihood estimation
# and the normalized weights for resampling step


def importance_ratio(likelihood_func, y, xs):
    log_weights = [likelihood_func(y, x) for x in xs]
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
    # algo described in the paper A.3.
    pi = np.append(0, weights)
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
    new_particles = np.zeros(N)
    for i in range(T):
        for j in range(N):
            new_particles[j] = transition(initial_particles[j])
        new_particles = np.sort(new_particles)
        likelihood, normalized_weights = importance_ratio(
            likelihood_func, observations[i], new_particles)
        likelihoods[i] = likelihood
        # print(likelihood)
        initial_particles = continuous_stratified_resample(
            normalized_weights, new_particles)
        # print('time step {} finished with likelihood {}'.format(i, likelihood))
    return likelihoods


# AR(1) model

# AR(1) sample generator


def generator_sv_with_leverage(mu=0.5, phi=0.975, sigma_eta_square=0.02, rho=-0.8, T=1000, seed=2345):
    np.random.seed(seed=seed)
    x = 0
    ys = np.zeros(T)
    for i in range(T):
        eta_t = np.random.randn(1)
        x = mu * (1 - phi) + phi * x + np.sqrt(sigma_eta_square) * eta_t
        ys[i] = rho * np.exp(x / 2) * eta_t + np.random.randn(1) * \
            np.sqrt((1 - rho ** 2) * np.exp(x))
    print('sample generated with success !')
    return ys

# generate initial particles with stationary distribution if it exists


def initial_particle(N):
    return np.random.randn(N)

# likelihood function


def likelihood_function(y, x):
    return norm.logpdf(y, loc=0, scale=np.sqrt((1 - rho ** 2) * np.exp(x) + (rho * np.exp(x / 2)) ** 2))


# transition function
def transition_sample(x):
    return mu * (1 - phi) + phi * x + np.sqrt(sigma_eta_square) * np.random.randn(1)

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
N = 300
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


initial_particles = initial_particle(N=N)


# likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                               likelihood_func=likelihood_function, transition=transition_sample, N=N)
# loglikelihood = sum(np.log(likelihoods))
# print(loglikelihood)

mus = [i * 0.02 for i in range(17, 34)]
loglikelihoods = np.zeros(len(mus))

for k in range(len(mus)):
    mu = mus[k]
    likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
                                  likelihood_func=likelihood_function, transition=transition_sample, N=N)
    loglikelihood = sum(np.log(likelihoods))
    loglikelihoods[k] = loglikelihood
    print('log-likelihood calculation finished for mu = {} : {}'.format(mu, loglikelihood))
print(loglikelihoods)
