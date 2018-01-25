import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd


def generator(beta0=0.01, beta1=0.2, beta2=0.75, sigma=0.1, T=5000, seed=2345):
    np.random.seed(seed=seed)
    sigma_t_square = 1
    ys = np.zeros(T)
    for i in range(T):
        x_t = np.random.randn(1) * np.sqrt(sigma_t_square)
        ys[i] = x_t + np.random.randn(1) * sigma
        sigma_t_square = beta0 + beta1 * (x_t**2) + beta2 * sigma_t_square
    print('sample generated with success !')
    return ys


def initial_particle(N):
    return np.random.uniform(0, 2, N)


def likelihood_function(y, sigma_t_square):
    return norm.logpdf(y, loc=0, scale=np.sqrt(sigma_t_square + sigma**2))


def transition_sample(y, sigma_t_square):
    b_square = (sigma**2) * sigma_t_square / (sigma**2 + sigma_t_square)
    x = b_square * y / (sigma**2) + np.random.randn(1) * np.sqrt(b_square)
    return beta0 + beta1 * (x**2) + beta2 * sigma_t_square


def importance_ratio(likelihood_func, y, xs):
    log_weights = [likelihood_func(y, x) for x in xs]
    maximum = np.max(log_weights)
    weights_ratio = np.exp(log_weights - maximum)
    likelihood = np.mean(weights_ratio) * np.exp(maximum)
    normalized_weights = weights_ratio / sum(weights_ratio)
    return likelihood, normalized_weights


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
    for i in range(T):
        initial_particles = np.sort(initial_particles)
        likelihood, normalized_weights = importance_ratio(
            likelihood_func, observations[i], initial_particles)
        likelihoods[i] = likelihood
        initial_particles = continuous_stratified_resample(
            normalized_weights, initial_particles)
        for j in range(N):
            initial_particles[j] = transition(
                observations[i], initial_particles[j])
        # print('time step {} finished with likelihood {}'.format(i, likelihood))
    return likelihoods

######################
# parameters section #
######################

# real parameters for generating samples
beta0_0 = 0.01
beta1_0 = 0.2
beta2_0 = 0.75
sigma_0 = 0.1

# test parameters for estimating likelihood
beta0 = 0.01
beta1 = 0.2
# beta2 = 0.75
sigma = 0.1

# specify N for number of particles, T for length of obervations
T = 500
N = 500


#######################
# observation section #
#######################
# 1. sample as observation
observations = generator(beta0=beta0_0, beta1=beta1_0,
                         beta2=beta2_0, sigma=sigma_0, T=T)
# # 2. real data as observation(ex: uk pound vs us dollar data)
# from dateutil import parser
# df = pd.read_csv('./data/grach_data.csv')
# index = df['DEXUSUK']
# index_shift = df['DEXUSUK'].shift(1)
# delay = np.zeros(df.shape[0])
# for i in range(1, df.shape[0]):
#     delay[i] = (parser.parse(df.values[i][1]) -
#                 parser.parse(df.shift(1).values[i][1])).days
# observations = np.zeros(500)
# for i in range(3, 503):
#     if delay[i] <= 1:
#         observations[i - 3] = (index[i] - index_shift[i]
#                                ) / index_shift[i] * 100
#     else:
#         observations[
#             i - 3] = (np.exp(np.log(index[i] / index_shift[i]) / delay[i]) - 1) * 100
# plt.plot(observations)
# plt.show()

########################################
# Estimation section with three modes ##
########################################
# !choose one mode and comment the other 2 modes

# # 1. evaluate loglikelihood for one specific value of one parameter (ex.
# # beta2=0.75)
# beta2 = 0.75
# initial_particles = initial_particle(N=N)
# likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                               likelihood_func=likelihood_function, transition=transition_sample, N=N)
# loglikelihood = sum(np.log(likelihoods))
# print('loglikelihood for beta2 {} : {}'.format(beta2, loglikelihood))


# # 2. evaluate loglikelihood for a serie of values of one parameter (ex. beta2 =
# # [0.5:0.95:0.05])
# beta2s = [i * 0.05 for i in range(10, 20)]
# initial_particles = initial_particle(N=N)
# loglikelihoods = np.zeros(len(beta2s))
# for i in range(len(beta2s)):
#     beta2 = beta2s[i]
#     likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                                   likelihood_func=likelihood_function, transition=transition_sample, N=N)
#     loglikelihood = sum(np.log(likelihoods))
#     loglikelihoods[i] = loglikelihood
#     print('loglikelihood for beta2 {} : {}'.format(beta2, loglikelihood))
# print(loglikelihoods)
# plt.plot(beta2s, loglikelihoods)
# plt.show()

# # 3.evaluate loglikelihood for a serie of values of one parameter with
# # multiple runs (each run with different random seeds)
# num_run = 50
# beta2s = [i * 0.05 for i in range(10, 20)]
# cumulated_loglikelihoods = np.zeros((num_run, len(beta2s)))

# estimations = np.zeros(num_run)
# for seed in range(num_run):
#     print('iteration {}'.format(seed))
#     initial_particles = initial_particle(N=N)
#     loglikelihoods = np.zeros(len(beta2s))
#     for k in range(len(beta2s)):
#         beta2 = beta2s[k]
#         # change seed for each run
#         likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                                       likelihood_func=likelihood_function, transition=transition_sample, N=N, seed=seed)
#         loglikelihood = sum(np.log(likelihoods))
#         loglikelihoods[k] = loglikelihood
#         print(
#             'log-likelihood calculation finished for beta2 = {} : {}'.format(beta2, loglikelihood))
#     # print(loglikelihoods)
#     # plt.plot(beta2s, loglikelihoods)
#     # plt.show()
#     estimations[seed] = beta2s[np.argmax(loglikelihoods)]
#     cumulated_loglikelihoods[seed] = loglikelihoods
# print(estimations)
# means = np.mean(cumulated_loglikelihoods, axis=0)
# variances = np.var(cumulated_loglikelihoods, axis=0)
# print(means)
# print(variances)
