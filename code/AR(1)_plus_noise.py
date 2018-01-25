import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# sample generator for AR(1) plus noise Model
def generator_ar_1(sigma_epsilon_square=2, sigma_eta_square=0.02, phi=0.975, mu=0.5, T=5000, seed=2345):
    np.random.seed(seed=seed)
    x = 0
    ys = np.zeros(T)
    for i in range(T):
        ys[i] = x + np.random.randn(1) * np.sqrt(sigma_epsilon_square)
        x = (x - mu) * phi + mu + np.random.randn(1) * \
            np.sqrt(sigma_eta_square)
    print('sample generated with success !')
    return ys


# define the initial distribution of particles
def initial_particle(N):
    return np.random.randn(N)


# define the likelihood function
def likelihood_function(y, x):
    return norm.logpdf(y, loc=x, scale=np.sqrt(sigma_epsilon_square))


# define the transition function
def transition_sample(x):
    return (x - mu) * phi + mu + np.random.randn(1) * np.sqrt(sigma_eta_square)


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
def stratified_resample(weights, xs):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i + 1]) for i in range(n)]
    u0, j = np.random.uniform(size=1), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return [xs[ind] for ind in indices]


# continuous resample method described in the article
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


# Main algo for particle filtering
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
            initial_particles[j] = transition(initial_particles[j])
        # print('time step {} finished with likelihood {}'.format(i, likelihood))
    return likelihoods


######################
# parameters section #
######################

# real parameters for generating samples
sigma_epsilon_square_0 = 2
sigma_eta_square_0 = 0.02
phi_0 = 0.975
mu_0 = 0.5

# test parameters for estimating likelihood
sigma_epsilon_square = 2
sigma_eta_square = 0.02
phi = 0.975

# specify N for number of particles, T for length of obervations
T = 5000
N = 350

# Generate sample as observations with parameters specified in the above
# section
observations = generator_ar_1(sigma_epsilon_square=sigma_epsilon_square_0,
                              sigma_eta_square=sigma_eta_square_0, phi=phi_0, mu=mu_0, T=T)

########################################
# Estimation section with three modes ##
########################################
# !choose one mode and comment the other 2 modes

# # 1. evaluate loglikelihood for one specific value of one parameter (ex.
# # mu=0.5)
# mu = 0.5
# initial_particles = initial_particle(N=N)
# likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                               likelihood_func=likelihood_function, transition=transition_sample, N=N)
# loglikelihood = sum(np.log(likelihoods))
# print('loglikelihood for mu {} : {}'.format(mu, loglikelihood))


# # 2. evaluate loglikelihood for a serie of values of one parameter (ex. mu =
# # [0.33:0.66:0.03])
# mus = [i * 0.03 for i in range(11, 23)]
# initial_particles = initial_particle(N=N)
# loglikelihoods = np.zeros(len(mus))
# for i in range(len(mus)):
#     mu = mus[i]
#     likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                                   likelihood_func=likelihood_function, transition=transition_sample, N=N)
#     loglikelihood = sum(np.log(likelihoods))
#     loglikelihoods[i] = loglikelihood
#     print('loglikelihood for mu {} : {}'.format(mu, loglikelihood))
# print(loglikelihoods)
# plt.plot(mus, loglikelihoods)
# plt.show()

# # 3.evaluate loglikelihood for a serie of values of one parameter with
# # multiple runs (each run with different random seeds)
# num_run = 50
# mus = [i * 0.03 for i in range(11, 23)]
# cumulated_loglikelihoods = np.zeros((num_run, len(mus)))

# estimations = np.zeros(num_run)
# for seed in range(num_run):
#     print('iteration {}'.format(seed))
#     initial_particles = initial_particle(N=N)
#     loglikelihoods = np.zeros(len(mus))
#     for k in range(len(mus)):
#         mu = mus[k]
#         # change seed for each run
#         likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                                       likelihood_func=likelihood_function, transition=transition_sample, N=N, seed=seed)
#         loglikelihood = sum(np.log(likelihoods))
#         loglikelihoods[k] = loglikelihood
#         print(
#             'log-likelihood calculation finished for mu = {} : {}'.format(mu, loglikelihood))
#     # print(loglikelihoods)
#     # plt.plot(mus, loglikelihoods)
#     # plt.show()
#     estimations[seed] = mus[np.argmax(loglikelihoods)]
#     cumulated_loglikelihoods[seed] = loglikelihoods
# print(estimations)
# means = np.mean(cumulated_loglikelihoods, axis=0)
# variances = np.var(cumulated_loglikelihoods, axis=0)
# print(means)
# print(variances)
