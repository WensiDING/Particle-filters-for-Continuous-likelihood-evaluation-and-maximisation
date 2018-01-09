import numpy as np
from scipy.stats import norm
from scipy.stats import invgamma
import matplotlib.pyplot as plt
# return the mean value of non-normalized weights for loglikelihood estimation
# and the normalized weights for resampling step


def importance_ratio(likelihood_func, y, sigma_hat, new_particles):
    log_weights = [likelihood_func(y, x) for x in sigma_hat]
    # print(log_weights)
    maximum = np.max(log_weights)
    weights_ratio = np.exp(log_weights - maximum)
    likelihood = np.mean(weights_ratio) * np.exp(maximum)

    length = len(new_particles)
    smoothed_weights = np.zeros(length)
    phi = kernel_distance(new_particles)
    # print(log_weights)
    # print(new_particles)
    # print(phi)
    for i in range(length):
        dist_sum = np.sum(phi[i])
        inter_log_weights = np.zeros(length)
        for j in range(length):
            if phi[i][j] != 0:
                inter_log_weights[j] = log_weights[j] + np.log(phi[i][j])
        inter_maximum = np.max(inter_log_weights)
        inter_ratio = np.exp(inter_log_weights - inter_maximum)
        smoothed_weights[i] = np.log(np.sum(
            inter_ratio)) + inter_maximum - np.log(dist_sum)
    # print(smoothed_weights)
    maximum = np.max(smoothed_weights)
    weights_ratio = np.exp(smoothed_weights - maximum)
    # _ = np.mean(weights_ratio) * np.exp(maximum)
    normalized_weights = weights_ratio / sum(weights_ratio)
    return likelihood, normalized_weights


def kernel_distance(new_particles):
    num = len(new_particles)
    phi = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):
            if np.abs(new_particles[i] - new_particles[j]) < 3 * h:
                dist = norm.pdf((new_particles[i] - new_particles[j]) / h)
                phi[i, j] = dist
                phi[j, i] = dist
    return phi

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
    # algo described in the paper A.3.
    # pi = np.append(0, weights)
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


def particle_filter(observations, initial_particles, likelihood_func, transition, n, seed=1234):
    np.random.seed(seed=seed)
    T = len(observations)
    likelihoods = np.zeros(T)
    for i in range(T):
        sigma_hat = np.zeros(N)
        new_particles = np.zeros(N)
        for j in range(N):
            inter_array, new_particles[j] = transition(initial_particles[j])
            sigma_hat[j] = delta * np.sum(inter_array)
        order = np.argsort(new_particles)
        new_particles = new_particles[order]
        sigma_hat = sigma_hat[order]

        likelihood, normalized_weights = importance_ratio(
            likelihood_func, observations[i], sigma_hat, new_particles)
        likelihoods[i] = likelihood
        initial_particles = continuous_stratified_resample(
            normalized_weights, new_particles)
        # print('time step {} finished with likelihood {}'.format(i, likelihood))
    return likelihoods


# AR(1) model

# AR(1) sample generator


def generator(k=0.02, theta=0.5, xi=0.0178, delta_s=0.01, n=4000, seed=2345):
    np.random.seed(seed=seed)
    sigma_square = beta_0 / (alpha_0 - 1)
    dys = np.zeros(n)
    for i in range(n):
        sigma_square = sigma_square + k_0 * (theta_0 - sigma_square) * delta_s + np.sqrt(
            xi_0) * sigma_square * np.random.randn(1) * np.sqrt(delta_s)
        dys[i] = np.sqrt(sigma_square) * np.random.randn(1) * np.sqrt(delta_s)
    print('sample generated with success !')
    return dys

# generate initial particles with stationary distribution if it exists


def initial_particle(N):
    return invgamma.rvs(a=alpha, scale=beta, size=N)

# likelihood function


def likelihood_function(y, sigma_hat):
    return norm.logpdf(y, loc=0, scale=np.sqrt(sigma_hat))


# transition function
# def transition_sample(sigma_square):
#     sigma_square_array = np.zeros(Ms)
#     sigma_square_array[0] = sigma_square
#     for i in range(1, Ms + 1):
#         a = k * (theta - sigma_square_array[i - 1])
#         b = np.sqrt(xi) * sigma_square_array[i - 1]
#         if i < Ms:
#             sigma_square_array[i] = sigma_square_array[i - 1] + \
#                 a * delta + b * np.sqrt(delta) * np.random.randn(1)
#         else:
#             sigma_square_new = sigma_square_array[i - 1] + \
#                 a * delta + b * np.sqrt(delta) * np.random.randn(1)
#     return sigma_square_array, sigma_square_new
def transition_sample(sigma_square):
    sigma_square_array = np.zeros(Ms + 1)
    sigma_square_array[0] = sigma_square
    for i in range(1, Ms + 1):
        a = k * (theta - sigma_square_array[i - 1])
        b = np.sqrt(xi) * sigma_square_array[i - 1]
        sigma_square_array[i] = sigma_square_array[i - 1] + \
            a * delta + b * np.sqrt(delta) * np.random.randn(1)
    return sigma_square_array[:-1], sigma_square_array[-1]

# parameters
k_0 = 0.02
theta_0 = 0.5
xi_0 = 0.0178
alpha_0 = 1 + 2 * k_0 / xi_0
beta_0 = 2 * k_0 * theta_0 / xi_0
delta_s = 1


k = 0.02
# theta = 0.5
xi = 0.0178
# alpha = 1 + 2 * k / xi
# beta = 2 * k * theta / xi

# n = 4000
# N = 1000
n = 4000
N = 1000
Ms = 20
delta = delta_s / Ms
# kernel smothing parameters
c = 1
h = c / N
# T = 150
# N = 300
observations = generator(k=k_0, theta=theta_0, xi=xi_0, n=n, delta_s=delta_s)
# plt.plot(observations)
# plt.show()

thetas = [0.1 * i for i in range(2, 8)]
for i in range(len(thetas)):
    theta = thetas[i]
    alpha = 1 + 2 * k / xi
    beta = 2 * k * theta / xi
    initial_particles = initial_particle(N=N)
    likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
                                  likelihood_func=likelihood_function, transition=transition_sample, n=n)
    loglikelihood = sum(np.log(likelihoods))
    print(loglikelihood)
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


# a list of testing mu values
# cumulated_loglikelihoods = np.zeros((50, len(beta2s)))
# estimations = np.zeros(50)
# for seed in range(50):
#     print('iteration {}'.format(seed))
#     loglikelihoods = np.zeros(len(beta2s))
#     for i in range(len(beta2s)):
#         beta2 = beta2s[i]
#         likelihoods = particle_filter(observations=observations, initial_particles=initial_particles,
#                                       likelihood_func=likelihood_function, transition=transition_sample, N=N, seed=seed)
#         loglikelihood = sum(np.log(likelihoods))
#         loglikelihoods[i] = loglikelihood
#     print(loglikelihoods)
#     estimations[seed] = mus[np.argmax(loglikelihoods)]
#     cumulated_loglikelihoods[seed] = loglikelihoods
# means = np.mean(cumulated_loglikelihoods, axis=0)
# variances = np.var(cumulated_loglikelihoods, axis=0)
# print(means)
# print(variances)
# plt.plot(rhos, means, 'r--', rhos, means + np.sqrt(variances),
#          'b--', rhos, means - np.sqrt(variances), 'b--')
# plt.show()
