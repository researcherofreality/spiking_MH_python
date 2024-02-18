import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

def ToyModelExponentialFilter(n, p, S, mu, G, v0, tMax, eta, gain=1):

    Omega = G.T @ np.linalg.inv(S) @ G
    v = np.full((tMax, n), np.nan)
    sp = np.zeros((tMax, n))
    r = np.zeros((tMax, n))
    v[0, :] = v0
    constMean = mu.shape == (p,) or mu.shape == (p, 1)

    if constMean:
        muProj = eta * (mu / S) @ G
    else:
        muDiff = mu[1:, :] - (1 - eta) * mu[:-1, :]
        muProj = np.array([np.linalg.solve(S, muDiff[i, :]) @ G for i in range(muDiff.shape[0])])

    # Iterate
    for t in range(1, tMax):
        spikeProp = np.random.randint(0, n)

        a = min(1, np.exp(gain * (v[t - 1, spikeProp] - Omega[spikeProp, spikeProp] / 2)))
        if np.random.rand() <= a:
            sp[t, spikeProp] = 1

        if constMean:
            v[t, :] = (1 - eta) * (v[t - 1, :] - sp[t, :] @ Omega) + muProj
        else:
            v[t, :] = (1 - eta) * (v[t - 1, :] - sp[t, :] @ Omega) + muProj[t - 1, :]

        r[t, :] = (1 - eta) * r[t - 1, :] + sp[t, :]
    theta = r @ G.T

    return sp, theta, r, v

####### RUN THE SIMULATION #######

p = 10
n = 100
dt = 1e-5
tMax = 2
tMaxSteps = round(tMax / dt)
tau = 0.020
eta = dt / tau

tOff = round(0.5 / dt)
tOn = tMaxSteps - tOff
maskVec = np.concatenate([np.zeros(tOff, dtype=bool), np.ones(tOn, dtype=bool)])
mu = np.outer(maskVec, np.ones(p))

sigma = 1
rho = 0.75
Sigma = sigma * (np.eye(p) + rho * (np.ones((p, p)) - np.eye(p)))

A = np.random.randn(p, n // 2)
gammaNaive = np.hstack([A, -A])
gammaGeom = sqrtm(Sigma) @ np.hstack([A, -A])

vInitNaive = mu[0, :] @ np.linalg.inv(Sigma) @ gammaNaive
vInitGeom = mu[0, :] @ np.linalg.inv(Sigma) @ gammaGeom

import time
start_time = time.time()
spNaive, thetaNaive, rateNaive, vNaive = ToyModelExponentialFilter(n, p, Sigma, mu, gammaNaive, vInitNaive, tMaxSteps, eta)
print("Naive Sampler Time: ", time.time() - start_time)

start_time = time.time()
spGeom, thetaGeom, rateGeom, vGeom = ToyModelExponentialFilter(n, p, Sigma, mu, gammaGeom, vInitGeom, tMaxSteps, eta)
print("Geom Sampler Time: ", time.time() - start_time)

tSec = np.arange(tMaxSteps) * dt

########## PLOTTING ##########

def downsample(data, factor):
    return data[::factor]

def fast_moving_average(data, window_size):
    return uniform_filter1d(data, size=int(window_size), axis=0)

tSec = np.arange(tMaxSteps) * dt

corder = np.array([[0.850980392, 0.37254902, 0.00784313725],
                   [0.458823529, 0.439215686, 0.701960784]])

interp_naive = interp1d([1, 0], np.vstack([[1, 1, 1], corder[0, :]]), axis=0)(np.linspace(1, 0, p))
interp_geom = interp1d([1, 0], np.vstack([[1, 1, 1], corder[1, :]]), axis=0)(np.linspace(1, 0, p))

tau_average = 0.01
downsample_factor = 100

plt.figure(figsize=(10, 14))
for i in range(p):
    plt.plot(downsample(tSec, downsample_factor), 
             downsample(fast_moving_average(thetaNaive[:, i], tau_average/dt), downsample_factor),
             color=interp_naive[i], linewidth=1)
plt.plot(downsample(tSec, downsample_factor), downsample(mu, downsample_factor), '--k', linewidth=1)
plt.xlabel('time (s)')
plt.ylabel('moving average of parameter estimate (arb units)')
plt.title(f'Naive geometry, rho = {rho:.2f}, p = {p}, n = {n}')
plt.axis('square')
plt.show()

plt.figure(figsize=(10, 14))
for i in range(p):
    plt.plot(downsample(tSec, downsample_factor), 
             downsample(fast_moving_average(thetaGeom[:, i], tau_average/dt), downsample_factor),
             color=interp_geom[i], linewidth=1)
plt.plot(downsample(tSec, downsample_factor), downsample(mu, downsample_factor), '--k', linewidth=1)
plt.xlabel('time (s)')
plt.ylabel('moving average of parameter estimate (arb units)')
plt.title(f'Natural geometry, rho = {rho:.2f}, p = {p}, n = {n}')
plt.axis('square')
plt.show()
