import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# functions
def negative_likelihood(y, X, theta,):
    return np.sum(np.logaddexp(0, -(y * np.dot(X, theta))))

def objective_function(y, X, theta, gamma, D):
    return negative_likelihood(y, X, theta) + gamma * 0.5 * (np.linalg.norm(np.dot(D, theta)))**2

def gradient_method(y, X, theta, gamma, D):
    z = y * np.dot(X, theta)
    gradient = np.dot(-X.T, (y * expit(-z)))
    penalty = gamma * np.dot(D, theta)
    return gradient + penalty

def find_optimal_theta(y,
                       X,
                       theta_start,
                       gamma,
                       D,
                       step_size,
                       stop_criterion,
                       max_number_iterations,
                       acceleration = None,
                       accelerated = False):

    theta = [theta_start.copy(), theta_start.copy()]
    objective_function_values = []
    steps = []

    if accelerated:
        for k in range(max_number_iterations):
            steps.append(k)
            grad = gradient_method(y, X, theta[-1], gamma, D)
            theta.append(theta[-1] - step_size * grad + acceleration * (theta[-1] - theta[-2]))
            objective_function_values.append(objective_function(y, X, theta[-1], gamma, D))

            if np.linalg.norm(theta[-1] - theta[-2]) < stop_criterion:
                break

    else:
        for k in range(max_number_iterations):
            steps.append(k)
            grad = gradient_method(y, X, theta[-1], gamma, D)
            theta.append(theta[-1] - step_size * grad)
            objective_function_values.append(objective_function(y, X, theta[-1], gamma, D))

            if np.linalg.norm(theta[-1] - theta[-2]) < stop_criterion:
                break


    return theta[1:], steps, objective_function_values

def trade_off(y, X, theta, D):
    nll = negative_likelihood(y, X, theta)
    regularisation_term = 0.5 * np.linalg.norm(np.dot(D, theta))**2
    return nll, regularisation_term

#import data
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = 2 * y - 1

#add bias to X vector
X_tilde = np.zeros((X.shape[0], X.shape[1]+1))
for i, x in enumerate(X):
    x = np.append(x, 1)
    X_tilde[i] = x

#set variables
gamma = 3
D = np.identity(X_tilde.shape[1])
D[-1][-1] = 0
theta_start = np.zeros(X_tilde.shape[1])
L = gamma + 0.25 * np.linalg.norm(X_tilde)**2

step_size = 1/L
stop_criterion = 1e-3
max_number_iterations = 15000
acceleration = 0.7

#exercise 5
theta, steps, ob_func_values = find_optimal_theta(y=y,
                                  X=X_tilde,
                                  theta_start=theta_start,
                                  gamma=gamma,
                                  D=D,
                                  step_size=step_size,
                                  stop_criterion=stop_criterion,
                                  max_number_iterations=max_number_iterations,
                                  acceleration=None,
                                  accelerated=False)

theta_acc, steps_acc, ob_func_values_acc = find_optimal_theta(y=y,
                                  X=X_tilde,
                                  theta_start=theta_start,
                                  gamma=gamma,
                                  D=D,
                                  step_size=step_size,
                                  stop_criterion=stop_criterion,
                                  max_number_iterations=max_number_iterations,
                                  acceleration=acceleration,
                                  accelerated=True)

#exercise 6
gamma_values = np.logspace(-3, 3, 20)
theta_values = [theta_start]
trade_off_points = []

for gamma_value in gamma_values:
    L_constant = gamma_value + 0.25 * np.linalg.norm(X_tilde)**2
    theta_values.append(find_optimal_theta(y=y,
                                       X=X_tilde,
                                       theta_start=theta_values[-1],
                                       gamma=gamma_value,
                                       D=D,
                                       step_size=1/L_constant,
                                       stop_criterion=stop_criterion,
                                       max_number_iterations=max_number_iterations,
                                       acceleration=acceleration,
                                       accelerated=True)[0][-1])
    trade_off_points.append(trade_off(y=y,
                                      X=X_tilde,
                                      theta=theta_values[-1],
                                      D=D))

nll, reg = zip(*trade_off_points)

# plots
plt.figure(1)
plt.plot(steps, ob_func_values, label="not accelerated")
plt.plot(steps_acc, ob_func_values_acc, label="accelerated")
plt.yscale("log")
plt.xlabel("steps")
plt.ylabel("objective function values")
plt.legend()
plt.show()

plt.figure(2)
plt.scatter(nll[1:], reg[1:], label="trade-off curve")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("negative log likelihood")
plt.ylabel("regularisation term")
plt.legend()
plt.show()