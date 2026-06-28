import numpy as np
from numpy.typing import NDArray
from scipy.special import expit



def negative_log_likelihood(y: NDArray[np.float64],
                            X: NDArray[np.float64],
                            theta: NDArray[np.float64]) -> float:
    """Compute the negative log-likelihood for logistic regression.

    Uses np.logaddexp for numerical stability, avoiding overflow in
    log(1 + exp(x)).

    Args:
        y:     Label vector of shape (n,) with values in {-1, +1}.
        X:     Feature matrix of shape (n, p).
        theta: Parameter vector of shape (p,).

    Returns:
        Scalar negative log-likelihood value.
    """
    return np.sum(np.logaddexp(0, -(y * np.dot(X, theta))))

def l2_regularization(theta: NDArray[np.float64],
                      D: NDArray[np.float64]) -> float:
    """Compute L2 regularization term without gamma.
    Gamma is introduced in gradient_method function.

    L2 Regularization = (1 / 2) * ||D @ theta||^2

    Args:
        theta: Parameter vector of shape (p,).
        D:     Regularization matrix of shape (p, p). Typically the identity
               with the bias entry zeroed out.

    Returns:
        Scalar objective function value.
    """
    return 0.5 * (np.linalg.norm(np.dot(D, theta)))**2

def binary_entropy(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise binary entropy H(p) = -p*log(p) - (1-p)*log(1-p).

    Numerically stable: clips p away from 0 and 1.
    """
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def dual_objective(y: NDArray[np.float64],
                   X: NDArray[np.float64],
                   theta: NDArray[np.float64],
                   gamma: float,
                   bias_index: int) -> float:
    """Compute the dual objective at the current iterate.

    D(theta) = sum_i H(p_i) - (1/2*gamma) * ||X^T (y ⊙ p)||^2

    where p_i = sigma(-y_i * x_i^T theta) are the dual variables
    (misclassification probabilities).

    Args:
        y:     Label vector of shape (n,) with values in {-1, +1}.
        X:     Feature matrix of shape (n, p).
        theta: Current parameter vector of shape (p,).
        gamma: Regularization strength (must be > 0).
        bias_index: Index of bias column in design matrix X.

    Returns:
        Scalar dual objective value.
    """
    z = y * np.dot(X, theta)
    p = expit(-z)  # dual variables p_i = σ(-y_i x_i^T θ)

    entropy_sum = np.sum(binary_entropy(p))

    # Dual feasibility: recover the implicit dual parameter
    # At optimum: X^T(y ⊙ p) = gamma * D^T D theta
    # Away from optimum, we project onto the dual-feasible set
    grad_data = np.dot(X.T, y * p)  # X^T (y ⊙ p), shape (p,)
    grad_data[bias_index] = 0
    dual_penalty = (1 / (2 * gamma)) * np.linalg.norm(grad_data) ** 2

    return entropy_sum - dual_penalty

def compute_gradient(y: NDArray[np.float64],
                     X: NDArray[np.float64],
                     theta: NDArray[np.float64],
                     gamma: float,
                     D: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the gradient of the regularized objective w.r.t. theta.

    Gradient = -X^T (y * sigma(-y * X theta)) + gamma * D^T D theta,
    where sigma is the sigmoid function.

    Args:
        y:     Label vector of shape (n,) with values in {-1, +1}.
        X:     Feature matrix of shape (n, p).
        theta: Parameter vector of shape (p,).
        gamma: Regularization strength (non-negative scalar).
        D:     Regularization matrix of shape (p, p).

    Returns:
        Gradient vector of shape (p,).
    """
    z = y * np.dot(X, theta)
    grad = np.dot(-X.T, (y * expit(-z)))
    if gamma == 0.0:
        return grad
    else:
        penalty = gamma * np.dot(D, theta)
        return grad + penalty

def run_gradient_method(y: NDArray[np.float64],
                        X: NDArray[np.float64],
                        theta_start = None,
                        step_size = None,
                        bias_index: int = -1,
                        stop_criterion: float = 1e-04,
                        max_number_iterations: int = 1000,
                        acceleration: float = 0.0,
                        gamma: float = 0.0) -> tuple[NDArray[np.float64], list[float],list[float], bool]:
    """Run (accelerated) gradient descent to minimize the regularized objective.

    Implements standard gradient descent when acceleration=0.0, or a
    heavy-ball / momentum scheme for acceleration > 0.

    Args:
        y:                      Label vector of shape (n,) with values in {-1, +1}.
        X:                      Feature matrix of shape (n, p).
        theta_start:            Initial parameter vector of shape (p,).
        step_size:              Gradient descent step size. A stable choice is
                                1 / L where L is the Lipschitz constant of the
                                gradient.
        bias_index:             Index of bias column in design matrix X.
        stop_criterion:         Stop when ||theta_k - theta_{k-1}|| < this value.
        max_number_iterations:  Maximum number of gradient steps.
        acceleration:           Momentum coefficient (default 0.0 = no momentum).
                                Typical values are in [0.5, 0.9].
        gamma:                  Regularisation strength (non-negative scalar).

    Returns:
        A tuple (optimal_theta, objective_values, duality_gap, converged) where:
          - optimal_theta is theta of last iteration,
          - objective_values is a list of objective function values in each iteration,
          - duality_gap is a list of the duality gap in each iteration
            only in regularized regression available,
          - converged is True if the stop criterion was met, False if
            max_number_iterations was reached.
    """

    if theta_start is None:
        theta_start = np.zeros(X.shape[1])
    if step_size is None:
        step_size = 1 / (gamma + 0.25 * np.linalg.norm(X)**2)
    D = np.identity(X.shape[1])
    D[bias_index][bias_index] = 0
    current_theta = theta_start.copy()
    prior_theta = theta_start.copy()
    objective_function_values = []
    duality_gap = []
    converged = False

    for k in range(max_number_iterations):
        grad = compute_gradient(y, X, current_theta, gamma, D)
        if k>=1:
            dummy = current_theta
            current_theta = current_theta - step_size * grad + acceleration * (current_theta - prior_theta)
            prior_theta = dummy
        else:
            current_theta = current_theta - step_size * grad
        primal = negative_log_likelihood(y, X, current_theta) + gamma * l2_regularization(current_theta, D)
        objective_function_values.append(primal)
        if gamma != 0.0:
            dual = dual_objective(y, X, current_theta, gamma, bias_index)
            duality_gap.append(primal-dual)
            if primal-dual < stop_criterion:
                converged = True
                break

        elif np.linalg.norm(current_theta - prior_theta) < stop_criterion:
            converged = True
            break

    return current_theta, objective_function_values, duality_gap, converged

def nll_regularization_trade_off(y: NDArray[np.float64],
                                 X: NDArray[np.float64],
                                 theta: NDArray[np.float64],
                                 bias_index: int = -1) -> tuple[float, float]:
    """Compute the NLL and regularization term separately for a trade-off curve.

    Useful for plotting the Pareto frontier between data fit and regularization
    across a range of gamma values.

    Args:
        y:     Label vector of shape (n,) with values in {-1, +1}.
        X:     Feature matrix of shape (n, p).
        theta: Parameter vector of shape (p,).
        bias_index: Index of bias column in design matrix X.

    Returns:
        A tuple (nll, regularization_term) of scalar floats.
    """
    D = np.identity(X.shape[1])
    D[bias_index][bias_index] = 0
    nll = negative_log_likelihood(y, X, theta)
    regularisation_term = l2_regularization(theta, D)
    return nll, regularisation_term