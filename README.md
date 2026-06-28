# Logistic Regression via Gradient Methods

A Python implementation of regularized logistic regression using gradient descent and accelerated (momentum-based) gradient descent, including duality gap tracking and a Pareto trade-off curve between data fit and regularization.

---

## Overview

This project implements binary logistic regression with L2 regularization, solved via:

- **Gradient descent** (standard)
- **Accelerated gradient descent** (heavy-ball / momentum scheme)

It also computes the **duality gap** at each iteration to provide a convergence certificate, and supports plotting the **NLL vs. regularization trade-off curve** (Pareto frontier) across a range of regularization strengths.

---

## Files

| File | Description |
|------|-------------|
| `methods.py` | Core implementation: loss functions, gradient, optimizer |
| `example_gradient_methods.ipynb` | End-to-end example on the breast cancer dataset |
| `requirements.txt` | Python dependencies |

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies: `scikit-learn`, `numpy`, `scipy`, `matplotlib`

---

## Usage

### Basic Setup

Labels must be encoded as `{-1, +1}`. A bias column should be appended to the feature matrix.

```python
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from methods import run_gradient_method, nll_regularization_trade_off

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = 2 * y - 1  # encode as {-1, +1}

# Append bias column
X_tilde = np.hstack([X, np.ones((X.shape[0], 1))])
bias_index = -1
```

### Running Gradient Descent

```python
gamma = 3        # regularization strength
L = gamma + 0.25 * np.linalg.norm(X_tilde)**2  # Lipschitz constant
step_size = 1 / L

theta, obj_values, duality_gap, converged = run_gradient_method(
    y=y,
    X=X_tilde,
    gamma=gamma,
    step_size=step_size,
    bias_index=bias_index,
    stop_criterion=1e-4,
    max_number_iterations=4000,
)
```

### Accelerated Gradient Descent (Momentum)

```python
theta_acc, obj_values_acc, gap_acc, converged_acc = run_gradient_method(
    y=y,
    X=X_tilde,
    gamma=gamma,
    step_size=step_size,
    bias_index=bias_index,
    stop_criterion=1e-4,
    max_number_iterations=4000,
    acceleration=0.7,   # momentum coefficient in [0.5, 0.9]
)
```

### Trade-off Curve (Regularization Path)

```python
gamma_values = np.logspace(-3, 3, 20)
current_theta = np.zeros(X_tilde.shape[1])
trade_off_points = []

for gamma_value in gamma_values:
    current_theta = run_gradient_method(
        y=y, X=X_tilde,
        theta_start=current_theta,
        gamma=gamma_value,
        acceleration=0.7,
    )[0]
    trade_off_points.append(
        nll_regularization_trade_off(y=y, X=X_tilde, theta=current_theta)
    )
```

---

## API Reference

### `run_gradient_method`

```
run_gradient_method(y, X, theta_start=None, step_size=None, bias_index=-1,
                    stop_criterion=1e-4, max_number_iterations=1000,
                    acceleration=0.0, gamma=0.0)
```

Runs (accelerated) gradient descent on the regularized logistic regression objective.

| Parameter | Description |
|-----------|-------------|
| `y` | Label vector of shape `(n,)`, values in `{-1, +1}` |
| `X` | Feature matrix of shape `(n, p)` |
| `theta_start` | Initial parameter vector; defaults to zeros |
| `step_size` | Step size; defaults to `1 / (gamma + 0.25 * ‖X‖²)` |
| `bias_index` | Column index of the bias term in `X` |
| `stop_criterion` | Convergence threshold (duality gap or iterate change) |
| `max_number_iterations` | Maximum number of gradient steps |
| `acceleration` | Momentum coefficient — `0.0` for standard GD, `[0.5, 0.9]` for accelerated |
| `gamma` | L2 regularization strength |

**Returns:** `(optimal_theta, objective_values, duality_gap, converged)`

---

### `nll_regularization_trade_off`

```
nll_regularization_trade_off(y, X, theta, bias_index=-1)
```

Returns `(nll, regularization_term)` separately, useful for plotting the Pareto frontier.

---

## Convergence

- When `gamma > 0`: convergence is declared when the **duality gap** drops below `stop_criterion`.
- When `gamma = 0`: convergence is declared when `‖θ_k − θ_{k−1}‖ < stop_criterion`.

The duality gap provides a rigorous, computable upper bound on the suboptimality of the current iterate.

---

## Objective Function

The minimized objective is:

```
F(θ) = Σᵢ log(1 + exp(−yᵢ xᵢᵀ θ)) + (γ/2) ‖D θ‖²
```

where `D` is the identity matrix with the bias entry zeroed out (bias is not regularized).
