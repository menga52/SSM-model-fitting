import numpy as np

def kalman_filter(y, A, C, Q, R, mu, P):
    T = y.shape[1]
    n = A.shape[0]

    # Initialize storage
    x_f = np.zeros((n, T))
    P_f = np.zeros((n, n, T))
    K = np.zeros((n, C.shape[0], T))

    # Initial state
    x_t = mu
    P_t = P

    for t in range(T):
        # Prediction step
        x_pred = A @ x_t
        P_pred = A @ P_t @ A.T + Q

        # Update step
        S = C @ P_pred @ C.T + R
        K_t = np.linalg.solve(S.T, (P_pred @ C.T).T).T
        x_t = x_pred + K_t @ (y[:, t] - C @ x_pred)
        P_t = P_pred - K_t @ C @ P_pred

        # Store results
        x_f[:, t] = x_t
        P_f[:, :, t] = P_t
        K[:, :, t] = K_t

    return x_f, P_f, K

def kalman_smoother(x_f, P_f, A, Q):
    T = x_f.shape[1]
    n = A.shape[0]

    x_s = np.zeros_like(x_f)
    P_s = np.zeros_like(P_f)
    P_lag = np.zeros((n, n, T - 1))

    x_s[:, -1] = x_f[:, -1]
    P_s[:, :, -1] = P_f[:, :, -1]

    for t in reversed(range(T - 1)):
        P_pred = A @ P_f[:, :, t] @ A.T + Q
        J = np.linalg.solve(P_pred.T, (P_f[:, :, t] @ A.T).T).T
        x_s[:, t] = x_f[:, t] + J @ (x_s[:, t + 1] - A @ x_f[:, t])
        P_s[:, :, t] = P_f[:, :, t] + J @ (P_s[:, :, t + 1] - P_pred) @ J.T
        P_lag[:, :, t] = J @ P_s[:, :, t + 1]

    return x_s, P_s, P_lag

def baum_welch(y, state_dim, max_iter=1000, tol=1e-6):
    T = y.shape[1]
    obs_dim = y.shape[0]

    # Initialize parameters
    A = np.eye(state_dim) + 0.01 * np.random.randn(state_dim, state_dim)
    C = np.random.randn(obs_dim, state_dim)
    Q = np.eye(state_dim)
    R = np.eye(obs_dim)
    mu = np.random.randn(state_dim)
    P = np.eye(state_dim)

    log_likelihoods = []

    for iteration in range(max_iter):
        # E-Step: Kalman filter and smoother
        x_f, P_f, _ = kalman_filter(y, A, C, Q, R, mu, P)
        x_s, P_s, P_lag = kalman_smoother(x_f, P_f, A, Q)

        # Compute sufficient statistics
        S_xx = np.zeros((state_dim, state_dim))
        S_xx_lag = np.zeros((state_dim, state_dim))
        S_x = np.zeros((state_dim,))
        S_yx = np.zeros((obs_dim, state_dim))
        S_yy = np.zeros((obs_dim, obs_dim))

        for t in range(T):
            S_xx += P_s[:, :, t] + np.outer(x_s[:, t], x_s[:, t])
            S_x += x_s[:, t]
            S_yy += np.outer(y[:, t], y[:, t])
            S_yx += np.outer(y[:, t], x_s[:, t])
            if t > 0:
                S_xx_lag += P_lag[:, :, t - 1] + np.outer(x_s[:, t], x_s[:, t - 1])

        # M-Step: Update parameters
        A = S_xx_lag @ np.linalg.inv(S_xx - np.outer(x_s[:, 0], x_s[:, 0]))
        C = S_yx @ np.linalg.inv(S_xx)
        Q = (S_xx - A @ S_xx_lag.T) / (T - 1)
        R = (S_yy - C @ S_yx.T) / T
        mu = x_s[:, 0]
        P = P_s[:, :, 0]

        # Compute log-likelihood
        ll = -0.5 * np.sum(np.log(np.linalg.det(R))) * T
        log_likelihoods.append(ll)

        # Check for convergence
        if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return A, C, Q, R, mu, P, log_likelihoods

# Example usage:
# Define a known LGSSM
state_dim = 2
obs_dim = 2
T = 50

# Parameters of the LGSSM
A = np.array([[0.9, 0.1], [-0.1, 0.9]])
C = np.array([[1.0, 0.0], [0.0, 1.0]])
Q = 0.1 * np.eye(state_dim)
R = 0.1 * np.eye(obs_dim)
mu = np.array([0.0, 0.0])
P = np.eye(state_dim)

# Generate a sample sequence of observations
x = np.zeros((state_dim, T))
y = np.zeros((obs_dim, T))

x[:, 0] = np.random.multivariate_normal(mu, P)
y[:, 0] = np.random.multivariate_normal(C @ x[:, 0], R)

for t in range(1, T):
    x[:, t] = np.random.multivariate_normal(A @ x[:, t - 1], Q)
    y[:, t] = np.random.multivariate_normal(C @ x[:, t], R)


a,c,q,r,m,p,ll = baum_welch(y, state_dim=2)

print(A,"A")
print(a,'a')


