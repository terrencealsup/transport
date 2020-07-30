import numpy as np

# Define the quadratic map.
def T(x, w):
    x1 = x[0]
    x2 = x[1]
    T1 = w[0] + w[1]*x1 + w[2]*x1**2
    T2 = w[3] + w[4]*x1 + w[5]*x2 + w[6]*x1*x2 + w[7]*x1**2 + w[8]*x2**2
    return np.array([T1, T2])

# Inverse of the map.
def T_inv(y, w):
    y1 = y[0]
    y2 = y[1]
    x1 = (-w[1] + np.sqrt(w[1]**2 - 4*w[2]*(w[0] - y1)))/(2*w[2])
    a = w[8]
    b = w[6]*x1 + w[5]
    c = w[7]*x1**2 + w[4]*x1 + w[3] - y2
    x2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    return np.array([x1, x2])

# Returns the Jacobian matrix.
def T_grad(x, w):
    x1 = x[0]
    x2 = x[1]
    g11 = w[1] + 2*w[2]*x1
    g12 = 0
    g21 = w[4] + w[6]*x2 + 2*w[7]*x1
    g22 = w[5] + w[6]*x1 + 2*w[8]*x2
    return np.array([[g11, g12], [g21, g22]])

# Returns Jacobian of inverse.
def T_inv_grad(y, w):
    return np.linalg.inv(T_grad(T_inv(y, w), w))



# Helper method to vectorize computation of determinants.
def grad_det(X, w):
    N = X.shape[1]
    gd = np.zeros(N)
    for i in range(N):
        gd[i] = np.abs(np.linalg.det(T_grad(X[:,i], w)))
    return gd

# Estimated chi-squared divergence.
# p is target
# q is reference (i.e. before transport)
def D(w, X, p, q):
    N = X.shape[1] # Number of columns/samples
    likeli = p(T(X, w))*grad_det(X, w)/(q(X))
    return np.mean(likeli**2)/(np.mean(likeli)**2)

# Gradient of chi-squared divergence w.r.t. weights
def grad_D(w, X, p, q):
    h = 1e-4
    d = len(w)
    g = np.zeros(d) # The gradient of finite differences.

    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0 # Unit basis vector.
        fp1 = D(w + h*e, X, p, q)
        f = D(w, X, p, q)
        g[i] = (fp1 - f)/h

    return [g, f]

# Gradient descent to minimize chi-squared divergence
def minimize_chisq(w0, X, p, q, lr, iters):
    w = np.copy(w0)

    obj_vals = np.zeros(iters)
    gnorm = np.zeros(iters)

    for i in range(iters):
        [g, f] = grad_D(w, X, p, q)

        w -= lr * g/(i+1) # Update with decaying learning rate
        gnorm[i] = np.linalg.norm(g)
        obj_vals[i] = f

        # Total of 10 prints
        if np.mod(i,iters//10) == 0:
            print('Iteration {:d}, grad norm = {:0.3f}, obj val = {:0.3f}'.format(i, gnorm[i], f))

    return [w, obj_vals, gnorm]
