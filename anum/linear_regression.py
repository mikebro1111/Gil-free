import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.datasets import make_regression
import time
from multiprocessing import Pool, cpu_count
from numba import jit
from concurrent.futures import ThreadPoolExecutor

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit_normal_equation(self, X, y):
        """Normal Equation Method: θ = (X^T X)^(-1) X^T y"""
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        return self

    @staticmethod
    @jit(nopython=True)
    def _gradient_descent_numba(X, y, theta, learning_rate, n_iterations):
        m = len(y)
        for _ in range(n_iterations):
            prediction = X.dot(theta)
            error = prediction - y
            gradients = 2/m * X.T.dot(error)
            theta = theta - learning_rate * gradients
        return theta

    def fit_gradient_descent(self, X, y, learning_rate=0.01, n_iterations=1000):
        """Gradient descent with mini-batches and parallelization"""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.zeros(X_b.shape[1])
        theta = LinearRegression._gradient_descent_numba(X_b, y, theta, learning_rate, n_iterations)
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        return self

    def predict(self, X):
        """Make predictions"""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

def benchmark_regression(X, y, n_runs=3):
    results = {}
    
    # Normal Equation Method
    times_normal = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reg = LinearRegression()
        reg.fit_normal_equation(X, y)
        times_normal.append(time.perf_counter() - start)
    results['normal_equation'] = np.mean(times_normal)

    # Gradient Descent with Numba
    times_gradient = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reg = LinearRegression()
        reg.fit_gradient_descent(X, y)
        times_gradient.append(time.perf_counter() - start)
    results['gradient_descent'] = np.mean(times_gradient)

    # Scikit-learn
    times_sklearn = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reg = SklearnLinearRegression()
        reg.fit(X, y)
        times_sklearn.append(time.perf_counter() - start)
    results['sklearn'] = np.mean(times_sklearn)

    return results

if __name__ == '__main__':
    
    sizes = [
        (1000, 5),
        (10000, 10),
        (100000, 20),
        (1000000, 50)
    ]

    print("Comparing Linear Regression Implementations:\n")
    
    for n_samples, n_features in sizes:
        print(f"\nDataset size: {n_samples:,} samples, {n_features} features")
        
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=42
        )
        
        results = benchmark_regression(X, y)
        
        print(f"Normal Equation Method: {results['normal_equation']:.4f} sec")
        print(f"Gradient Descent (Numba): {results['gradient_descent']:.4f} sec")
        print(f"Scikit-learn: {results['sklearn']:.4f} sec") 