import numpy as np
import pytest
from src.linear_regression import (
    NormalEquationRegression,
    GradientDescentRegression,
    ParallelGradientDescentRegression,
    SklearnWrapper
)
from src.utils import DataGenerator, compute_metrics

@pytest.fixture
def regression_data():
    """Generate regression dataset"""
    return DataGenerator.generate_regression_data(
        n_samples=1000,
        n_features=5,
        noise=0.1,
        random_state=42
    )

@pytest.fixture
def implementations():
    """Get all regression implementations"""
    return [
        NormalEquationRegression(),
        GradientDescentRegression(n_iterations=1000),
        ParallelGradientDescentRegression(n_iterations=1000),
        SklearnWrapper()
    ]

def test_implementations_similar_results(implementations, regression_data):
    """Test that all implementations give similar results"""
    X_train, X_test, y_train, y_test = regression_data
    scores = []
    
    for impl in implementations:
        impl.fit(X_train, y_train)
        score = impl.score(X_test, y_test)
        scores.append(score)
    
    # All scores should be within 0.1 of each other
    assert max(scores) - min(scores) < 0.1, "Implementations give too different results"

@pytest.mark.parametrize("n_samples,n_features", [
    (100, 3),
    (1000, 5),
    (10000, 10)
])
def test_different_sizes(implementations, n_samples, n_features):
    """Test implementations with different dataset sizes"""
    X_train, X_test, y_train, y_test = DataGenerator.generate_regression_data(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42
    )
    
    for impl in implementations:
        impl.fit(X_train, y_train)
        score = impl.score(X_test, y_test)
        assert 0 <= score <= 1, f"Invalid score for {impl.__class__.__name__}"

def test_input_validation(implementations):
    """Test input validation"""
    X = np.random.random((100, 5))
    y = np.random.random(200)  # Wrong size
    
    for impl in implementations:
        with pytest.raises(ValueError):
            impl.fit(X, y)

def test_prediction_shape(implementations, regression_data):
    """Test prediction shapes"""
    X_train, X_test, y_train, y_test = regression_data
    
    for impl in implementations:
        impl.fit(X_train, y_train)
        y_pred = impl.predict(X_test)
        assert y_pred.shape == y_test.shape, f"Wrong prediction shape for {impl.__class__.__name__}"

def test_warm_start(implementations, regression_data):
    """Test multiple fits with warm start"""
    X_train, X_test, y_train, y_test = regression_data
    
    for impl in implementations:
        # First fit
        impl.fit(X_train, y_train)
        score1 = impl.score(X_test, y_test)
        
        # Second fit
        impl.fit(X_train, y_train)
        score2 = impl.score(X_test, y_test)
        
        assert abs(score1 - score2) < 1e-10, f"Inconsistent results for {impl.__class__.__name__}"

@pytest.mark.parametrize("impl_class", [
    GradientDescentRegression,
    ParallelGradientDescentRegression
])
def test_gradient_descent_convergence(impl_class, regression_data):
    """Test gradient descent convergence"""
    X_train, X_test, y_train, y_test = regression_data
    
    # Test with different learning rates
    learning_rates = [0.1, 0.01, 0.001]
    scores = []
    
    for lr in learning_rates:
        model = impl_class(learning_rate=lr, n_iterations=1000)
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    
    # At least one learning rate should give good results
    assert max(scores) > 0.5, f"Failed to converge for {impl_class.__name__}"

def test_memory_efficiency(implementations, regression_data):
    """Test memory usage during training"""
    from src.utils import MemoryTracker
    X_train, X_test, y_train, y_test = regression_data
    
    for impl in implementations:
        tracker = MemoryTracker()
        with tracker.track() as memory:
            impl.fit(X_train, y_train)
            impl.predict(X_test)
        
        # Check memory usage (should be reasonable)
        data_size = X_train.nbytes + y_train.nbytes
        assert memory < data_size * 10, f"High memory usage in {impl.__class__.__name__}" 