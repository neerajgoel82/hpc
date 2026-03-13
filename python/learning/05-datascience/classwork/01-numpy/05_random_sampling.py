"""
NumPy Random Sampling
=====================
Random number generation and statistical distributions.

Topics:
- Random number generators
- Common distributions (uniform, normal, binomial, etc.)
- Random sampling techniques
- Reproducibility with seeds
- Random array generation

Run: python 05_random_sampling.py
"""

import numpy as np

def main():
    print("=" * 60)
    print("NumPy Random Sampling")
    print("=" * 60)

    # 1. Basic random number generation
    print("\n1. Basic Random Number Generation")
    print("-" * 40)

    # Random float between 0 and 1
    print(f"Single random float [0, 1): {np.random.random()}")
    print(f"5 random floats: {np.random.random(5)}")

    # Random integers
    print(f"\nRandom int [0, 10): {np.random.randint(0, 10)}")
    print(f"5 random ints [0, 100): {np.random.randint(0, 100, size=5)}")

    # Random array with shape
    arr = np.random.random((3, 4))
    print(f"\nRandom 3x4 array:\n{arr}")

    # 2. Random seeds for reproducibility
    print("\n2. Random Seeds (Reproducibility)")
    print("-" * 40)

    # Set seed
    np.random.seed(42)
    print(f"With seed 42: {np.random.random(3)}")

    # Reset seed to get same results
    np.random.seed(42)
    print(f"With seed 42 again: {np.random.random(3)}")

    # Different seed
    np.random.seed(123)
    print(f"With seed 123: {np.random.random(3)}")

    # 3. Uniform distribution
    print("\n3. Uniform Distribution")
    print("-" * 40)

    # Random floats in range [0, 1)
    uniform_01 = np.random.uniform(0, 1, size=5)
    print(f"Uniform [0, 1): {uniform_01}")

    # Random floats in range [10, 20)
    uniform_range = np.random.uniform(10, 20, size=5)
    print(f"Uniform [10, 20): {uniform_range}")

    # 2D uniform array
    uniform_2d = np.random.uniform(-1, 1, size=(3, 3))
    print(f"\nUniform [-1, 1) 3x3:\n{uniform_2d}")

    # 4. Normal (Gaussian) distribution
    print("\n4. Normal (Gaussian) Distribution")
    print("-" * 40)

    # Standard normal (mean=0, std=1)
    np.random.seed(42)
    standard_normal = np.random.randn(5)
    print(f"Standard normal (μ=0, σ=1): {standard_normal}")

    # Normal with custom mean and std
    mean = 100
    std = 15
    normal_custom = np.random.normal(mean, std, size=5)
    print(f"\nNormal (μ={mean}, σ={std}): {normal_custom}")

    # Large sample statistics
    np.random.seed(42)
    large_sample = np.random.normal(50, 10, size=10000)
    print(f"\nLarge sample (10000 points):")
    print(f"  Mean: {large_sample.mean():.2f} (expected: 50)")
    print(f"  Std: {large_sample.std():.2f} (expected: 10)")

    # 5. Other distributions
    print("\n5. Other Common Distributions")
    print("-" * 40)

    # Binomial (coin flips)
    np.random.seed(42)
    binomial = np.random.binomial(n=10, p=0.5, size=5)
    print(f"Binomial (10 trials, p=0.5): {binomial}")
    print("  (Number of successes in 10 coin flips)")

    # Poisson (rare events)
    poisson = np.random.poisson(lam=3, size=5)
    print(f"\nPoisson (λ=3): {poisson}")
    print("  (Number of events in a time period)")

    # Exponential (time between events)
    exponential = np.random.exponential(scale=2, size=5)
    print(f"\nExponential (scale=2): {exponential}")
    print("  (Time between events)")

    # Chi-square
    chi_square = np.random.chisquare(df=3, size=5)
    print(f"\nChi-square (df=3): {chi_square}")

    # Beta distribution
    beta = np.random.beta(a=2, b=5, size=5)
    print(f"\nBeta (a=2, b=5): {beta}")

    # 6. Random sampling
    print("\n6. Random Sampling from Arrays")
    print("-" * 40)

    # Sample from array
    arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(f"Original array: {arr}")

    # Random choice
    choice = np.random.choice(arr)
    print(f"\nRandom choice: {choice}")

    # Multiple choices (with replacement)
    choices = np.random.choice(arr, size=5)
    print(f"5 random choices (with replacement): {choices}")

    # Choices without replacement
    choices_unique = np.random.choice(arr, size=5, replace=False)
    print(f"5 random choices (without replacement): {choices_unique}")

    # Weighted choices
    probs = [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1]
    weighted_choices = np.random.choice(arr, size=10, p=probs)
    print(f"\nWeighted choices: {weighted_choices}")

    # 7. Shuffling and permutations
    print("\n7. Shuffling and Permutations")
    print("-" * 40)

    # Shuffle in place
    arr = np.arange(10)
    print(f"Original array: {arr}")

    np.random.shuffle(arr)
    print(f"Shuffled array: {arr}")

    # Permutation (returns shuffled copy)
    arr = np.arange(10)
    permuted = np.random.permutation(arr)
    print(f"\nOriginal (unchanged): {arr}")
    print(f"Permuted (new array): {permuted}")

    # Random permutation of indices
    indices = np.random.permutation(5)
    print(f"\nRandom index permutation: {indices}")

    # 8. Random arrays with specific properties
    print("\n8. Random Arrays with Specific Properties")
    print("-" * 40)

    # Random integers in range
    random_ints = np.random.randint(1, 101, size=(4, 5))
    print(f"Random integers [1, 100] (4x5):\n{random_ints}")

    # Random samples from standard distributions
    np.random.seed(42)

    # Create random correlation matrix
    n = 3
    A = np.random.randn(n, n)
    corr_matrix = A @ A.T  # Positive semi-definite
    print(f"\nRandom correlation-like matrix:\n{corr_matrix}")

    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(3, 3))
    print(f"\nRandom orthogonal matrix:\n{Q}")
    print(f"Q @ Q.T (should be identity):\n{Q @ Q.T}")

    # 9. Practical example: Monte Carlo simulation
    print("\n9. Monte Carlo Simulation Example")
    print("-" * 40)

    print("Estimating π using Monte Carlo:")
    np.random.seed(42)

    n_samples = 100000
    # Random points in [0, 1] x [0, 1]
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)

    # Count points inside quarter circle
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * inside_circle.sum() / n_samples

    print(f"  Samples: {n_samples}")
    print(f"  Inside circle: {inside_circle.sum()}")
    print(f"  π estimate: {pi_estimate:.6f}")
    print(f"  Actual π: {np.pi:.6f}")
    print(f"  Error: {abs(pi_estimate - np.pi):.6f}")

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Generate 1000 samples from normal distribution, plot histogram")
    print("2. Simulate 10000 dice rolls, calculate mean and std")
    print("3. Create random walk: cumsum of random steps {-1, +1}")
    print("4. Generate 5x5 matrix with random ints [1-100], no duplicates")
    print("=" * 60)

if __name__ == "__main__":
    main()
