"""
Probability Distributions
=========================
Common probability distributions and their applications.

Topics:
- Normal (Gaussian) distribution
- Binomial distribution
- Poisson distribution
- Uniform distribution
- Distribution properties
- Real-world applications

Run: python 02_distributions.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Probability Distributions")
    print("=" * 60)

    # 1. Normal Distribution
    print("\n1. Normal (Gaussian) Distribution")
    print("-" * 40)

    # Parameters
    mu = 100      # Mean
    sigma = 15    # Standard deviation

    print(f"Normal distribution: N(μ={mu}, σ={sigma})")
    print(f"\nProperties:")
    print(f"  - Bell-shaped, symmetric around mean")
    print(f"  - Mean = Median = Mode = {mu}")
    print(f"  - 68% of data within ±1σ")
    print(f"  - 95% of data within ±2σ")
    print(f"  - 99.7% of data within ±3σ")

    # Generate random samples
    np.random.seed(42)
    normal_samples = np.random.normal(mu, sigma, 1000)

    print(f"\nGenerated {len(normal_samples)} samples:")
    print(f"  Sample mean: {normal_samples.mean():.2f} (expected: {mu})")
    print(f"  Sample std: {normal_samples.std():.2f} (expected: {sigma})")

    # Probability density function
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    pdf = stats.norm.pdf(x, mu, sigma)

    print(f"\nProbability calculations:")
    # P(X < 115)
    prob_less = stats.norm.cdf(115, mu, sigma)
    print(f"  P(X < 115) = {prob_less:.4f} ({prob_less*100:.2f}%)")

    # P(85 < X < 115)
    prob_between = stats.norm.cdf(115, mu, sigma) - stats.norm.cdf(85, mu, sigma)
    print(f"  P(85 < X < 115) = {prob_between:.4f} ({prob_between*100:.2f}%)")

    # Find value at 95th percentile
    percentile_95 = stats.norm.ppf(0.95, mu, sigma)
    print(f"  95th percentile: {percentile_95:.2f}")

    # Z-scores
    values = [85, 100, 115, 130]
    print(f"\nZ-scores (standardized values):")
    for val in values:
        z = (val - mu) / sigma
        print(f"  X={val}: z={(val-mu)/sigma:.2f} ({abs(z):.1f} std devs from mean)")

    # 2. Binomial Distribution
    print("\n2. Binomial Distribution")
    print("-" * 40)

    n = 10        # Number of trials
    p = 0.3       # Probability of success

    print(f"Binomial distribution: B(n={n}, p={p})")
    print(f"\nProperties:")
    print(f"  - Discrete distribution")
    print(f"  - Fixed number of independent trials")
    print(f"  - Each trial has two outcomes (success/failure)")
    print(f"  - Constant probability of success")

    # Expected value and variance
    expected = n * p
    variance = n * p * (1 - p)
    print(f"\nTheoretical:")
    print(f"  Expected value (mean): {expected:.2f}")
    print(f"  Variance: {variance:.2f}")
    print(f"  Standard deviation: {np.sqrt(variance):.2f}")

    # Generate samples
    binomial_samples = np.random.binomial(n, p, 1000)
    print(f"\nGenerated {len(binomial_samples)} samples:")
    print(f"  Sample mean: {binomial_samples.mean():.2f}")
    print(f"  Sample variance: {binomial_samples.var():.2f}")

    # Probability mass function
    k = np.arange(0, n+1)
    pmf = stats.binom.pmf(k, n, p)

    print(f"\nProbability calculations:")
    print(f"  P(X = 3) = {stats.binom.pmf(3, n, p):.4f}")
    print(f"  P(X <= 4) = {stats.binom.cdf(4, n, p):.4f}")
    print(f"  P(X >= 2) = {1 - stats.binom.cdf(1, n, p):.4f}")

    # Real-world example
    print(f"\nExample: Coin flips")
    print(f"  Flip 10 coins, probability of heads = 0.3")
    print(f"  Expected number of heads: {expected:.1f}")
    print(f"  P(exactly 3 heads) = {stats.binom.pmf(3, n, p):.4f}")

    # 3. Poisson Distribution
    print("\n3. Poisson Distribution")
    print("-" * 40)

    lambda_param = 5  # Average rate

    print(f"Poisson distribution: Pois(λ={lambda_param})")
    print(f"\nProperties:")
    print(f"  - Discrete distribution")
    print(f"  - Models number of events in fixed interval")
    print(f"  - Events occur independently")
    print(f"  - Average rate is constant")

    print(f"\nTheoretical:")
    print(f"  Mean: {lambda_param}")
    print(f"  Variance: {lambda_param}")
    print(f"  Standard deviation: {np.sqrt(lambda_param):.2f}")

    # Generate samples
    poisson_samples = np.random.poisson(lambda_param, 1000)
    print(f"\nGenerated {len(poisson_samples)} samples:")
    print(f"  Sample mean: {poisson_samples.mean():.2f}")
    print(f"  Sample variance: {poisson_samples.var():.2f}")

    # Probability mass function
    k_poisson = np.arange(0, 15)
    pmf_poisson = stats.poisson.pmf(k_poisson, lambda_param)

    print(f"\nProbability calculations:")
    print(f"  P(X = 5) = {stats.poisson.pmf(5, lambda_param):.4f}")
    print(f"  P(X <= 3) = {stats.poisson.cdf(3, lambda_param):.4f}")
    print(f"  P(X > 7) = {1 - stats.poisson.cdf(7, lambda_param):.4f}")

    # Real-world example
    print(f"\nExample: Customer arrivals")
    print(f"  Average 5 customers per hour")
    print(f"  P(exactly 5 customers) = {stats.poisson.pmf(5, lambda_param):.4f}")
    print(f"  P(more than 7 customers) = {1 - stats.poisson.cdf(7, lambda_param):.4f}")

    # 4. Uniform Distribution
    print("\n4. Uniform Distribution")
    print("-" * 40)

    a, b = 0, 10  # Range

    print(f"Uniform distribution: U(a={a}, b={b})")
    print(f"\nProperties:")
    print(f"  - All values equally likely")
    print(f"  - Constant probability density")

    # Theoretical
    mean_uniform = (a + b) / 2
    variance_uniform = (b - a) ** 2 / 12
    print(f"\nTheoretical:")
    print(f"  Mean: {mean_uniform:.2f}")
    print(f"  Variance: {variance_uniform:.2f}")
    print(f"  Standard deviation: {np.sqrt(variance_uniform):.2f}")

    # Generate samples
    uniform_samples = np.random.uniform(a, b, 1000)
    print(f"\nGenerated {len(uniform_samples)} samples:")
    print(f"  Sample mean: {uniform_samples.mean():.2f}")
    print(f"  Sample std: {uniform_samples.std():.2f}")

    print(f"\nProbability calculations:")
    print(f"  P(X < 5) = {stats.uniform.cdf(5, a, b-a):.4f}")
    print(f"  P(3 < X < 7) = {stats.uniform.cdf(7, a, b-a) - stats.uniform.cdf(3, a, b-a):.4f}")

    # 5. Comparing Distributions
    print("\n5. Comparing Distributions")
    print("-" * 40)

    print("Distribution characteristics:")
    print("\nNormal:")
    print("  Type: Continuous, symmetric")
    print("  Use: Natural phenomena, measurement errors")
    print("  Examples: Heights, test scores, temperatures")

    print("\nBinomial:")
    print("  Type: Discrete, trials with success/failure")
    print("  Use: Fixed number of independent trials")
    print("  Examples: Coin flips, quality control, survey responses")

    print("\nPoisson:")
    print("  Type: Discrete, count of events")
    print("  Use: Rare events over time/space")
    print("  Examples: Customer arrivals, defects per unit, calls per hour")

    print("\nUniform:")
    print("  Type: Continuous, all values equally likely")
    print("  Use: Random selection, simulation")
    print("  Examples: Random number generation, waiting times")

    # 6. Visualization
    print("\n6. Visualizing Distributions")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Normal distribution PDF
    x_normal = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    axes[0, 0].plot(x_normal, stats.norm.pdf(x_normal, mu, sigma), 'b-', linewidth=2)
    axes[0, 0].fill_between(x_normal, stats.norm.pdf(x_normal, mu, sigma), alpha=0.3)
    axes[0, 0].axvline(mu, color='r', linestyle='--', label=f'Mean={mu}')
    axes[0, 0].axvline(mu-sigma, color='orange', linestyle=':', alpha=0.7)
    axes[0, 0].axvline(mu+sigma, color='orange', linestyle=':', alpha=0.7, label='±1σ')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title(f'Normal Distribution N({mu}, {sigma}²)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Normal distribution samples histogram
    axes[0, 1].hist(normal_samples, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[0, 1].plot(x_normal, stats.norm.pdf(x_normal, mu, sigma), 'r-', linewidth=2, label='Theoretical')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Normal: Samples vs Theoretical')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Binomial distribution PMF
    axes[0, 2].bar(k, pmf, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(expected, color='r', linestyle='--', linewidth=2, label=f'Mean={expected:.1f}')
    axes[0, 2].set_xlabel('Number of Successes')
    axes[0, 2].set_ylabel('Probability')
    axes[0, 2].set_title(f'Binomial Distribution B({n}, {p})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Binomial distribution samples
    axes[1, 0].hist(binomial_samples, bins=np.arange(-0.5, n+1.5, 1), density=True,
                    alpha=0.7, edgecolor='black')
    axes[1, 0].plot(k, pmf, 'ro-', linewidth=2, markersize=6, label='Theoretical')
    axes[1, 0].set_xlabel('Number of Successes')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title('Binomial: Samples vs Theoretical')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Poisson distribution PMF
    axes[1, 1].bar(k_poisson, pmf_poisson, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(lambda_param, color='r', linestyle='--', linewidth=2,
                      label=f'Mean={lambda_param}')
    axes[1, 1].set_xlabel('Number of Events')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_title(f'Poisson Distribution (λ={lambda_param})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Uniform distribution
    axes[1, 2].hist(uniform_samples, bins=20, density=True, alpha=0.7, edgecolor='black')
    x_uniform = np.linspace(a, b, 100)
    axes[1, 2].plot(x_uniform, stats.uniform.pdf(x_uniform, a, b-a), 'r-',
                   linewidth=2, label='Theoretical')
    axes[1, 2].axhline(1/(b-a), color='orange', linestyle='--', alpha=0.7)
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title(f'Uniform Distribution U({a}, {b})')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/distributions.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("Created visualizations:")
    print("  - Normal distribution PDF and samples")
    print("  - Binomial distribution PMF and samples")
    print("  - Poisson distribution PMF")
    print("  - Uniform distribution PDF and samples")
    print("  Saved to: /tmp/distributions.png")

    print("\n" + "=" * 60)
    print("Summary:")
    print("Probability distributions model random phenomena")
    print("- Normal: Continuous, symmetric, natural measurements")
    print("- Binomial: Discrete, fixed trials with two outcomes")
    print("- Poisson: Discrete, rare events over intervals")
    print("- Uniform: Continuous, all values equally likely")
    print("Choose distribution based on data characteristics")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Generate 1000 samples from N(50, 10) and verify mean/std")
    print("2. Calculate P(X=7) for B(n=15, p=0.4)")
    print("3. Find P(X≤3) for Poisson with λ=2.5")
    print("4. Simulate 20 coin flips with p=0.6, count heads")
    print("5. Compare sample histogram to theoretical PDF")
    print("=" * 60)

if __name__ == "__main__":
    main()
