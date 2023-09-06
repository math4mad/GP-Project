using ApproximateGPs, Random,Distributions, LinearAlgebra
rng = MersenneTwister(1453)  # set a random seed


f = GP(Matern32Kernel())

x = rand(rng, 100)
fx = f(x, 0.1)  # Observe the GP with Gaussian observation noise (σ² = 0.1)
y = rand(rng, f(x))  # Sample from the GP prior at x


exact_posterior = posterior(fx, y)


M = 15  # The number of inducing points
z = x[1:M]



q = MvNormal(zeros(length(z)), I)


fz = f(z, 1e-6)  # 'observe' the process at z with some jitter for numerical stability 
approx = SparseVariationalApproximation(fz, q)  # Instantiate everything needed for the approximation

sva_posterior = posterior(approx)  # Create the approximate posterior


# elbo(SparseVariationalApproximation(fz, q), fx, y)


# SparseVariationalApproximation(Centered(), fz, q)
# SparseVariationalApproximation(NonCentered(), fz, q)

