"""
from https://turinglang.org/v0.24/tutorials/07-possion-regression/

"""

using GaussianProcesses,Random,Plots,Optim,MAT,DataFrames,LinearAlgebra,Distributions
Random.seed!(12);

theta_noalcohol_meds = 1    # no alcohol, took medicine
theta_alcohol_meds = 3      # alcohol, took medicine
theta_noalcohol_nomeds = 6  # no alcohol, no medicine
theta_alcohol_nomeds = 36   # alcohol, no medicine

# no of samples for each of the above cases
q = 100

#Generate data from different Poisson distributions
noalcohol_meds = Poisson(theta_noalcohol_meds)
alcohol_meds = Poisson(theta_alcohol_meds)
noalcohol_nomeds = Poisson(theta_noalcohol_nomeds)
alcohol_nomeds = Poisson(theta_alcohol_nomeds)

nsneeze_data = vcat(
    rand(noalcohol_meds, q),
    rand(alcohol_meds, q),
    rand(noalcohol_nomeds, q),
    rand(alcohol_nomeds, q),
)
alcohol_data = vcat(zeros(q), ones(q), zeros(q), ones(q))
meds_data = vcat(zeros(q), zeros(q), ones(q), ones(q))

df = DataFrame(;
    nsneeze=nsneeze_data,
    alcohol_taken=alcohol_data,
    nomeds_taken=meds_data,
    product_alcohol_meds=meds_data .* alcohol_data,
)
df[sample(1:nrow(df), 5; replace=false), :]


data = Matrix(df[:, [:alcohol_taken, :nomeds_taken, :product_alcohol_meds]])
data_labels = df[:, :nsneeze]

data = (data .- mean(data; dims=1)) ./ std(data; dims=1)

k = Matern(3/2,zeros(3),0.0)   # Matern 3/2 kernel
l = PoisLik()             # Poisson likelihood
gpmc = GP(data', vec(data_labels), MeanZero(), k, l)
#gpvi = GP(X, vec(Y), MeanZero(), k, l)



set_priors!(gpmc.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0),Normal(-2.0,4.0),Normal(-2.0,4.0)])
 @time samples = mcmc(gpmc; nIter=10000);

#optimize!(gpmc)     
