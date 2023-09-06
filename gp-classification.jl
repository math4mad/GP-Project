using GaussianProcesses,RDatasets
import Distributions:Normal
using Random
using MLDataUtils: shuffleobs, stratifiedobs, rescale!
Random.seed!(113355)

# Import the "Default" dataset.
data = RDatasets.dataset("ISLR", "Default");

# Convert "Default" and "Student" to numeric values.
data[!, :DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
data[!, :StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]

# Delete the old columns which say "Yes" and "No".
select!(data, Not([:Default, :Student]))

# Show the first six rows of our edited dataset.
#first(data, 6)
function split_data(df, target; at=0.70)
    shuffled = shuffleobs(df)
    return trainset, testset = stratifiedobs(row -> row[target], shuffled; p=at)
end

features = [:StudentNum, :Balance, :Income]
numerics = [:Balance, :Income]
target = :DefaultNum

trainset, testset = split_data(data, target; at=0.05)
for feature in numerics
    μ, σ = rescale!(trainset[!, feature]; obsdim=1)
    rescale!(testset[!, feature], μ, σ; obsdim=1)
end

# Turing requires data in matrix form, not dataframe
train = Matrix(trainset[:, features])
test = Matrix(testset[:, features])
train_label = trainset[:, target]
test_label = testset[:, target];


mZero = MeanZero();               # Zero mean function
kern = Matern(3/2,zeros(3),0.0);   # Matern 3/2 ARD kernel (note that hyperparameters are on the log scale)
#lik = BernLik();

gp = GP(train',train_label,mZero,kern)   

set_priors!(gp.kernel,[Normal(0.0,2.0) for i in 1:4])

samples = mcmc(gp; nIter=10000, burn=1000, thin=10);