using DataFrames
using LinearAlgebra
using MAT
using  AugmentedGaussianProcesses
using GLMakie

span=1:20:20000
train_data = matread("./data/sarcos_inv.mat")
test_data=matread("./data/sarcos_inv_test.mat")
data=train_data["sarcos_inv"]
xs=data[1:5:20000,1:21]
rows=size(xs,1)
ys=data[1:5:20000,22]
kernel = with_lengthscale(SqExponentialKernel(), 1.0) # We create a standard kernel with lengthscale 1
    
Ms = [10,20,300];
models = Vector{AbstractGPModel}(undef, length(Ms));
σ = 0.05

 function agp(X,y)
    kernel = with_lengthscale(SqExponentialKernel(), 1.0)
    for (index, num_inducing) in enumerate(Ms)
        @info "Training with $(num_inducing) points"
        m = SVGP(
            kernel, # Kernel
            GaussianLikelihood(σ), # Likelihood used
            AnalyticVI(), # Inference usede to solve the problem
            inducingpoints(KmeansAlg(num_inducing), X); # Inducing points initialized with kmeans
            optimiser=false, # Keep kernel parameters fixed
            Zoptimiser=false, # Keep inducing points locations fixed
        )
        
        @time train!(m, X, y, 100) # Train the model for 100 iterations
        models[index] = m # Save the model in the array
    end
end

agp(xs,ys)


function plot_ys()
    fig=Figure(resolution=(800,600))
    ax1=Axis(fig[1,1])
    y_grid, sig_y_grid = proba_y(models[3], xs)
    scatter!(ax1,1:rows...,ys,marker=:circle,markersize=10,color=(:lightgreen,0.2),strokewidth=1,strokecolor=:black)
    σ_arr=sqrt.(sig_y_grid)
    band!(ax1,1:rows...,y_grid-σ_arr,y_grid+σ_arr,color=(:red,0.6))
    fig
end

#plot_ys()