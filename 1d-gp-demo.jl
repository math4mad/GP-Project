using GaussianProcesses,Random,Plots,Optim

Random.seed!(20140430)
# Training data
n=10;                          #number of training points
x = 2π * rand(n);              #predictors
y = sin.(x) + 0.05*randn(n);   #regressors



#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
#kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

kernel=SE(4.0,4.0)+ RQ(0.0,0.0,-1.0) + SE(-2.0,-2.0)
 logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
 gp = GP(x,y,mZero,kernel,logObsNoise)       #Fit the GP


 #μ, σ² = predict_y(gp,range(0,stop=2π,length=100));

 #p1=plot(gp; xlabel="x", ylabel="y", title="Gaussian process", legend=false, fmt=:png) 

 #savefig(p1,"./image/1d-gp-demo.png")

 optimize!(gp; method=ConjugateGradient())

 p2=plot(gp; legend=false, fmt=:png,title="optim with Optim.jl")
 #savefig(p2,"./image/1d-gp-demo-2.png")