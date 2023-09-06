using GaussianProcesses,Random,Plots,Optim

Random.seed!(20140430)
# Training data
n=100; 
xs=range(-2, stop=2, length=n)                         #number of training points
           #predictors
f(x) =5*x^3-(10*x^2)+(0.5*x)  #original function

ys=[f(x)+10*rand() for x in xs]



mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
 logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
 gp = GP(xs,ys,mZero,kern,logObsNoise)  
 #optimize!(gp; method=ConjugateGradient())



 p4=plot(gp; legend=false, fmt=:png)
 #savefig(p4,"./image/multinomial-gp-regression.png")

