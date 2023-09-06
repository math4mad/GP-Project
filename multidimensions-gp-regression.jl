using GaussianProcesses,Random,Plots,Optim


#Training data
d, n = 2, 50;         #Dimension and number of observations
x = 2π * rand(d, n);                               #Predictors
y = vec(sin.(x[1,:]).*sin.(x[2,:])) + 0.05*rand(n);  #Responses

#plot(y)

"""
For problems of dimension>1 we can use isotropic (Iso) kernels or automatic relevance determination (ARD) kernels. For Iso kernels, the length scale parameter ℓ
ℓ is the same for all dimensions. For ARD kernels, each dimension has different length scale parameter.
"""


mZero = MeanZero()                             # Zero mean function
kern = Matern(5/2,[0.0,0.0],0.0) + SE(0.0,0.0)


gp = GP(x,y,mZero,kern,-2.0)

optimize!(gp)



img=surface(gp ,fmt=:png)
    contour!(gp)
#savefig(img,"./image/multi-dimension-gp-regression-3.png")