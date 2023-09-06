"""
Automatic Model Construction with Gaussian Processes.pdf page 25
"""

using KernelFunctions,GLMakie,Random,Distributions,LinearAlgebra
Random.seed!(123)


SE=RBFKernel()

fig=Figure(resolution=(1200,600))
ax1=[Axis3(fig[1,i]) for i in 1:4]
ax2=[Axis(fig[2,i]) for i in 1:4]

function prior_sample(ax,ker)
    local xs=ys=range(-5,5,50)
    k1 = kernelmatrix(ker, xs)
    k2 = kernelmatrix(ker, ys)
    f=rand(MvNormal(k1+k2+1e-6*I))|>transpose
    surface!(ax,xs,ys,f,linewidth=2)
    #@info f
    
end

prior_sample(ax1[1],SE)
fig