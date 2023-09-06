"""
from   A Practical Guide to Gaussian Processes
https://infallible-thompson-49de36.netlify.app   figure2
"""

using KernelFunctions,GLMakie,Random,Distributions,LinearAlgebra
Random.seed!(123)

SE(length_scales)=with_lengthscale(SqExponentialKernel(), length_scales)

scales=[1.00,2.00,10.00]

fig=Figure(resolution=(900,300))
axs=[Axis(fig[1,i]) for i in 1:3]
xs=range(-5,5,100)

function prior_sample(idx)
    ker=SE(scales[idx])
    K = kernelmatrix(ker, xs)
    f=rand(MvNormal(K+1e-6*I),3)|>transpose
    series!(axs[idx],xs,f,linewidth=1)
    axs[idx].xlabel=L"Lengthscale =%$(scales[idx])"
    axs[idx].limits=(-5,5,-3,3)
end

[prior_sample(idx) for idx in 1:3]
fig
#save("./image/gp-with-different-lengthscales.png",fig)