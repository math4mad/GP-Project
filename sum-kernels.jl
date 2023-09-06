"""
Automatic Model Construction with Gaussian Processes.pdf page 24
"""


using KernelFunctions,GLMakie,Random,Distributions,LinearAlgebra
Random.seed!(123)

Lin=LinearKernel()
SE=RBFKernel()
Per=PeriodicKernel()

# sum  kernel to new kernel
k1=Lin+Per
k2=SE+Per
k3=SE+Lin
k4=SE∘ScaleTransform(2.0)+SE∘ScaleTransform(0.5)

kernels=[k1,k2,k3,k4]
titles=["Lin+Per","SE+Per","SE+Lin","SE(long)+SE(short)"]


fig=Figure(resolution=(1200,600))
ax1=[Axis(fig[1,i]) for i in 1:4]
ax2=[Axis(fig[2,i]) for i in 1:4]

function prior_sample(ax,ker)
    local xrange=range(-5,5,100)
    K = kernelmatrix(ker, xrange)
    f=rand(MvNormal(K+1e-6*I),2)|>transpose
    series!(ax,xrange,f,linewidth=2)
    ax.title="prior samples"
end

function ker_cut(i,ax,ker)
    local xrange=range(-5,5,100)
    ys=ker.(xrange, 1.0);
    lines!(ax,xrange,ys)
end

for i in 1:4
    prior_sample(ax1[i],kernels[i])
    ax1[i].title=L"%$(titles[i])"
    ax2[i].title="x(with x′=1)"
    ker_cut(i,ax2[i],kernels[i])
end
save("./image/sum-kernels.png",fig)



