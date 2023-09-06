"""
1. Automatic Model Construction with Gaussian Processes.pdf page30
2. paper: https://www.sciencedirect.com/science/article/abs/pii/S0008884698001653
3. data:  https://www.kaggle.com/datasets/maajdl/yeh-concret-data
"""

import MLJ:predict,transform,fit!

using GaussianProcesses,Random,GLMakie,Optim,CSV,DataFrames,MLJ
Random.seed!(2332)

path="./data/Concrete_Data_Yeh.csv"
df = CSV.read(path, DataFrame)
X=df[!,1:end-1]
y=df[!,end]|>Vector.|>Float64

Standardizer = @load Standardizer pkg=MLJModels
stand1 = Standardizer()
starndarX=transform(fit!(machine(stand1, X)), X)|>Matrix|>transpose


mZero = MeanZero()                   
kern = kern = SEArd(fill(0.0,8),0.0) + SE(0.0,0.0)               


gp = GP(starndarX,y,mZero,kern)

#xs=range(extrema(X[:,2])...,100)

#(μ, σ²)= predict_y(gp,xs);


function plot_posterior()
    fig=Figure()
    ax=Axis(fig[1,1])
    scatter!(ax,X[:,2],y)
    #band!(ax,xs,μ-sqrt.(σ²),μ+sqrt.(σ²),color=(:red,0.5))
    series!(ax,xs,samples)
    fig
end

gp 


