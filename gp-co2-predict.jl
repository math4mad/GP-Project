"""
kernel define  from  AbstractGPs.jl example
"""

using GaussianProcesses,CSV,GLMakie,DataFrames


urls(str) = "./data/$str.csv"
get(str)=urls(str)|>CSV.File|>DataFrame|>dropmissing
data=get("CO2_data")
data=DataFrame(year =data[:,1],co2=data[:,2])
train_data=filter(row->row.year|>floor|>Int<2004 ,data)
dataof(yr::Int)=filter(row->row.year|>floor|>Int==yr ,data)
dataof(yr::Array{Int64})=filter(row->row.year|>floor|>Int in yr  ,data)
data_2004=dataof(2004)
data_span=dataof([2004:2014...])

xtrain = train_data[:,:year]; ytrain = train_data[:,:co2];
xtest = data_span[:,:year]; ytest=data_span[:,:co2]

kernel = SE(4.0,4.0) + Periodic(0.0,1.0,0.0)*SE(4.0,0.0) + RQ(0.0,0.0,-1.0) + SE(-2.0,-2.0);

gp = GP(xtrain,ytrain,MeanZero(),kernel,-2.0)

optimize!(gp)

function plot_models()
    μ, Σ = predict_y(gp,xtest);
    fig=Figure(resolution=(800,600))
    
    ax1=Axis(fig[1,1],title="GP co2 prediction",titlealign = :center)
    scatter!(ax1,xtest,ytest,marker=:circle,markersize=10,color=(:lightgreen,0.2),strokewidth=1,strokecolor=:black,label="Observations")
    σ_arr=sqrt.(Σ)
    band!(ax1,xtest,μ-σ_arr,μ+σ_arr,color=(:red,0.6),label="Predictions")
    #save("gp-co2-xtest-fit-with-optimization.png",fig)
    fig
    
end 

plot_models()