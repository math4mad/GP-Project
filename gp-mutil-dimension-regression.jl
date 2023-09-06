"""
数据来自 :https://www.scribbr.com/statistics/multiple-linear-regression/

自行车骑行时间与心脏病是负相关

"""

using GaussianProcesses,Random,Plots,Optim,CSV,DataFrames

src="/Users/lunarcheung/Public/Julia-Code/JuliaProject/GP-Project/data/heart.data.csv"

data=CSV.File(src)|>DataFrame|>Matrix|>transpose



function mutlidimensionreg()
    X=data[2:3,:]
    y=data[4,:]
    mZero = MeanZero()                            
    kern = Matern(5/2,[0.0,0.0],0.0)
    
    gp = GP(X,y,mZero,kern,-2.0)  
    
    optimize!(gp)  
    
    surface(gp;title="Gaussian process",  legend=false, fmt=:png)
end

function singlevarreg(;var="bike")
    idx = var=="bike" ? 2 : 3
    X=data[idx,:]
    y=data[4,:]

    mZero = MeanZero()                   
    kernel = SE(0.0,0.0)                   

    logObsNoise = -1.0                     
gp = GP(X,y,mZero,kernel,logObsNoise) 

plot(gp; obsv=false,title="var=$var")
optimize!(gp)
plot(gp;obsv=true, label="GP posterior mean", fmt=:png)
samples = rand(gp, X, 5)
plot!(X, samples)

end

singlevarreg(;var="smoke")


