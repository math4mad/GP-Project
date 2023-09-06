using GaussianProcesses,Random,Plots,Optim,LaTeXStrings

Random.seed!(20140430)
# Training data
# time=[0,30,60,90,120].|>Float64

# temp=[69.58,66.11,61.41,58.07,55.60]

time=[

30
60
90
120
150
180
210
240
270
300
330
360
390
420
450
480
510
540
570
600
630
660
690
720
750
780
810
840
870
900
930
960
990]

# temp=[
# 69.58 
# 66.11 
# 61.41 
# 58.07 
# 55.60 
# 53.58 
# 51.66 
# 50.05 
# 48.52 
# 47.24 
# 46.00 
# 44.96 
# 43.96 
# 42.92 
# 41.95 
# 41.05 
# 40.18 
# 39.40 
# 38.70 
# 38.00 
# 37.32 
# 36.67 
# 36.08 
# 35.50 
# 35.00 
# 34.53 
# 34.04 
# 33.59 
# 33.20 
# 32.76 
# 32.37 
# 32.00 
# 31.64 
# 31.30]

time=range(0.0,1000,step=100)
f(t)=(256/(1.55034+0.00082*t))+22.3
temp=[f(t)+rand() for t in time]

#scatter(time, temp, label=false,xlabel=L"time(s)",ylabel=L"temp(℃)",frame=:box,ms=3,alpha=0.5,mc=:lightgreen)


mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
 #logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
gp = GP(time,temp,mZero,kern) 


optimize!(gp; method=ConjugateGradient())

 p2=plot(gp; legend=false, fmt=:png,title="optim with Optim.jl")
# savefig(p2,"./image/gp-cup-cooling-procsss.png")