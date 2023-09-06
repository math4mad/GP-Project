using GaussianProcesses,Random,Plots,Optim,MAT,DataFrames,LinearAlgebra
span=1:20:20000
train_data = matread("./data/samples.mat")
#  data=train_data["sarcos_inv"]
#  train_row,train_col=size(data)