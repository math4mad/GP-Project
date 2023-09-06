"""
using LinearRegression
"""

using Random,Plots,Optim,MAT,DataFrames,LinearAlgebra,MLJ
span=1:20:20000
train_data = matread("./data/sarcos_inv.mat")
test_data=matread("./data/sarcos_inv_test.mat")
#train_df=DataFrame(train_data["sarcos_inv"],:auto)
#test_df=DataFrame(test_data["sarcos_inv_test"],:auto)
 data=train_data["sarcos_inv"]
 #train_row,train_col=size(data)
 xs=data[1:5:20000,1:21]'
 #xs= convert(Matrix,data[1:1000,1:27]');   
 #ys=data[1:1000,28]
 ys=data[1:5:20000,22]

ze=zeros(21)
mZero = MeanZero()                             # Zero mean function
kernel1= Matern(5/2,fill(0.2,21),0.0)+SE(3.0,3.0)
kernel2=SqExponentialKernel()
kernel3=5.0 * SqExponentialKernel() ∘ ScaleTransform(1.0)
kernel4= kern=Mat(5/2,0.0,0.0) + SE(0.0,0.0)

gp = GaussianProcesses.GP(xs,ys,mZero,kernel1)

optimize!(gp)

#plot(contour(gp) ,heatmap(gp); fmt=:png)


#test_data["sarcos_inv_test"]

