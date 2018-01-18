#Computing the efficient Frontier for the Dow Jones 30 Index

#Importing libraries 
import datetime as dt
import pandas as pd
import numpy as np
import random
import pylab # To display the plot formed
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas_datareader import data, wb
import pandas_datareader.data as pdr
import quandl

comp = ["DIS","KO","NKE","HD","JNJ","JPM","PFE","V","MMM","BA","VZ","AXP","MCD","DD","PG","CSCO","TRV","UTX","UNH","IBM","WMT","XOM","GE","CAT","GS","MSFT","MRK","CVX","INTC","AAPL"]
#Dow Jones 30 Companies

#Defining the start and the end dates 
st = dt.datetime(2016,8,1)
end = dt.datetime(2016,8,31)

#Extrating the Adjusted Closing Price from the Yahoo Finance 
f = pdr.DataReader(comp, 'yahoo', st, end)
f1 = pd.DataFrame(f.ix['Adj Close'])

#Treasury bond rates
treasury = quandl.get("USTREASURY/YIELD")
tb = np.matrix(treasury.iloc[-71:-49,2])

A = np.transpose(f1.pct_change(1))
B= A[A.columns[1:23]] 
C = np.concatenate((B,tb),axis=0)
avg = np.mean(C,axis=1) 

#Getting the covariance matrix
p = np.matrix(np.cov(C))

#Optimal matrix
r1 = np.arange(-0.0009,0.002,0.00001)
Optimal_Matrix = np.empty([2,np.size(r1)])
temp=avg

for i,r in enumerate(r1,1):
	OW1 = np.linalg.inv(p).dot(temp)*r
	OW2 = np.transpose(temp).dot(np.linalg.inv(p)).dot(temp)
	OW = OW1/OW2
	OS = np.sqrt(np.absolute(np.dot(np.dot(np.transpose(OW),p),OW)))
	Optimal_Matrix[0,i-1]=r
	Optimal_Matrix[1,i-1]=OS

#polyfit - The polyfit is the best function we could use to rerpesent the efficient frontier.
z = np.polyfit(Optimal_Matrix[0,:],Optimal_Matrix[1,:],3)
q = np.poly1d(z)
xp = np.linspace(-0.001,0.002)

#Plotting the Efficient Frontier obtained 
plt.plot(Optimal_Matrix[1,:], Optimal_Matrix[0,:],'o', q(xp), xp, '-')
plt.xlabel("std")
plt.ylabel("mean")
pylab.show()
