data1= read.csv("hw4data.csv")
head(data1)
y2=ts(data1$y2, start=1980, freq=12)
y2.in=window(y2, end=c(2000,12))
y2.out=window(y2, start=2001)
tail(y2.in)
y2.in
three.plot=function(y){
	op=par(mfrow=c(3,1), mar=c(2,4,1,0.5))
	plot.ts(y, ylab="y")
	acf(y)
	pacf(y)
	par(op)
	}

three.plot(y2.in)
#AR(1), AR(6), MA(1), MA(7), ARMA(1,1)

out.ar1=arima(y2.in,order=c(1,0,0))
out.ar6=arima(y2.in,order=c(6,0,0))
out.ma1=arima(y2.in,order=c(0,0,1))
out.ma7=arima(y2.in,order=c(0,0,7))
out.arma=arima(y2.in,order=c(1,0,1))
AIC(out.ar1, out.ar6, out.ma1, out.ma7, out.arma)
BIC(out.ar1, out.ar6, out.ma1, out.ma7, out.arma)
#AR(6) or MA(7) are the best models

y.osh=function(y, start,p,q){
n.fcst= length(y)- start+1
y.osh= rep(NA,n.fcst)
for(i in 1:n.fcst){
	out.i= arima(y[1:(start-2+i)], order=c(p,0,q))
	y.osh[i]= predict(out.i, n.ahead=1)$pred 
	}
	y.osh
	}
four.measures= function(fe){
	me=mean(fe)
	mse=mean(fe^2)
	rmse=sqrt(mse)
	mae=mean(abs(fe))
	four=cbind(me,mse,rmse,mae)
	print(four)
	}
out.osh=function(y,start,p,q){
y.osh=y.osh(y,start,p,q)
fe= y.osh-y[start:length(y)]
four.measures(fe)
	}
start0= length(y2.in)+1

y.fcst.ar6=y.osh(y2, start= start0, p=6, q=0)  #AR(6) forecasts
y.fcst.ar6=ts(y.fcst.ar6,start=2001,freq=12)
y.two= ts.union(y2,y.fcst.ar6, dframe= TRUE)

plot(y2, main="AR(6) Actual VAlues W/ Forecasts")
lines(y.two$y.fcst, col=4)
legend("topleft", legend= c("Actual", "Forecast"), col= c("black", "blue"), lty=1)




y.fcst.ma7= y.osh(y2, start= start0, p=0, q=7) #MA(7)forecasts
y.two= ts.union(y2.in, y.fcst.ma7, dframe= TRUE)
plot(y2, main="MA(7) Actual Values W/ Forecasts")
lines(y.two$y.fcst, col=4)
legend("topleft", legend= c("Actual", "Forecast"), col= c("black", "blue"), lty=1)

out.osh(y2, start= start0, p=6, q=0) #AR(6) four measures
out.osh(y2, start= start0, p=0, q=7) #MA(7) four measures











data2= read.table("cbe(1).dat", header=TRUE)
head(data2)
y=ts(data2$choc, start= 1958, freq=12)
y=log(y)
three.plot(y)
time=1:length(y)
time=ts(time, start= 1958, freq=12)
require(uroot)
sd=seasonal.dummies(y)
sd= ts(sd, start=1958, freq=12)
y.in= window(y, end=c(1980,12))
time.in= 1:length(y.in)
sd.in= seasonal.dummies(y.in)
out1= lm(y.in~ time.in + sd.in[,-1])
out2= lm(y.in~ time.in + I(time.in^2) + sd.in[,-1])
out3= lm(y.in~ time.in + I(time.in^2) + I(time.in^3) + sd.in[,-1])
out4= lm(y.in~ time.in + I(time.in^2) + I(time.in^3) + I(time.in^4) + sd.in[,-1])

AIC(out1, out2, out3, out4)
BIC(out1, out2, out3, out4)
summary(out2)
#quadratic is best 
three.plot(out2$resid)
#AR(3), MA(6), ARMA(1,1) 


out.ma6=arima(y.in, order=c(0, 0, 6), xreg= cbind(time.in, time.in^2, sd.in[,-1]), include.mean=TRUE)
out.ar3=arima(y.in, order=c(3, 0, 0), xreg= cbind(time.in, time.in^2, sd.in[,-1]), include.mean=TRUE)
out.arma=arima(y.in, order=c(1, 0, 1), xreg= cbind(time.in, time.in^2, sd.in[,-1]), include.mean=TRUE)
AIC(out.ma6, out.ar3, out.arma)
BIC(out.ma6, out.ar3, out.arma)
#AR(3) or ARMA(1,1)

start0= length(y.in+1)
out.osh(y, start= start0, p=3, q=0)
out.osh(y, start= start0, p=1, q=1)

y.osh=function(y, start,p,q){
n.fcst= length(y)- start+1
time= 1:length(y)
require(uroot)
sd=seasonal.dummies(y)
y.osh= rep(NA,n.fcst)
for(i in 1:n.fcst){
	end.in= start - 2+ i
	out.i= arima(y[1:end.in], order=c(p,0,q), xreg=cbind(time[1:end.in], sd[1:end.in,-1]))
	y.osh[i]= predict(out.i, n.ahead=1, newxreg= cbind(end.in+1, t(sd[end.in+1,-1])))$pred
	}
	y.osh
	}
y.fcst.ar3= y.osh(y, start= start0, p=3, q=0)#AR(3)
y.fcst.ar3=ts(y.fcst.ar3, start=1981, frequency=12)
plot(y, main= "AR(3) Actual Values W/ Forecasts")
y.two= ts.union(y, y.fcst.ar3, dframe= TRUE)
lines(y.two$y.fcst, col=4)
legend("topleft", legend= c("Actual", "Forecast"), col= c("black", "blue"), lty=1)




y.fcst.arma= y.osh(y, start= start0, p=1, q=1)#ARMA
y.fcst.arma= ts(y.fcst.arma, start=1981, freq=12) 
plot(y, main="ARMA(1,1) Actual Values W/ Forecasts")
y.two= ts.union(y, y.osh, dframe= TRUE)
lines(y.two$y.fcst, col=4)
legend("topleft", legend= c("Actual", "Forecast"), col= c("black", "blue"), lty=1)



out.osh(y, start= start0, p=3, q=0) #AR(3) four measures
out.osh(y2, start= start0, p=0, q=7) #ARMA four measures




