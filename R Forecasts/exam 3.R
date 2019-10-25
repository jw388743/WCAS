data1=read.csv("exam3data.csv")
y=ts(data1$y1, start=1987, freq=12)
time=1:length(y)
time=ts(time, start=1987, freq=12)
require(uroot)
sd=seasonal.dummies(y)
y.in=window(y, end=c(2010,12))
y.out=window(y,start=2011)
time.in=window(time, end=c(2010,12))
time.out=window(time, start=2011)
sd.in=window(sd, end=c(2010,12))
sd.out=window(sd, start=2011)

out1= lm(y.in~ time.in + sd.in[,-1])
out2= lm(y.in~ time.in + I(time.in^2) + sd.in[,-1])
out3= lm(y.in~ time.in + I(time.in^2) + I(time.in^3) + sd.in[,-1])
out4= lm(y.in~ time.in + I(time.in^2) + I(time.in^3) + I(time.in^4) + sd.in[,-1])

AIC(out1, out2, out3, out4)
BIC(out1, out2, out3, out4)

three.plot=function(y, name="y"){
	op=par(mfrow=c(3,1), mar=c(2,4,1,0.5))
	plot.ts(y, ylab="y")
	acf(y)
	pacf(y)
	par(op)
	}
three.plot(out3$resid, name="out3.resid")

out.ar2=arima(y.in,order=c(2,0,0), xreg= cbind(time.in, time.in^2, time.in^3, sd.in[,-1]), include.mean=TRUE)

out.ar15=arima(y.in,order=c(15,0,0), xreg= cbind(time.in, time.in^2, time.in^3, sd.in[,-1]), include.mean=TRUE)
out.ma4=arima(y.in,order=c(0,0,4), xreg= cbind(time.in, time.in^2, time.in^3, sd.in[,-1]), include.mean=TRUE)
out.arma=arima(y.in,order=c(1,0,1), xreg= cbind(time.in, time.in^2, time.in^3, sd.in[,-1]), include.mean=TRUE)
AIC(out.ar2, out.ar15, out.ma4, out.arma)
BIC(out.ar2, out.ar15, out.ma4, out.arma)

op=par(mfrow=c(2,1), mar=c(2,4,1,0.5))
plot(out.ar2$resid)
plot(out.arma$resid)
par(op)

y.osh=function(y, start,p,q){
n.fcst= length(y)- start+1
time= 1:length(y)
require(uroot)
sd=seasonal.dummies(y)
y.osh= rep(NA,n.fcst)
for(i in 1:n.fcst){
	j= start - 2+ i
	out.i= arima(y[1:j], order=c(p,0,q), xreg=cbind(time[1:j], sd[1:j,-1]))
	y.osh[i]= predict(out.i, n.ahead=1, newxreg= cbind(j+1, t(sd[j+1,-1])))$pred
	}
	y.osh
	}
start0= length(y.in+1)
y.fcst.ar2= y.osh(y, start= start0, p=2, q=0)
y.fcst.ar2=ts(y.fcst.ar2, start=2011, frequency=12)
y.fcst.arma= y.osh(y, start= start0, p=1, q=1)
y.fcst.arma= ts(y.fcst.arma, start=2011, freq=12)
y.fcst.arma

three=ts.union(y, y.fcst.ar2, y.fcst.arma)
plot(y.out, main="Y.out W/ Forecasts")
lines(y.fcst.ar2, col=2)
lines(y.fcst.arma, col=3)
legend("topleft", legend=c("actual", "forecast-AR(2)","forecast-ARMA(1,1)"), col=1:3, lty=1)

fe.ar2= y.fcst.ar2-y.out
fe.arma= y.fcst.arma-y.out
four.measures= function(fe){
	me=mean(fe)
	mse=mean(fe^2)
	rmse=sqrt(mse)
	mae=mean(abs(fe))
	four=cbind(me,mse,rmse,mae)
	print(four)
	}
four.measures(fe.ar2)
four.measures(fe.arma)
out.osh=function(y,start,p,q){
y.osh=y.osh(y,start,p,q)
fe= y.osh-y[start:length(y)]
four.measures(fe)
	}
out.osh(y, start=start0, p=2, q=0)
out.osh(y, start=start0, p=1,q=1)

y=ts(data1$y2, start=1987, freq=12)
time=1:length(y)
time=ts(time, start=1987, freq=12)
require(uroot)
sd=seasonal.dummies(y)
dy=diff(y)
y.in=window(y, end=c(2010,12))
y.out=window(y,start=2011)
time.in=window(time, end=c(2010,12))
time.out=window(time, start=2011)
sd.in=window(sd, end=c(2010,12))
sd.out=window(sd, start=2011)


dy=diff(y)
three.plot(dy)

out1=lm(y~time)
out2=lm(y~time+I(time^2))
out3=lm(y~time+I(time^2)+I(time^3))
out4=lm(y~time+I(time^2)+I(time^3)+I(time^4))
AIC(out1,out2,out3,out4)
BIC(out1,out2,out3,out4)
AIC(out1, out2, out3, out4)
BIC(out1, out2, out3, out4)
three.plot(out4$resid)
out.ar4=arima(y,order=c(4,0,0), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)

out.ar6=arima(y,order=c(6,0,0), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)
out.ma12=arima(y,order=c(0,0,12), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)
out.arma=arima(y,order=c(1,0,1), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)
AIC(out.ar4, out.ar6, out.ma12, out.arma)
BIC(out.ar4, out.ar6, out.ma12, out.arma)


three.plot=function(y, name="y"){
	op=par(mfrow=c(3,1), mar=c(2,4,1,0.5))
	plot.ts(y, ylab="y")
	acf(y)
	pacf(y)
	par(op)
	}
three.plot(out4$resid, name="out4.resid")

time2=(1:36)+length(y)
yfcst.ma12=predict(out.ma12,n.ahead=36, newxreg=cbind(time2, time2^2, time2^3, time2^4))
yfcst.ma12=yfcst.ma12$pred
yfcst.ma12
dy=diff(y)
require(urca)
df.y= ur.df(y, type= "trend", selectlags="BIC")
summary(df.y)
df.dy= ur.df(dy, type= "none", selectlags="BIC")
summary(df.dy)

out.ar4=arima(y,order=c(4,1,0), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)

out.ar6=arima(y,order=c(6,1,0), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)
out.ma12.arima=arima(y,order=c(0,1,12), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)
out.arma=arima(y,order=c(1,1,1), xreg= cbind(time, time^2, time^3, time^4), include.mean=TRUE)
AIC(out.ar4, out.ar6, out.ma12, out.arma)
BIC(out.ar4, out.ar6, out.ma12, out.arma)

yfcst.ma12.arima=predict(out.ma12.arima,n.ahead=36, newxreg=cbind(time2, time2^2, time2^3, time2^4))
yfcst.ma12.arima=yfcst.ma12.arima$pred
yfcst.ma12.arima

three=ts.union(y, yfcst.ma12, yfcst.ma12.arima)
plot(three[,1], ylim=c(min(y),max(yfcst.ma12.arima)))
lines(three[,2], col=2)
lines(three[,3], col=3)
legend("bottomright", legend=c("actual", "forecast-MA(12)","forecast-ARIMA(0,1,12)"), col=1:3, lty=1)