insurance=read.csv("desktop/WCAS.ECONOMETRICS/insurance_training_data.csv", na.strings=c(""," ","NA"))
head(insurance)
library(psych)
describe(insurance, check=T, omit = T)
na.cols <- which(colSums(is.na(insurance)) > 0)
sort(colSums(sapply(insurance[na.cols], is.na)), decreasing = TRUE)
hist(insurance$TARGET_AMT)
hist(insurance$KIDSDRIV)
hist(insurance$AGE)
hist(insurance$HOMEKIDS)
hist(insurance$YOJ)
hist(insurance$TRA)
hist(insurance$TIF)
hist(insurance$CLM_FREQ)
hist(insurance$MVR_PTS)
hist(insurance$INCOME)
hist(insurance$HOME_VAL)

require(ggplot2)
plot.categoric <- function(cols, df){
  for (col in cols) {
    order.cols <- names(sort(table(insurance[,col]), decreasing = TRUE))
    
    num.plot <- qplot(insurance[,col]) +
      geom_bar(fill = 'red') +
      geom_text(aes(label = ..count..), stat='count', vjust=-0.5) +
      theme_minimal() +
      scale_y_continuous(limits = c(0,max(table(df[,col]))*1.1)) +
      scale_x_discrete(limits = order.cols) +
      xlab(col) +
      theme(axis.text.x = element_text(angle = 30, size=12))
    
    print(num.plot)
  }
}
plot.categoric('JOB', insurance)
insurance$CAR_AGE=ifelse(is.na(insurance$CAR_AGE),mean(insurance$CAR_AGE,na.rm=T),insurance$CAR_AGE)
insurance$YOJ=ifelse(is.na(insurance$YOJ),median(insurance$YOJ,na.rm=T),insurance$YOJ)
insurance$HOME_VAL=ifelse(is.na(insurance$HOME_VAL),mean(insurance$HOME_VAL,na.rm=T),insurance$HOME_VAL)
insurance$AGE=ifelse(is.na(insurance$AGE),mean(insurance$AGE,na.rm=T),insurance$AGE)
insurance$INCOME=ifelse(is.na(insurance$INCOME),mean(insurance$INCOME, na.rm=T),insurance$INCOME)
insurance$JOB[is.na(insurance$JOB)] <- "z_Blue Collar"
describe(insurance)
insurance["bluebook"] = as.numeric(insurance$BLUEBOOK)
insurance["old.claim"] = as.numeric(insurance$OLDCLAIM)
head(insurance)
insurance2=insurance[,-c(1,2,17,21)]
head(insurance2)
insur.subset <- insurance2[ which( insurance2$TARGET_AMT > 0), ]
describe(insur.subset)
lmMod <- lm(insur.subset$TARGET_AMT ~. , data = insur.subset)
lmMod
summary(lmMod)
anova(lmMod)
lmMod$coefficients
hist(lmMod$residuals)
plot(lmMod$residuals)
qqnorm(lmMod$residuals)
qqline(lmMod$residuals)
skew(lmMod$residuals)
require(MASS)
lmMod2 <- lm(log(insur.subset$TARGET_AMT) ~KIDSDRIV+log(AGE)+ EDUCATION + HOMEKIDS+ sqrt(YOJ)+ log(INCOME)+ PARENT1+ sqrt(HOME_VAL)+MSTATUS+SEX+JOB+log(TRAVTIME)+CAR_USE+TIF+CAR_TYPE+CLM_FREQ+REVOKED+CAR_AGE+sqrt(insur.subset$bluebook)+sqrt(old.claim), data=insur.subset)
summary(lmMod2)
anova(lmMod2)
hist(lmMod2$residuals)
plot(lmMod2$residuals)
qqnorm(lmMod2$residuals)
qqline(lmMod2$residuals)
summary(lmMod2)
lmMod3=step(lmMod)          
summary(lmMod3)  
lmMod4=step(lmMod2)
summary(lmMod4)           
insurance3=insurance[,-c(1,3,17,21)]          
logit1= glm(insurance3$TARGET_FLAG ~. , data = insurance3)          
summary(logit1)  
require(regclass)
logit2= glm(insurance3$TARGET_FLAG~ KIDSDRIV+ YOJ + PARENT1+MSTATUS+SEX+EDUCATION+JOB+TRAVTIME+CAR_USE+TIF+CAR_TYPE+CLM_FREQ+MVR_PTS+URBANICITY+ old.claim, data = insurance3)      
summary(logit2)   
logit3= step(glm(insurance3$TARGET_FLAG ~. , data = insurance3)) 
summary(logit3)
plot(lmMod3$residuals, main="lmMod3")
plot(lmMod4$residuals, main="lmMod4")
hist(lmMod3$residuals)
hist(lmMod4$residuals)

cbind(AIC(lmMod3,lmMod4))
cbind(BIC(lmMod3,lmMod4))
test=read.csv("desktop/WCAS.ECONOMETRICS/test.insurance.csv", na.strings=c(""," ","NA"))
test$CAR_AGE=ifelse(is.na(test$CAR_AGE),mean(test$CAR_AGE,na.rm=T),test$CAR_AGE)
test$YOJ=ifelse(is.na(test$YOJ),median(test$YOJ,na.rm=T),test$YOJ)
test$HOME_VAL=ifelse(is.na(test$HOME_VAL),mean(test$HOME_VAL,na.rm=T),test$HOME_VAL)
test$AGE=ifelse(is.na(test$AGE),mean(test$AGE,na.rm=T),test$AGE)
test$INCOME=ifelse(is.na(test$INCOME),mean(test$INCOME, na.rm=T),test$INCOME)
test$JOB[is.na(test$JOB)] <- "z_Blue Collar"
describe(test)
test["bluebook"] = as.numeric(test$BLUEBOOK)
test["old.claim"] = as.numeric(test$OLDCLAIM)
head(test)
test2=test[,-c(1,2,3,17,21)] 
pred.lmMod3=predict(lmMod3, test2)
require(Metrics)
pred.lmMod4=predict(lmMod4, test2)
rmse(insur.subset$TARGET_AMT[1:2141], pred.lmMod3 )
rmse(log(insur.subset$TARGET_AMT[1:2141]), pred.lmMod4) 

cbind(AIC(logit1,logit2, logit3))
summary(logit1$fitted.values)
summary(logit2$fitted.values)
summary(logit3$fitted.values)
hist(logit1$fitted.values,main = " Histogram ", col = 'light green')
hist(logit2$fitted.values,main = " Histogram ", col = 'light green')
hist(logit3$fitted.values,main = " Histogram ", col = 'light green')
pred.logit1=predict(logit1, test2)
pred.logit2=predict(logit2, test2)
pred.logit3=predict(logit3, test2)
insurance3$TARGET_FLAG= ifelse(insurance3$TARGET_FLAG$ >= 0.5,"no.crash","crash")
mytable <- table(insurance3$TARGET_FLAG,pred.logit1)
rownames(mytable) <- c("Obs. neg","Obs. pos")
colnames(mytable) <- c("Pred. neg","Pred. pos")
install.packages("pROC")
require(pROC)
roc(insurance3$TARGET_FLAG[1:2141]~pred.logit1, data = insurance3, plot = TRUE, main = "ROC CURVE", col= "blue")
roc(insurance3$TARGET_FLAG[1:2141]~pred.logit2, data = insurance3, plot = TRUE, main = "ROC CURVE", col= "blue")
roc(insurance3$TARGET_FLAG[1:2141]~pred.logit3, data = insurance3, plot = TRUE, main = "ROC CURVE", col= "blue")


write.csv(pred.lmMod4,file= "prediction.model4.csv", row.names = F)
write.csv(pred.logit2,file= "prediction.logit2.csv", row.names = F)
