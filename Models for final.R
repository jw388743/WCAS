model=lm(StringChanges_train$Levenshtein_Distance~ StringChanges_train$Fuzzy_Match_Score+StringChanges_train$Workflow_Step_Name+ StringChanges_train$Target_Locale_ID+ StringChanges_train$String_Word_Count+StringChanges_train$Translation_Character_Count+ StringChanges_train$Translation_Updated_Percent+StringChanges_train$Content_Type)
summary(model)
plot(model$residuals) #conistent varience, most values on 0. bad outliers
hist(model$resid)#major left skew
skew(model$residuals)
qqnorm(model$residuals)
qqline(model$residuals)#shape is really bad


model.beta=lm(StringChanges_train$Levenshtein_Distance~ + StringChanges_train$Target_Locale_ID+ StringChanges_train$String_Word_Count+StringChanges_train$Translation_Character_Count+ StringChanges_train$Translation_Updated_Percent)
summary(model.beta)
plot(model.beta$residuals) 
hist(model.beta$resid)
skew(model.beta$residuals)
qqnorm(model$residuals)
qqline(model$residuals)#shape is really bad







hist(StringChanges_train$Levenshtein_Distance)
hist(log(StringChanges_train$Levenshtein_Distance))#much more ideal distribution 

#skew of Lev. Distence is screwing up models. 



log.linear.mod=lm(log(StringChanges_train$Levenshtein_Distance)~ StringChanges_train$Fuzzy_Match_Score+StringChanges_train$Workflow_Step_Name+ StringChanges_train$Target_Locale_ID+ StringChanges_train$String_Word_Count+StringChanges_train$Translation_Character_Count+ StringChanges_train$Translation_Updated_Percent+StringChanges_train$Content_Type)
summary(log.linear.mod)
plot(log.linear.mod$residuals) #consistent varience, less outliers, but several are more extreme
hist(log.linear.mod$resid)#still skewed
skew(model$residuals)
qqnorm(log.linear.mod$residuals)
qqline(log.linear.mod$residuals)#shape is much better
#objectivley better model, Decrease in Adj, R squared, but better distribution of residuals/qqline

require(MASS)
bcox= boxcox(StringChanges_train$Levenshtein_Distance~ StringChanges_train$Fuzzy_Match_Score+StringChanges_train$Workflow_Step_Name+ StringChanges_train$Target_Locale_ID+ StringChanges_train$String_Word_Count+StringChanges_train$Translation_Character_Count+ StringChanges_train$Translation_Updated_Percent+StringChanges_train$Content_Type)
x=bcox$x
y=bcox$y
bc=cbind(x,y)
bc[order(-y),]#log best transformation 

#I was going to make a log-log model, but many of the numeric explanatory variables have 0s, so we cannot. 

step.log.linear.model= step(log.linear.mod)
summary(step.log.linear.model)#same as original


step.linear=step(model)
summary(step.linear)#same as original 

log.linear.mod2= log.linear.mod=lm(log(StringChanges_train$Levenshtein_Distance)~ StringChanges_train$Fuzzy_Match_Score+ StringChanges_train$Target_Locale_ID+ StringChanges_train$String_Word_Count+StringChanges_train$Translation_Character_Count+ StringChanges_train$Translation_Updated_Percent+StringChanges_train$Content_Type)
summary(log.linear.mod2)
plot(log.linear.mod2$residuals) #consistent varience, less outliers, and extreme outliers. 
hist(log.linear.mod2$resid)#still skewed
skew(log.linear.mod$residuals)
#best model in my opinion, some negative skew but much less than the others. Basically same results as log.linear.mod 


cbind(AIC(model, model.beta, log.linear.mod, log.linear.mod2))
cbind(BIC(model, model.beta, log.linear.mod, log.linear.mod2))
require(regclass)
all_vifs1 <- VIF(model)
print(all_vifs1)

all_vifs2 <- VIF(log.linear.mod2)
print(all_vifs2)

all_vifs3 <- VIF(model.beta)
print(all_vifs3) #word/charecter count are highly correlated, lets see what the model looks like without one of them. I elect to remove word count. 



log.linear.mod3= log.linear.mod=lm(log(StringChanges_train$Levenshtein_Distance)~ StringChanges_train$Fuzzy_Match_Score+ StringChanges_train$Target_Locale_ID+StringChanges_train$Translation_Character_Count+ StringChanges_train$Translation_Updated_Percent+StringChanges_train$Content_Type)
summary(log.linear.mod3)
plot(log.linear.mod3$residuals) #consistent varience, less outliers, and extreme outliers. 
hist(log.linear.mod3$resid)
skew(log.linear.mod3$residuals) #lowest thus far

qqnorm(log.linear.mod3$residuals)
qqline(log.linear.mod3$residuals) #best shape

all_vifs4 <- VIF(log.linear.mod3)
print(all_vifs4)#no collinierty 

cbind(AIC(model, model.beta, log.linear.mod, log.linear.mod2, log.linear.mod3))
cbind(BIC(model, model.beta, log.linear.mod, log.linear.mod2, log.linear.mod3))

na.cols <- which(colSums(is.na(StringChanges_test)) > 0)
sort(colSums(sapply(StringChanges_test[na.cols], is.na)), decreasing = TRUE) #none of the varibales we are concerned with. 

head(StringChanges_test)
predict1= predict(log.linear.mod2, StringChanges_test)
predict1
exp(predict1)
length(predict1)
length(StringChanges_train$Levenshtein_Distance)

predict2= predict(model, StringChanges_test)
predict3= predict(model.beta, StringChanges_test)
predict4=predict(log.linear.mod3, StringChanges_test)


require(Metrics)



rmse(log(StringChanges_train$Levenshtein_Distance), predict1)
rmse(log(StringChanges_train$Levenshtein_Distance), predict2)
rmse(log(StringChanges_train$Levenshtein_Distance), predict3)
rmse(log(StringChanges_train$Levenshtein_Distance), predict4)



