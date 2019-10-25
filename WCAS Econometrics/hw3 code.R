wine=read.csv("desktop/WCAS.ECONOMETRICS/wine.training.data.csv", na.strings=c(""," ","NA", "n.a.", "-", "--"))
head(wine)
wine.test=read.csv("desktop/WCAS.ECONOMETRICS/wine.test.data.csv", na.strings=c(""," ","NA", "n.a.", "-", "--"))
na.cols <- which(colSums(is.na(wine)) > 0)
sort(colSums(sapply(wine[na.cols], is.na)), decreasing = TRUE)
hist(wine$TARGET)
hist(wine$FixedAcidity)
hist(wine$VolatileAcidity)
hist(wine$CitricAcid)
hist(wine$ResidualSugar)
hist(wine$Chlorides)
hist(wine$FreeSulfurDioxide)
hist(wine$TotalSulfurDioxide)
hist(wine$Density)
hist(wine$pH)
hist(wine$Sulphates)
hist(wine$Alcohol)
hist(wine$LabelAppeal)
hist(wine$AcidIndex)
hist(wine$STARS)
require(psych)
describe(wine)
require(corrplot)
corrplot(wine)
wine$STARS=ifelse(is.na(wine$STARS),2,wine$STARS)
wine$Sulphates=ifelse(is.na(wine$Sulphates),mean(wine$Sulphates, na.rm = T),wine$Sulphates)
wine$TotalSulfurDioxide=ifelse(is.na(wine$TotalSulfurDioxide),mean(wine$TotalSulfurDioxide, na.rm = T),wine$TotalSulfurDioxide)
wine$Alcohol=ifelse(is.na(wine$Alcohol),mean(wine$Alcohol, na.rm = T),wine$Alcohol)
wine$FreeSulfurDioxide=ifelse(is.na(wine$FreeSulfurDioxide),mean(wine$FreeSulfurDioxide, na.rm = T),wine$FreeSulfurDioxide)
wine$Chlorides=ifelse(is.na(wine$Chlorides),mean(wine$Chlorides, na.rm = T),wine$Chlorides)
wine$ResidualSugar=ifelse(is.na(wine$ResidualSugar),mean(wine$ResidualSugar, na.rm = T),wine$ResidualSugar)
wine$pH=ifelse(is.na(wine$pH),mean(wine$pH, na.rm = T),wine$pH)
wine=wine[,-1]
head(wine)
cor(wine)
wine$STARS[wine$STARS == 4] <- "Great"
wine$STARS[wine$STARS == 3] <- "Good"
wine$STARS[wine$STARS == 2] <- "Fair"
wine$STARS[wine$STARS == 1] <- "Poor"
head(wine.subset)
wine$LabelAppeal[wine$LabelAppeal == 2] <- "Exeptional"
wine$LabelAppeal[wine$LabelAppeal == 1] <- "Great"
wine$LabelAppeal[wine$LabelAppeal == 0] <- "Good"
wine$LabelAppeal[wine$LabelAppeal == -1] <- "Fair"
wine$LabelAppeal[wine$LabelAppeal == -2] <- "Poor"
wine.subset <- wine[which( wine$TARGET > 0), ]
describe(wine.subset)
mod1 <- lm(wine$TARGET ~. , data = wine)
summary(mod1)
hist(mod1$residuals)
plot(mod1$residuals)
qqnorm(mod1$residuals)
qqline(mod1$residuals)
skew(mod1$residuals)
require(MASS)


mod2=lm(wine$TARGET~wine$VolatileAcidity+wine$Chlorides+wine$FreeSulfurDioxide+wine$TotalSulfurDioxide+wine$pH+wine$Sulphates+wine$Alcohol+wine$AcidIndex+wine$LabelAppeal+wine$STARS)
summary(mod2)
hist(mod2$residuals)
plot(mod2$residuals)
qqnorm(mod2$residuals)
qqline(mod2$residuals)
skew(mod2$residuals)
summary(mod2)
sum(wine$TARGET==0)
mod3=glm(wine$TARGET~ wine$VolatileAcidity+wine$FreeSulfurDioxide+wine$Sulphates+wine$AcidIndex+wine$LabelAppeal+wine$STARS, family="poisson")
summary(mod3)
hist(mod3$residuals)
plot(mod3$residuals)
qqnorm(mod3$residuals)
qqline(mod3$residuals)
skew(mod3$residuals)
install.packages("pscl")
require(pscl)
help(zeroinfl)
mod4= zeroinfl(wine$TARGET~ wine$VolatileAcidity+wine$FreeSulfurDioxide+wine$pH+wine$AcidIndex+wine$LabelAppeal+wine$STARS, data=wine, dist = 'poisson', link = 'log')

summary(mod4)
hist(mod4$residuals)
plot(mod4$residuals)
qqnorm(mod4$residuals)
qqline(mod4$residuals)
skew(mod4$residuals)
hist(target)
mod5=glm.nb(wine$TARGET~wine$Chlorides+wine$TotalSulfurDioxide+wine$pH+wine$Alcohol+wine$LabelAppeal+wine$STARS)
summary(mod5)
hist(mod5$residuals)
plot(mod5$residuals)
qqnorm(mod5$residuals)
qqline(mod5$residuals)
skew(mod5$residuals)
mod6= zeroinfl(wine$TARGET~wine$Chlorides+wine$TotalSulfurDioxide+wine$Sulphates+wine$Alcohol+wine$LabelAppeal+wine$STARS, data=wine, dist = 'negbin')
summary(mod6)
hist(mod6$residuals)
plot(mod6$residuals)
qqnorm(mod6$residuals)
qqline(mod6$residuals)
skew(mod6$residuals)


mod7=zeroinfl(wine$TARGET~wine$Chlorides + wine$TotalSulfurDioxide+wine$Sulphates+wine$Alcohol+wine$Sulphates+wine$LabelAppeal+wine$STARS, data=wine, dist = 'negbin')
summary(mod7)
dispersiontest(mod3)
plot(mod7$residuals)
qqnorm(mod7$residuals)
qqline(mod7$residuals)
skew(mod7$residuals)
cbind(AIC(mod2, mod6, mod7))


wine.test$STARS=ifelse(is.na(wine.test$STARS),2,wine.test$STARS)
wine.test$Sulphates=ifelse(is.na(wine.test$Sulphates),mean(wine.test$Sulphates, na.rm = T),wine.test$Sulphates)
wine.test$TotalSulfurDioxide=ifelse(is.na(wine.test$TotalSulfurDioxide),mean(wine.test$TotalSulfurDioxide, na.rm = T),wine.test$TotalSulfurDioxide)
wine.test$Alcohol=ifelse(is.na(wine.test$Alcohol),mean(wine.test$Alcohol, na.rm = T),wine.test$Alcohol)
wine.test$FreeSulfurDioxide=ifelse(is.na(wine.test$FreeSulfurDioxide),mean(wine.test$FreeSulfurDioxide, na.rm = T),wine.test$FreeSulfurDioxide)
wine.test$Chlorides=ifelse(is.na(wine.test$Chlorides),mean(wine.test$Chlorides, na.rm = T),wine.test$Chlorides)
wine.test$ResidualSugar=ifelse(is.na(wine.test$ResidualSugar),mean(wine.test$ResidualSugar, na.rm = T),wine.test$ResidualSugar)
wine.test$pH=ifelse(is.na(wine.test$pH),mean(wine.test$pH, na.rm = T),wine.test$pH)
head(wine.test)
wine.test=wine.test[,-c(1,2)]
head(wine)
cor(wine)
wine.test$STARS[wine.test$STARS == 4] <- "Great"
wine.test$STARS[wine.test$STARS == 3] <- "Good"
wine.test$STARS[wine.test$STARS == 2] <- "Fair"
wine.test$STARS[wine.test$STARS == 1] <- "Poor"

wine.test$LabelAppeal[wine.test$LabelAppeal == 2] <- "Exeptional"
wine.test$LabelAppeal[wine.test$LabelAppeal == 1] <- "Great"
wine.test$LabelAppeal[wine.test$LabelAppeal == 0] <- "Good"
wine.test$LabelAppeal[wine.test$LabelAppeal == -1] <- "Fair"
wine.test$LabelAppeal[wine.test$LabelAppeal == -2] <- "Poor"

na.cols <- which(colSums(is.na(wine.test)) > 0)
sort(colSums(sapply(wine.test[na.cols], is.na)), decreasing = TRUE)
length(wine.test$TARGET)

pred.mod2=predict(mod2, wine.test)
pred.mod6=predict(mod6, wine.test)
pred.mod7= predict(mod7, wine.test)
head(wine)


pred.mod2
head(wine.test)

rmse(wine$TARGET[1:3335], pred.mod2)
rmse(wine$TARGET[1:3335], pred.mod6)
rmse(wine$TARGET[1:3335], pred.mod7)
write.csv(pred.mod7,file= "prediction.model7.csv", row.names = F)



