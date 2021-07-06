library(lubridate)
library(fpp2)
library(dplyr)
library(tidyr)
library(seasonal)
library(urca)

df <- read.csv("C:/Users/jawilliams/Desktop/Forecasting/Data/station_rio_temp.csv", stringsAsFactors = F)

df <- df %>% 
  pivot_longer(., -YEAR, names_to = "Month", values_to = "Temp")

for (i in 1:nrow(df)){
  if(df$Temp[i]==999.90){
    df$Temp[i]=df$Temp[i-1]
  }
  
}

autoplot(temp)+
  ylab("Celcius")+
  ggtitle("Avgerage Monthly Temperature: Rio 1973 - 2020")

autoplot(decompose(temp, type = 'a'))+
  ggtitle("Additive Decomposition")

autoplot(decompose(temp, type = 'm'))+
  ggtitle("Multiplicative Decomposition")


acf(diff(temp))
pacf(diff)
