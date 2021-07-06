##### LOAD DATA, DESCRIPTIVE STATS ####

library(forecast)
library(dplyr)
library(seasonal)
library(ggplot2)
library(tidyr)
library(stringr)
library(lubridate)
library(scales)
library(psych)
library(gt)
library(kableExtra)

setwd("C:/Users/jawilliams/Desktop/Forecasting/Data/Midterm Data")


air_reserve <- read.csv("air_reserve.csv", stringsAsFactors = F)

air_store_info <- read.csv("air_store_info.csv", stringsAsFactors = F)

air_visit_data <- read.csv("air_visit_data.csv", stringsAsFactors = F)

date_info <- read.csv("date_info.csv", stringsAsFactors = F)

air_reserve %>% 
  select(reserve_visitors) %>% 
  describe() %>% 
  as_tibble() %>% 
  gt() %>% 
  tab_header(title = "Expected Visitors From Reservations Descriptive Statitsics")

air_visit_data %>% 
  select(visitors) %>% 
  describe() %>% 
  as_tibble() %>% 
  gt() %>% 
  tab_header(title = "Observed Visitors Descriptive Statitsics")

air_reserve_sums <- air_reserve %>% 
  mutate(visit_datetime = as.Date(visit_datetime)) %>% 
  group_by(visit_datetime) %>% 
  summarise(Visits = sum(reserve_visitors)) %>% 
  rename(Date = visit_datetime)

air_visit_sums <- air_visit_data %>% 
  mutate(visit_date = as.Date(visit_date)) %>% 
  group_by(visit_date) %>% 
  summarise(Visits = sum(visitors)) %>% 
  rename(Date = visit_date)


print(
  paste(
    "The expected visitors from reservations series begins on",
    head(air_reserve_sums$Date, 1),
    "and ends on",
    tail(air_reserve_sums$Date, 1)
  )
)

print(paste(
  "The observed visitors series begins on",
  head(air_visit_sums$Date, 1),
  "and ends on",
  tail(air_visit_sums$Date, 1)
))

print(
  paste(
    "There are",
    length(unique(air_reserve$air_store_id)),
    "unique firms in the expected visitors from reservations series"
  )
)

print(paste(
  "There are",
  length(unique(air_visit_data$air_store_id)),
  "unique firms in the observed visitors series"
))

##### PLOTS #####

reserve_firms <- unique(air_reserve$air_store_id) 

air_reserve_sums_small <- 
  air_reserve %>%
  mutate(visit_datetime = as.Date(visit_datetime)) %>% 
  filter(visit_datetime <= "2017-04-22") %>% 
  select(-reserve_datetime)%>% 
  group_by(visit_datetime) %>% 
  summarise(Visits = sum(reserve_visitors)) %>% 
  rename(Date = visit_datetime)

air_reserve_dates <- unique(air_reserve_sums_small$Date)

air_visit_reserve <- air_visit_data %>%
  mutate(visit_date = as.Date(visit_date)) %>%
  filter(air_store_id %in% reserve_firms) %>%
  group_by(visit_date) %>%
  summarise(Visits = sum(visitors)) %>%
  rename(Date = visit_date) %>%
  filter(Date %in% air_reserve_dates) %>%
  mutate(Reservations = air_reserve_sums_small$Visits)

air_visit_reserve %>%
  pivot_longer(-Date, names_to = "Interaction", values_to = "Visitors") %>%
  ggplot(., aes(y = Visitors, x = Interaction, fill = Interaction)) +
  geom_col(stat = "identity") +
  ggtitle("Number of Expected Visitors From Reservations Versus Observed Visitors")

air_visit_series <- air_visit_data %>% 
  mutate(visit_date = as.Date(visit_date)) %>% 
  filter(air_store_id %in% reserve_firms) %>%  
  group_by(visit_date) %>% 
  summarise(Visits = sum(visitors)) %>% 
  rename(Date = visit_date)

air_visit_series %>% 
  ggplot(., aes(x = Date , y = Visits)) +
  geom_line() +
  xlab("Time") +
  ylab("# of Visitors") +
  ggtitle("Visitors From 2016-01-01 to 2017-04-22")

air_visit_series %>%
  mutate(wday = wday(Date, label = TRUE, week_start = 1)) %>%
  group_by(wday) %>%
  summarise(AVG_visits = round(mean(Visits), 2),
            SD = round(sd(Visits), 2),
            N= n()) %>% 
  mutate(se = SD / sqrt(N),
         lower = AVG_visits - qt(1 - (0.05 / 2), N - 1) * se, 
         upper = AVG_visits + qt(1 - (0.05 / 2), N - 1) * se) %>% 
  ggplot(aes(wday, AVG_visits, fill = wday)) +
  geom_col() +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.35)+
  theme(legend.position = "none", axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  labs(x = "Day of the week", y = "Mean visitors", title = "Mean # Of Visitors By Week Day") +
  scale_fill_hue()

air_visit_series %>%
  mutate(month = month(Date, label = TRUE)) %>%
  group_by(month) %>%
  summarise(AVG_visits = round(mean(Visits), 2),
            SD = round(sd(Visits), 2),
            N= n()) %>% 
  mutate(se = SD / sqrt(N),
         lower = AVG_visits - qt(1 - (0.05 / 2), N - 1) * se, 
         upper = AVG_visits + qt(1 - (0.05 / 2), N - 1) * se) %>% 
  ggplot(aes(month, AVG_visits, fill = month)) +
  geom_col() +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.35)+
  theme(legend.position = "none") +
  labs(x = "Month", y = "Median visitors", title = "Mean # Of Visitors By Month")+
  scale_fill_hue()


holidays <- date_info %>% 
  mutate(calendar_date = as.Date(calendar_date)) %>% 
  filter(holiday_flg == 1) %>% 
  rename(Date = calendar_date)

holiday_vists <- air_visit_series %>% 
  filter(Date %in% holidays$Date)


holidays <- holidays %>%
  filter(Date %in% holiday_vists$Date) %>% 
  mutate(Visitors = holiday_vists$Visits)

holidays %>% 
  rename(Weekday = day_of_week) %>% 
  mutate(Weekday = wday(Date, label = TRUE,  week_start = 1)) %>% 
  ggplot(.,aes(x = Date, y = Visitors, color = Weekday))+
  geom_point()+
  geom_hline(yintercept = mean(air_visit_series$Visits), linetype = "dashed", color = "red")+
  labs(title = "Number of Visitors On Holidays")


##### MODELS #####

## MAke TS

visits_ts <- ts(air_visit_series$Visits, 
                start = c(2016,1), 
                end = c(2017, 
                        as.numeric(
                          strftime(
                            as.POSIXct(
                              tail(
                                air_visit_series$Date, 1)), "%j"))),
                frequency = 365)



## Train and test splits

train <- window(visits_ts, end = c(2017, 
                                   as.numeric(
                                     strftime(as.POSIXct(
                                       "2017-02-28"), "%j"))))

test <- window(visits_ts, start = c(2017, 
                                    as.numeric(
                                      strftime(as.POSIXct(
                                        "2017-03-01"), "%j"))))
## Iterate over models to find K W/ Lowest AIC
memory.size(10000000000000)

fourier_arima <- lapply(seq_along(1:25),
                        function(x){
                          
                          aic_fourier_arima <- 
                            AIC(
                              auto.arima(train, 
                                         seasonal=FALSE, 
                                         xreg=fourier(train, K=x))
                            )
                          
                        }) ### Min AIC K = 19

## Fit Models

fit_arima <- auto.arima(train)

fit_arima_fourier <- auto.arima(train, 
                                seasonal = FALSE, 
                                xreg=fourier(train, K = 19))


fit_TBATS <- tbats(train)

## fcst length of test

arima_fcsts <- forecast(fit_arima, h = length(test))

arima_fourier_fcsts <- forecast(fit_arima_fourier, 
                                xreg = fourier(train, K=19, length(test)),
                                h = length(test))

TBATS_fcsts <- forecast(fit_TBATS, h = length(test))

gt(
  as_tibble(accuracy(arima_fcsts, test))
) %>% 
  tab_header("ARIMA(5,1,3) Model Selection Metrics")

gt(
  as_tibble(accuracy(arima_fourier_fcsts, test))
) %>% 
  tab_header("Fourier ARIMA(5,1,3), K=19 Model Selection Metrics")

gt(
  as_tibble(accuracy(arima_fourier_fcsts, test))
) %>% 
  tab_header("BATS (0.187, {5,3}) Model Selection Metrics")

# Auto ARIMA is best

autoplot(visits_ts) +
  autolayer(arima_fcsts, series = "Forecasts", PI = F)+
  ggtitle("Training Model Forecast ARIMA(5,1,3)") + 
  ylab("Visitors")

cbind('Residuals' = residuals(fit_arima),
      'Forecast errors' = residuals(fit_arima,type='response')) %>%
  autoplot(facet=TRUE) + xlab("Year") + ylab("") + 
  ggtitle("Residuals and Forecast Errors From ARIMA(5,1,3)")

hist(fit_arima$residuals, col = "grey", main = "ARIMA(5,1,3)",
     xlab = "Visitors")

fit_arima <- auto.arima(visits_ts)

arima_fcsts <- forecast(fit_arima, h = 39)

autoplot(visits_ts) +
  autolayer(arima_fcsts, series = "Forecasts", PI = F)+
  ggtitle("Out of Sample Forecasts 2017-04-23 to 2017-05-31 ARIMA(5,1,3)") + 
  ylab("Visitors")

air_visit_data <- air_visit_data %>% 
  mutate(visit_date = as.Date(visit_date))

firm_ids <- unique(air_visit_data$air_store_id)

#Iterate through firms to get ARIMAS

forecast_list <- lapply(setNames(unique(firm_ids), unique(firm_ids)),
                        function(x) {
                          my_data <- air_visit_data %>%
                            filter(air_store_id == x)
                          
                          my_ts <- ts(
                            my_data$visitors,
                            start = c(2016, 1),
                            end = c(2017,
                                    as.numeric(strftime(
                                      as.POSIXct("2017-04-23"), "%j"
                                    ))),
                            frequency = 365
                          )
                          
                          fcst_dates <- seq(as.Date("2017-04-23"), as.Date("2017-05-31"), by = "days")
                          
                          myArima <- auto.arima(my_ts)
                          
                          my_fcts <- forecast(myArima, 
                                              h = length(fcst_dates))
                          df <- data.frame("id" = 
                                             paste(x, fcst_dates,
                                                   sep = "_"),
                                           "visitors" = 
                                             as.numeric(my_fcts$mean))
                          
                          return(df)
                          
                          
                        })

df_final <- do.call(rbind, forecast_list)

df_final$visitors <- round(df_final$visitors, 0)


setwd("C:/Users/jawilliams/Desktop/Forecasting/Assignments")

write.csv(df_final, "Midterm Forecasts.csv")
