#### QUESTION 6
df <-
  read.csv("/Users/james/desktop/be_bc_f19/ps1/sutterexperiment.csv")

names(df) <- tolower(names(df))

names(df)
#### Question 7

df <-
  df %>%
  mutate(uniquesubject = paste(session, treatment, subject, sep = "_"))


#### Q 8

df <-
  df %>%
  mutate(uniqueteam = paste(session, team, sep = "_"))

df$uniqueteam <-
  if_else(df$treatment == "INDIVIDUALS", df$uniquesubject, df$uniqueteam)

#### Q 9

df_narrow <- df %>%
  gather(rounds, investment, r1:r9) %>%
  arrange(session, subject, treatment, team)

#### Q 10

df_narrow$rounds <- gsub("R", " ", df_narrow$rounds)
head(df_narrow)

#### Q 11

df_narrow %>%
  select(subject, team, investment, treatment) %>% 
  group_by(treatment) %>%
  summarise_all(funs(mean, sd))

#### Q 12

df_narrow %>%
  group_by(treatment) %>%
  ggplot(., aes(treatment, investment, fill = treatment)) +
  stat_summary(fun.y = mean, geom = "bar", fill = "gray69") +
  stat_summary(fun.data = mean_se,
               geom = "errorbar",
               width = 0.1) +
  labs(
    y = "Average Investment",
    x = "Treatment",
    title = 'Avg Investment by Treatment',
    subtitle = 'Error Bar= Standard Errors'
  ) +
  theme_classic()

#### Q 13

teams_vs_indiv <-
  df_narrow %>%
  filter(treatment == c('INDIVIDUALS', 'TEAMS')) %>%
  group_by(treatment, rounds) %>%
  summarize(average = mean(investment))

teams_vs_indiv




p1 <-
  ggplot(teams_vs_indiv, aes(x = rounds, y = average, group = treatment)) +
  geom_line(aes(color = treatment)) +
  geom_point(aes(shape = treatment, color = treatment), size = 4) +
  scale_shape_manual(values = c(17, 19)) +
  ylim(20, 100) +
  theme_classic() +
  theme(legend.position = "top")
p1

#### Q 14

paycomm_vs_message <-
  df_narrow %>%
  filter(treatment == c('MESSAGE', 'PAY-COMM')) %>%
  group_by(treatment, rounds) %>%
  summarize(average = mean(investment))
paycomm_vs_message

individual <-
  df_narrow %>%
  filter(treatment == c('INDIVIDUALS')) %>%
  group_by(treatment, rounds) %>%
  summarize(average = mean(investment))

paycomm_vs_message_vs_indiv <- rbind(paycomm_vs_message, individual)
paycomm_vs_message_vs_indiv

p2 <- ggplot(paycomm_vs_message_vs_indiv,
             aes(x = rounds, y = average, group = treatment)) +
  geom_line(aes(color = treatment)) +
  geom_point(aes(shape = treatment, color = treatment), size = 4) +
  scale_shape_manual(values = c(17, 15, 19)) +
  ylim(20, 100) +
  theme_classic()
theme(legend.position = "top")

p2

### Q 15

mixed_vs_indiv <-
  df_narrow %>%
  filter(treatment == c('INDIVIDUALS', 'MIXED')) %>%
  group_by(treatment, rounds) %>%
  summarize(average = mean(investment))
mixed_vs_indiv




p3 <-
  ggplot(mixed_vs_indiv, aes(x = rounds, y = average, group = treatment)) +
  geom_line(aes(linetype = treatment, color = treatment)) +
  geom_point(aes(shape = treatment, color = treatment), size = 4) +
  scale_shape_manual(values = c(17, 19)) +
  ylim(20, 100) +
  theme_classic() +
  theme(legend.position = "top")
p3

#### Q 16

require(cowplot)
cowplot::plot_grid(p1, p2, p3, labels = "auto")

##### Q 19

df_team_avg <-
  df_narrow %>%
  group_by(treatment, uniqueteam) %>%
  summarise(mean = mean(investment))
df_team_avg

##### Q 20

df_team_avg_result1 <-
  df_team_avg %>%
  filter(treatment == 'INDIVIDUALS' | treatment == 'TEAMS')

df_team_avg_result1

wilcox.test(df_team_avg_result1$mean ~ df_team_avg_result1$treatment)

##### Q21

df_team_avg_result2a <-
  df_team_avg %>%
  filter(treatment == 'INDIVIDUALS' | treatment == 'PAY-COMM')
df_team_avg_result2a


wilcox.test(
  df_team_avg_result2a$mean ~ df_team_avg_result2a$treatment,
  mu = 0,
  paired = F
)

df_team_avg_result2b <-
  df_team_avg %>%
  filter(treatment == 'PAY-COMM' | treatment == 'MESSAGE')
df_team_avg_result2b


wilcox.test(
  df_team_avg_result2b$mean ~ df_team_avg_result2b$treatment,
  mu = 0,
  paired = F
)

df_team_avg_result2c <-
  df_team_avg %>%
  filter(treatment == 'TEAMS' | treatment == 'MESSAGE')
df_team_avg_result2c


wilcox.test(
  df_team_avg_result2c$mean ~ df_team_avg_result2c$treatment,
  mu = 0,
  paired = F
)

df_team_avg_result2d <-
  df_team_avg %>%
  filter(treatment == 'TEAMS' | treatment == 'PAY-COMM')
df_team_avg_result2d


wilcox.test(
  df_team_avg_result2d$mean ~ df_team_avg_result2d$treatment,
  mu = 0,
  paired = F
)

##### Q22

m <- lm(investment ~ treatment, data = df_narrow)

summary(m)

##### Q24

vcov_subjectid <- sandwich::vcovCL(m, cluster = df_narrow$subject)
lmtest::coeftest(m, vcov_subjectid)









































