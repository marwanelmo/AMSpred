library('tidyverse')

data <- read.csv("C:/Users/melmora/Documents/GitHub/Oprah - AMS/export/processeddata.csv", sep=";")


ggplot(data, aes(x=T0_age)) + geom_histogram() + 
  geom_vline(aes(xintercept=mean(data$T0_age)), size=1, linetype='dashed', color='blue')+
  theme_bw()+ xlab("Leeftijd") + ylab("Aantal mensen") + ggtitle("Verdeling van leeftijd")


ggplot(data, aes(x=T0_age)) + geom_histogram() + 
  facet_grid(pat_sexe ~ .) + 
  theme_bw()+ xlab("Leeftijd") + ylab("Aantal mensen") + ggtitle("Verdeling van leeftijd")
  