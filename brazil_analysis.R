### Analysis of demographic and economic data of Brazil for history research paper.

library(data.table)
library(tidyverse)
library(reshape2)
library(lubridate)
library(scales)
library(skimr)
library(agricolae)
library(ggplot2)
library(ggthemes)
library(ggsci)
library(gghighlight)
library(ggrepel)
library(harrypotter)
library(GGally)
library(patchwork)
#library(gapminder) #data on life expectancy, GDP per capita, and population by country (also gapminder_unfiltered)
#library(plotly)
library(StanHeaders)
library(zoo)
library(prophet)
library(forecast)

## read in various 'gapminder' datasets - https://www.gapminder.org/data/
## general citation example:
## Data can be reused freely but please attribute the original data source (where applicable) and Gapminder.
## Example: FREE DATA FROM WORLD BANK VIA GAPMINDER.ORG, CC-BY LICENSE
## note sources in paren
# population (http://gapm.io/dpop)
pop <- readxl::read_xlsx("/Users/rab04my/Downloads/population_total.xlsx", sheet=1) %>%
  pivot_longer(cols=-country, names_to="year", values_to="population") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%"Brazil",
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# income (http://gapm.io/dgdppc) - income_per_person_gdppercapita_ppp_inflation_adjusted.csv
income <- read_csv("/Users/rab04my/Downloads/income_per_person_gdppercapita_ppp_inflation_adjusted.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="income") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%"Brazil",
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# life expectancy (http://gapm.io/ilex) - life_expectancy_years.csv
le <- read_csv("/Users/rab04my/Downloads/life_expectancy_years.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="life_expectancy") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%"Brazil",
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# gini coefficient (http://gapm.io/ddgini) - gini.csv
gini <- read_csv("/Users/rab04my/Downloads/gini.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="gini_coeff") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%c("Brazil", "Venezuela", "United States", "Denmark"),
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# number of people in poverty (see screenshot) - number_of_people_in_poverty.csv
num_pov <- read_csv("/Users/rab04my/Downloads/number_of_people_in_poverty.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="num_people") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%"Brazil",
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# rural poverty (see screenshot) - rural_poverty_percent_rural_people_below_national_rural.csv
rural_pov <- read_csv("/Users/rab04my/Downloads/rural_poverty_percent_rural_people_below_national_rural.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="perc_people") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%"Brazil",
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# urban poverty (see screenshot) - urban_poverty_percent_urban_people_below_national_urban.csv
urban_pov <- read_csv("/Users/rab04my/Downloads/urban_poverty_percent_urban_people_below_national_urban.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="perc_people") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%"Brazil",
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# democracy score (https://www.systemicpeace.org/polityproject.html) - democracy_score_use_as_color.csv
dem_score <- read_csv("/Users/rab04my/Downloads/democracy_score_use_as_color.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="score") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%c("Brazil", "Venezuela", "United States", "Denmark"),
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

# corruption perception index (https://www.transparency.org/research/cpi) - corruption_perception_index_cpi.csv
corruption <- read_csv("/Users/rab04my/Downloads/corruption_perception_index_cpi.csv") %>%
  pivot_longer(cols=-country, names_to="year", values_to="score") %>%
  mutate(year=as.numeric(year)) %>%
  filter(country%in%c("Brazil", "Venezuela", "United States", "Denmark"),
         year>=1950,
         year<=2019) %>%
  mutate(year=as.factor(year))

## plots and analysis
# theme(
#   axis.text = element_text(size = rel(1)),
#   axis.text.x=element_text(angle=0, size=rel(1)),
#   axis.text.y=element_text(size=rel(1)),
#   axis.title = element_text(size = rel(1)),
#   panel.grid.major = element_line(color = "light gray"),
#   panel.grid.minor = element_line(color = "light gray"),
#   panel.background = element_rect(fill = "white"),
#   strip.background = element_rect(fill = "darkgray", size=0.5),
#   strip.text = element_text(face="bold", size=rel(1), color="white"),
#   legend.position = "bottom",
#   legend.justification = "left",
#   legend.title=element_text(size=rel(1)),
#   legend.text=element_text(size=rel(1)),
#   panel.border = element_rect(color="gray", fill=NA, size=0.5)
# )

# total population
p1 <- subset(pop, year%in%c(1964,1985))
p_pop <- pop %>%
  #mutate(year=as.numeric(year)) %>%
  ggplot(aes(x=year, y=population)) +
  geom_point(size=1.5) +
  #geom_smooth(aes(as.numeric(year)), method="loess") +
  #gghighlight(population>=190000000, label_key=year) +
  geom_point(data=p1, colour=c("red", "blue"), size=1.5) +
  geom_label_repel(data=p1, label=c("1964", "1985"), direction="x", force=10) +
  scale_y_continuous(labels=comma, breaks=seq(0,max(pop$population+10000000),10000000)) +
  #scale_fill_hp(option="Always") +
  scale_color_hp_d(option="HarryPotter") +
  labs(y="Population", x="", title="Brazilian Population") +
  theme_hc() +
  theme(
    axis.text.x=element_text(angle=90, hjust=0, vjust=0.5),
    legend.position="top"
    )
p_pop

# gini coefficient
p2 <- gini %>%
  filter(as.numeric(year)<=62) %>%
  ggplot(aes(x=year, y=gini_coeff, group=country)) +
  geom_line(aes(colour=country), alpha=0.6, size=1.5) +
  geom_point(aes(x="1964", y=gini_coeff[country=="Brazil" & year=="1964"]), colour="red", size=3) +
  geom_point(aes(x="1985", y=gini_coeff[country=="Brazil" & year=="1985"]), colour="blue", size=3) +
  geom_label_repel(aes(label=ifelse(year=="1964" & country=="Brazil", "1964", "")), force=10) +
  geom_label_repel(aes(label=ifelse(year=="1985" & country=="Brazil", "1985", "")), force=10) +
  #scale_fill_hp(option="Always") +
  scale_color_hp_d(option="HarryPotter") +
  labs(y="Gini Coefficient", x="", colour="",
       title="Gini Coefficient") +
  theme_hc() +
  theme(
    axis.text.x=element_text(angle=90, hjust=0, vjust=0.5),
    legend.position="none"
  )
p2

# corruption perception
p3 <- corruption %>%
  ggplot(aes(x=year, y=score, group=country)) +
  geom_line(aes(colour=country), alpha=0.6, size=1.5) +
  #scale_fill_hp(option="Always") +
  scale_color_hp_d(option="HarryPotter") +
  labs(y="Perception Score", x="", colour="",
       title="Corruption Score") +
  theme_hc() +
  theme(
    axis.text.x=element_text(angle=90, hjust=0, vjust=0.5),
    legend.position="right"
  )
p3

# democracy score
p4 <- dem_score %>%
  filter(as.numeric(year)<=2011) %>%
  ggplot(aes(x=year, y=score, group=country)) +
  geom_line(aes(colour=country), alpha=0.6, size=1.5) +
  geom_point(aes(x="1964", y=score[country=="Brazil" & year=="1964"]), colour="red", size=3) +
  geom_point(aes(x="1985", y=score[country=="Brazil" & year=="1985"]), colour="blue", size=3) +
  geom_label_repel(label=ifelse(dem_score$year=="1964" & dem_score$country=="Brazil", "1964", ""), force=10) +
  geom_label_repel(label=ifelse(dem_score$year=="1985" & dem_score$country=="Brazil", "1985", ""), force=10) +
  #geom_col(aes(fill=country), position="dodge", alpha=0.5) +
  #geom_point(aes(colour=country)) +
  #scale_fill_hp(option="Always") +
  scale_color_hp_d(option="HarryPotter") +
  labs(y="Polity Score", x="", colour="",
       title="Democracy Score") +
  theme_hc() +
  theme(
    axis.text.x=element_text(angle=90, hjust=0, vjust=0.5),
    legend.position="right"
  )
p4

p5 <- income %>%
  #filter(as.numeric(year)<=2011) %>%
  ggplot(aes(x=year, y=income, group=country)) +
  geom_line(aes(colour=country), alpha=0.6, size=1.5) +
  geom_point(aes(x="1964", y=income[country=="Brazil" & year=="1964"]), colour="red", size=3) +
  geom_point(aes(x="1985", y=income[country=="Brazil" & year=="1985"]), colour="blue", size=3) +
  geom_label_repel(label=ifelse(income$year=="1964" & income$country=="Brazil", "1964", ""), force=10) +
  geom_label_repel(label=ifelse(income$year=="1985" & income$country=="Brazil", "1985", ""), force=10) +
  scale_y_continuous(labels=dollar) +
  #geom_col(aes(fill=country), position="dodge", alpha=0.5) +
  #geom_point(aes(colour=country)) +
  #scale_fill_hp(option="Always") +
  scale_color_hp_d(option="HarryPotter") +
  labs(y="Income per Person", x="", colour="",
       title="Income") +
  theme_hc() +
  theme(
    axis.text.x=element_text(angle=90, hjust=0, vjust=0.5),
    legend.position="none"
  )
p5

p6 <- num_pov %>%
  filter(!is.na(num_people),
         as.numeric(levels(year))<=2011) %>%
  ggplot(aes(x=year, y=num_people, group=country)) +
  geom_line(aes(colour=country), alpha=0.6, size=1.5) +
  #geom_point(aes(x="1964", y=num_people[country=="Brazil" & year=="1964"]), colour="red", size=3) +
  geom_point(aes(x="1985", y=num_people[country=="Brazil" & year=="1985"]), colour="blue", size=3) +
  #geom_label_repel(aes(label=ifelse(year=="1964" & country=="Brazil", "1964", "")), force=10) +
  geom_label_repel(aes(label=ifelse(year=="1985" & country=="Brazil", "1985", "")), force=10) +
  #geom_col(aes(fill=country), position="dodge", alpha=0.5) +
  #geom_point(aes(colour=country)) +
  #scale_fill_hp(option="Always") +
  scale_color_hp_d(option="HarryPotter") +
  labs(y="Number of People (MM)", x="", colour="",
       title="Number of the Poor Population") +
  theme_hc() +
  theme(
    axis.text.x=element_text(angle=90, hjust=0, vjust=0.5),
    legend.position="none"
  )
p6

#label=scales::percent_format(accuracy=1)
# rural and urban poverty (%)
pov_sample <- inner_join(rural_pov[!is.na(rural_pov$perc_people),],
                         urban_pov[!is.na(urban_pov$perc_people),], by=c("country", "year"))
names(pov_sample) <- c("country","year","urban_poverty (%)","rural_poverty (%)")
pov_sample

# combinations and exports (details in text and captions)
ggsave(filename=paste0("/Users/rab04my/Downloads/population.png"), plot=p_pop, scale=1, width=11, height=6, units="in", dpi=300)

gini_plus_democracy <- p2 / p4
ggsave(filename=paste0("/Users/rab04my/Downloads/patchplot_demo_and_gini.png"), plot=gini_plus_democracy, scale=1, width=11, height=6, units="in", dpi=300)

ggsave(filename=paste0("/Users/rab04my/Downloads/corruption.png"), plot=p3, scale=1, width=11, height=6, units="in", dpi=300)
ggsave(filename=paste0("/Users/rab04my/Downloads/income.png"), plot=p5, scale=1, width=11, height=6, units="in", dpi=300)
ggsave(filename=paste0("/Users/rab04my/Downloads/numpoverty.png"), plot=p6, scale=1, width=11, height=6, units="in", dpi=300)

extra_plot <- p6 / (p2+theme(axis.text.x=element_text(size=rel(0.6))) | p4+theme(axis.text.x=element_text(size=rel(0.6)), legend.position="bottom"))
ggsave(filename=paste0("/Users/rab04my/Downloads/patchplot_demo_and_gini_and_pov.png"), plot=extra_plot, scale=1, width=11, height=6, units="in", dpi=300)

#
#library(corrr)

zz <- left_join(num_pov, dem_score, by=c("country", "year"))
zz <- inner_join(zz, gini, by=c("country", "year"))
zz <- inner_join(zz, income, by=c("country", "year"))
zz <- inner_join(zz, pop, by=c("country", "year"))
zz <- zz[!is.na(zz$num_people),]
names(zz) <- c("country","year","people_pov","dem_score","gini_coeff","income","population")

ggpairs(zz[,c(-1,-2)])
c1 <- ggcorr(zz[,c(-1,-2)], label=TRUE, label_size=5, label_color='black', label_round=2)
ggsave(filename=paste0("/Users/rab04my/Downloads/corplot.png"), plot=c1, scale=1, width=11, height=6, units="in", dpi=300)


## forecasting
p <- 3
df1 <- data.frame(ds=lubridate::ymd(paste0(gm_brazil$year,"-01-01")), y=gm_brazil$gdpPercap)
gm_brazil_m <- prophet(df=df1, yearly.seasonality=TRUE, daily.seasonality=FALSE, growth="linear")
future <- make_future_dataframe(gm_brazil_m, periods=p)
f <- predict(gm_brazil_m, future)
plot(gm_brazil_m, f)

fit1 <- HoltWinters(gm_brazil$gdpPercap, beta=TRUE, gamma=FALSE)
plot(forecast(fit1, p)) 

fit2 <- auto.arima(gm_brazil$gdpPercap) 
plot(forecast(fit2, p)) 
#autoplot(fit2)

#fit <- meanf(gm_brazil$gdpPercap)
#plot(forecast(fit, p)) 
#fit <- ets(gm_brazil$gdpPercap)
#plot(forecast(fit, p)) 


