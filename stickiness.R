### Member Stickiness Analysis
### Author: RAB

### Summary
# How sticky are our people's costs. The people in 2014 directly compared to 12 and 16 may be different 
# but they seem to be the same. The idea is to answer the question: how many people are spending their 
# deductible in multiple years. Not just by numbers but actual member.
# 
# Look at the distribution of people through time and see if there is stickiness.
# 
# Definition of Done:
# Number of people/members for whom their low state of spend is static and over 3k, 5k, and 10k year over year,
# using Allowed Amount (both year-to-year and "skip" years).
###

## packages
library(data.table)
library(dplyr)
library(reshape2)
library(stringi)
library(readxl)
library(readr)
library(tidyverse)
library(ggplot2)
library(ggrepel)
library(ggridges)
library(lubridate)
library(reticulate)
library(markovchain)
library(skimr)

library(tidygraph)
library(ggraph)
library(igraph)
library(plotly)

## Markov chain by hand on a vector
x <- (matrix((c(1,2,1,1,3,2,3,3,2)), nrow = 3, ncol = 3))
x <- c(1,2,1,1,3,4,4,1,2,4,1,4,3,4,4,4,3,1,3,2,3,3,3,3,3,3,3,3,4,2,2,3)
n <- p <- matrix(nrow = 4, ncol = 4, 0) #count and probability matrix where nrow/ncol = len(unique(x))
for (t in 1:(length(x) - 1)) n[x[t], x[t + 1]] <- n[x[t], x[t + 1]] + 1 #count of transition pairs, e.g. count(3->3, 3->3, ..., 3->3) = 7
for (i in 1:3) p[i, ] <- n[i, ] / sum(n[i, ]) # each cell divided by the sum of the row counts = probability
p
##

## path
fp <- "/Users/rab04my/Documents/claims/member_stickiness/"

## consecutive years
# z <- read_csv(paste0(fp, "annual_allowed.csv"))
# z$member <- as.factor(z$member)

## skip years
z <- read_csv(paste0(fp, "skip2Year_data(1).csv"))
z$member <- as.factor(z$member)
# names(z) <- str_replace_all(names(z), "_[0-9]+", "")

a <- as.matrix(z[,c(-1)]) # remove member id column
# colnames(a) <- str_replace_all(colnames(a), "_spend", "")

# create grouping labels
b <- a
a[b==0] <- "[0]"
#a[b>=1 & b<1000] <- "[1,1000)"
a[b>0 & b<3000] <- "[1,3000)"
a[b>=3000 & b<5000] <- "[3000,5000)"
a[b>=5000 & b<10000] <- "[5000,10000)"
a[b>=10000] <- "[10000, Inf)" #"NA"
a <- apply(a, 2, as.character)
for(i in 1:length(colnames(a))) { a[,i] <- paste0(colnames(a)[i], "_", a[,i]) }
rownames(a) <- z$member

## data table for Member ID information (to join with state transition probabilities)
Member <- a %>%
  as.tibble() %>%
  mutate("member"=rownames(a)) %>%
  gather(from_year, from_value, c(1,3,5,7)) %>%
  gather(to_year, to_value, c(1,2,3,4)) %>%
  mutate(from_year=as.numeric(str_sub(from_year, 1,4)),
         to_year=as.numeric(str_sub(to_year, 1,4)),
         from_value=unlist(lapply(str_split(from_value, "_"), function(x) x[2])),
         to_value=unlist(lapply(str_split(to_value, "_"), function(x) x[2])))

## create Markov model to estimate state transition probabilities
mcX_all <- markovchainFit(a)
mcX <- mcX_all$estimate
mcX_conf <- mcX_all$standardError

## tidy up Markov chain output into data table
mcX_df <- as_tibble(as(mcX, "data.frame")) %>% 
  arrange(-prob) %>%
  distinct(t0, t1, .keep_all=TRUE) %>%
  filter(prob>0) %>%
  mutate(t0_year=as.numeric(str_sub(t0, 1, 4)),
         t1_year=as.numeric(str_sub(t1, 1, 4)),
         #delta=abs(t0_year-t1_year),
         n_less=sum(t0_year<=t1_year),
         n_more=sum(t0_year>t1_year),
         prob2=prob*(n_less/n_more)) %>%
  filter(t0_year<=t1_year) %>%
  select(-t0_year:-prob2)
mcX_df

## widen Markov chain results table to join with Member ID information
MC <- mcX_df %>%
  mutate(from_year=as.numeric(str_sub(t0, 1,4)),
         to_year=as.numeric(str_sub(t1, 1,4)),
         from_value=unlist(lapply(str_split(t0, "_"), function(x) x[2])),
         to_value=unlist(lapply(str_split(t1, "_"), function(x) x[2]))) %>%
  select(-t0, -t1)

## table with joined member ID and Markov model information
DF <- inner_join(MC, Member, by=c('from_year', 'from_value', 'to_year', 'to_value')) %>%
  mutate(change=ifelse(from_value!=to_value,TRUE,FALSE),
         t0=paste0(from_year,"_",from_value),
         t1=paste0(to_year,"_",to_value)) %>%
  group_by(member) %>%
  mutate(n_change=sum(change)) %>%
  ungroup() %>% 
  select(-change) %>%
  arrange(-n_change, member)

## add counts and values for transitions indicating whether a member went UP, DOWN, or No Change for spend
from_vals <- sapply(str_split(DF$from_value, ","), function(x) {
  out <- str_trim(str_replace(x[2], "\\)", ""), side="both")
  out <- ifelse(out=="Inf", Inf, as.numeric(out))
  out <- ifelse(is.na(out), 0, out)
})
to_vals <- sapply(str_split(DF$to_value, ","), function(x) {
  out <- str_trim(str_replace(x[2], "\\)", ""), side="both")
  out <- ifelse(out=="Inf", Inf, as.numeric(out))
  out <- ifelse(is.na(out), 0, out)
})

DF <- DF %>% 
  mutate(from_upper=from_vals,
         to_upper=to_vals,
         from_upper_limit=as.numeric(from_vals),
         to_upper_limit=as.numeric(to_vals),
         change=as.factor(ifelse(from_upper<to_upper, "UP", ifelse(from_upper>to_upper, "DOWN", "NOCHANGE")))
  )
write_csv(DF, paste0(fp, "member_stickiness_skipyears.csv"))

## slice and dice member cost changes and probabilities (analysis using the DF table)
n_rows <- nrow(DF)
perc_up <- sum(DF$change=="UP")/n_rows
perc_down <- sum(DF$change=="DOWN")/n_rows
perc_nc <- sum(DF$change=="NOCHANGE")/n_rows

## create aggregated summary table of member spend
change_by_member <- DF %>%
  mutate(skip_years=as.factor(paste0(from_year, "_to_", to_year)),
         bucket=as.factor(paste0(from_value, "_to_", to_value))) %>%
  group_by(member, skip_years) %>%
  summarise(i=n(),
            UP=sum(change=="UP"),
            DOWN=sum(change=="DOWN"),
            NOCHANGE=sum(change=="NOCHANGE"),
            UP_prob=mean(prob[change=="UP"], na.rm=TRUE),
            DOWN_prob=mean(prob[change=="DOWN"], na.rm=TRUE),
            NOCHANGE_prob=mean(prob[change=="NOCHANGE"], na.rm=TRUE)
  ) %>%
  ungroup() %>%
  group_by(member) %>%
  summarise(UP=sum(UP), DOWN=sum(DOWN), NOCHANGE=sum(NOCHANGE),
            UP_prob=mean(UP_prob, na.rm=TRUE),
            DOWN_prob=mean(DOWN_prob, na.rm=TRUE),
            NOCHANGE_prob=mean(NOCHANGE_prob, na.rm=TRUE)) %>%
  arrange(member)
write_csv(change_by_member, paste0(fp, "member_stickiness_skipyears2.csv"))

## probability aggregate
tmp <- DF %>%
  group_by(member, from_year, to_year, from_upper, to_upper) %>%
  summarise(i=n(),
            UP=sum(change=="UP"),
            DOWN=sum(change=="DOWN"),
            NOCHANGE=sum(change=="NOCHANGE"),
            UP_prob=mean(prob[change=="UP"], na.rm=TRUE),
            DOWN_prob=mean(prob[change=="DOWN"], na.rm=TRUE),
            NOCHANGE_prob=mean(prob[change=="NOCHANGE"], na.rm=TRUE)
  ) %>%
  ungroup() %>%
  group_by(member) %>%
  summarise(year_and_spend=paste0(paste0(from_year, ":" ,from_upper, " to ", to_year, ":", to_upper), collapse=", "), 
            #to_upper=paste0(paste0(to_year, ":", to_upper), collapse=","),
            UP=sum(UP), DOWN=sum(DOWN), NOCHANGE=sum(NOCHANGE),
            UP_prob=mean(UP_prob, na.rm=TRUE),
            DOWN_prob=mean(DOWN_prob, na.rm=TRUE),
            NOCHANGE_prob=mean(NOCHANGE_prob, na.rm=TRUE)) %>%
  arrange(member)
write_csv(tmp, paste0(fp, "member_stickiness_skipyears3.csv"))

## 
tmp2 <- DF %>%
  filter(from_value==to_value) %>%
  group_by(from_value, to_value) %>% #from_year, to_year, member, 
  summarise(avg_prob=mean(prob, na.rm=TRUE)) %>%
  arrange(to_value)

tmp2 <- DF %>%
  # from_year, to_year, [3000, 5000]
  filter(to_value=="[3000,5000)") %>%
  group_by(member) %>%
  summarise(n=n(),
            prob=mean(prob, na.rm=TRUE)) %>%
  arrange(-n, -prob, member)

## plot spend probability trends over years and buckets
DF %>%
  #filter(from_value=="[3000,5000)") %>%
  #filter(member %in% c("10010", "9979", "9971", "10013", "10029")) %>%
  mutate(skip_years=as.factor(paste0(from_year, "_to_", to_year)),
         bucket=as.factor(paste0(from_value, "_to_", to_value))) %>%
  ggplot(aes(x=prob, y=as.factor(skip_years), fill=as.factor(change))) + 
  geom_density_ridges(alpha=0.2) +
  scale_fill_brewer(palette = "Set1") +
  #facet_grid(~bucket) +
  facet_wrap("bucket", nrow=12, ncol=3) +
  #scale_y_discrete(expand=c(0.01, 0)) +
  #scale_x_continuous(expand=c(0.01, 0)) +
  xlab("Estimated Transition Probability") +
  ylab("Skip Years") +
  theme_ridges() +
  guides(fill=guide_legend(title="Change Direction"))
ggsave(filename=paste0(fp, "member_stickiness_skipyears3.png"), 
       width=18, height=15, units="in", dpi=350)

## generate acyclic directed graph based on Markov model probabilities
tmp <- DF %>% 
  as_tibble() %>%
  mutate(source=t0,
         target=t1,
         type="directed",
         id=1:n(),
         prob=prob) %>%
  select(member, source, target, prob, member, id)

# as_tbl_graph(tmp, directed = TRUE)
# g_raw <- graph_from_data_frame(tmp, directed=TRUE) 
# g_adj <- as_adjacency_matrix(g_raw)
# g <- upgrade_graph(g_raw)

g <- tmp %>%
  #filter(n==4) %>%
  select(-member) %>%
  as_tbl_graph(directed=TRUE) %>%
  # Remove loops
  activate(edges) %>%
  filter(!edge_is_loop(), !edge_is_multiple()) %>%
  #       !edge_is_multiple()
  #       edge_is_incident(3)
  #       ) %>%
  # Remove isolated nodes
  activate(nodes) %>%
  filter(!node_is_isolated())
#mutate(centrality = centrality_authority()
#weighted_degree = centrality_degree()/local_ave_degree(),
#group = group_infomap()
#       )

## define graph colors per year
cols <- data.frame(levels=sort(as.character(unique(c(as.character(tmp$source), as.character(tmp$target))))), col=NA)
cols[grep("2011_", cols$levels), "col"] <- "green"
cols[grep("2012_", cols$levels), "col"] <- "blue"
cols[grep("2013_", cols$levels), "col"] <- "darkgray"
cols[grep("2014_", cols$levels), "col"] <- "darkorchid2"
cols[grep("2015_", cols$levels), "col"] <- "yellow"
cols[grep("2016_", cols$levels), "col"] <- "deeppink"
my_levels <- cols$col
levels(my_levels) <- cols$levels

# cols <- data.frame(levels=sort(as.character(unique(c(as.character(tmp$source), as.character(tmp$target))))), col=NA)
# cols[grep("2011[a-z]_", cols$levels), "col"] <- "green"
# cols[grep("2012[a-z]_", cols$levels), "col"] <- "blue"
# cols[grep("2013a_", cols$levels), "col"] <- "orange"
# cols[grep("2013c_", cols$levels), "col"] <- "darkgray"
# cols[grep("2014b_", cols$levels), "col"] <- "red"
# cols[grep("2014d_", cols$levels), "col"] <- "darkorchid2"
# cols[grep("2015[a-z]_", cols$levels), "col"] <- "yellow"
# cols[grep("2016[a-z]_", cols$levels), "col"] <- "deeppink"
# my_levels <- cols$col
# levels(my_levels) <- cols$levels

g %>% ggraph(layout='kk') + #drl, linear, kk
  #geom_edge_density(aes(fill=prob)) +
  #geom_edge_arc(aes(label=round(prob,3), width=prob), alpha=0.2, 
  #              arrow=arrow(length=unit(5,'mm'), type="closed", ends="last", angle=10)) + 
  geom_edge_link(aes(label=round(prob,3), width=prob), colour="light gray", alpha=0.5, check_overlap=TRUE,
                 arrow=arrow(length=unit(4,'mm'), type="closed", ends="last", angle=10),
                 label_dodge=unit(2.5,'mm')) + #aes(width=prob)
  scale_edge_width(range=c(0.1, 2)) +
  labs(edge_width="Probability") +
  geom_node_point(aes(color=factor(name)), size=6, alpha=0.15) + #, color="red"
  scale_color_manual(values=my_levels) +
  #scale_color_brewer(name="", palette="Set1") +
  geom_node_text(aes(label=name), size=5, colour='black', repel=TRUE, check_overlap=TRUE) + #nudge_y=0.2
  #geom_text(aes(-2.5, 3, label="2014 - 2016"), size=7) +
  #geom_text(aes(-0.5, 0.6, label="2011 - 2013"), size=7) +
  #geom_text(aes(2, 0.6, label="2012 - 2014"), size=7) +
  #geom_text(aes(0.3, 3.7, label="2013 - 2015"), size=7) +
  theme_graph() + 
  guides(color=FALSE)
ggsave(filename=paste0(fp, "annual_spend_probabilities_skipyears3.png"), 
       width=18, height=15, units="in", dpi=350)
write_csv(tmp, paste0(fp, "annual_spend_probabilities_skipyears_table.csv"))







