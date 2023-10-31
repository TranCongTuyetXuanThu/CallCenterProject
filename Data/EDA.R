install.packages("tuneR")
install.packages("audio")
install.packages("reticulate")
install.packages("plotrix")

library(readr)
library(dplyr)
library(ggplot2)
library(tuneR)
library(reticulate)
library(plotrix)

getwd()
setwd("/Users/baoxuyennguyenle/Desktop/SER")

#1. Quick look
# Read file csv
data <- read.csv("processed_label_data.csv")

head(data,10)
summary(data)
str(data)
table(data$label)

"/*
- File .csv includes 4 columns:
 + ID: names of audio files
 + label: Label of audio files
 + length: length of audio files
 + path: directory to the audio files
- Data has 10039 samples with 11 different labels. There are 2 unidentified label:xxx,other
*/"

# Data distribution (pie chart)
percentage <- round(prop.table(table(data$label)) * 100,1)
freq_data <- data.frame(percentage)
colnames(freq_data) <- c("Label", "Percentage")

ggplot(freq_data, aes(x = "", y = Percentage, fill = Label)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  ggtitle("Label Distribution") +
  theme_minimal() +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust = 0.5)) + 
  geom_text(aes(label = paste0(Label, "\n", Percentage, "%")), 
            position = position_stack(vjust = 0.5), 
            size = 4, color = "black") +
  scale_fill_brewer(palette = "Set3")

# Create new labels: neutral, negative, positive and remove all unidentified labels
remove = c("xxx", "oth")
data <- subset(data, !(data$label %in% remove))

nrow(data)
head(data,10)

categorize_emotions <- function(emotion){
  if ((emotion) %in% c("neu","fru")){
    return("neutral")
  }
  else if ((emotion) %in% c("exc","hap","sur")){
    return("positive")
  }
  else{
    return("negative")
  }
}
data$label <- sapply(data$label, categorize_emotions)

# Data distribution (pie chart)
percentage <- round(prop.table(table(data$label)) * 100,1)
freq_data <- data.frame(percentage)
colnames(freq_data) <- c("Label", "Percentage")

ggplot(freq_data, aes(x = "", y = Percentage, fill = Label)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  ggtitle("Label Distribution") +
  theme_minimal() +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust = 0.5)) + 
  geom_text(aes(label = paste0(Label, "\n", Percentage, "%")), 
            position = position_stack(vjust = 0.5), 
            size = 4, color = "black") +
  scale_fill_brewer(palette = "Set3")

# Length distribution
data$group <- cut(data$length, breaks = c(0, 15, 20, 25, 35), 
                labels = c("0-15", "16-20", "21-25", "26-35"), 
                include.lowest = TRUE)

ggplot(data, aes(x = group)) +
  geom_bar(fill = "skyblue", color = "black", stat = "count", show.legend = FALSE) +
  ggtitle("Distribution of Audio Length") +
  xlab("Time(s)") +
  ylab("Quantity") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_text(stat='count', aes(label=..count..), vjust=-1)

# waveform
getwd()
path <- '/Users/baoxuyennguyenle/Desktop/SER/IEMOCAP_full_release/wav/'
file <- c(data$ID[1],data$ID[9],data$ID[96])
full_paths <- paste0(path, file, ".wav")
sounds <- lapply(full_paths, readWave)
labels = c('negative', "neutral", "positive")
par(mfrow=c(length(sounds), 1))
for (i in 1:length(sounds)) {
  plot(sounds[[i]], main=paste("Waveform", labels[i]), xlab="Time", ylab="Amplitude")
}

# mfccs
path <- '/Users/baoxuyennguyenle/Desktop/SER/IEMOCAP_full_release/wav_mfcc/'
mfcc_files <- c(data$ID[1],data$ID[9],data$ID[96])
full_paths <- paste0(path, file, ".pt")
labels = c('negative', "neutral", "positive")
par(mfrow=c(length(sounds), 1))
for (file in mfcc_files) {
  py <- import("torch")
  checkpoint <- py$load(file)
  mfcc <- checkpoint$mfcc
  plot(mfcc, main=paste(labels[i]), xlab="Frame", ylab="Frequence")
}



path <- '/Users/baoxuyennguyenle/Desktop/SER/IEMOCAP_full_release/wav_mfcc/'
mfcc_files <- c(data$ID[1], data$ID[9], data$ID[96])
full_paths <- paste0(path, mfcc_files, ".pt")
labels <- c('negative', "neutral", "positive")
par(mfrow=c(length(mfcc_files), 1))

for (i in 1:length(mfcc_files)) {
  file <- full_paths[i]
  py <- import("torch")
  checkpoint <- py$load(file)
  mfcc <- checkpoint$mfcc
  plot(mfcc, main = paste(labels[i]), xlab = "Frame", ylab = "Frequency")
}

color.legend(1, 1, legend = "Amplitude", gradient = "x", rect.col = rainbow(100))
