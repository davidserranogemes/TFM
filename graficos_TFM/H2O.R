library(ggplot2)
library("reshape2")
route <- "../logs/H2O/FINAL/"
variants <- list.files(route)

for(k in variants){
  dir.create(k)
}

files.list <- list.files(paste(route,variants, sep=""),full.names = TRUE)
#Solo nos interesan los pares



for (i in seq(from=2, to= length(files.list),by=2)) {
    if(length(grep("AUTHOMATIC",files.list[i]))==1){
      dataset.info <- "H2O AUTHOMATIC"
      if(length(grep("imdb",files.list[i]))!=1){
        
        data <- read.csv(files.list[i], header = FALSE,col.names = c("Model","Mean Per-Class Error"),sep = " ")
        c.name <- "Mean Per-Class Error"
      }else{
        
        data <- read.csv(files.list[i], header = FALSE,col.names = c("Model","AUC"),sep = " ")
        c.name <- "AUC"
      }

      
    }else{
      dataset.info <- "H2O GRID"
      if(length(grep("imdb",files.list[i]))!=1){
        data <- read.csv(files.list[i], header = FALSE,col.names = c("Metric", "Value"),sep = ":")
      }else{
        
        data <- read.csv(files.list[i], header = FALSE,col.names = c("Metric","Value"),sep = ":")

      }
      c.name <- data$Metric[1]
      #Nos quedamos solo con el valor de train
      data <- data.frame(data$Value[seq(1,nrow(data),2)])
      data <- cbind(1:nrow(data),data)
      

    }
  

    
    if(as.character(c.name)=="Mean Per-Class Error"){
      data <- cbind(data,cummin(data[,2]))
      colnames(data) <- c("Model", "MPCE","MinMPCE")  
      y.lab <- "MPCE"
    }else{
      data <- cbind(data,cummax(data[,2]))
      colnames(data) <- c("Model", "AUC","MaxAUC")
      y.lab <- "AUC"
    }

    file.png <- paste(".",substr(files.list[i],18,10000),sep="")
    file.png <- paste(gsub(".txt","",file.png),".png",sep="")
  
    gg <- ggplot(data,aes(Model)) + 
            geom_line(aes(y = data[,2]), colour= "red") + 
            geom_line(aes(y = data[,3]), colour= "green") +
            xlab("Model") +
            ylab(y.lab) +
            ggtitle(paste("Dataset: ",dataset.info," ---- Best",y.lab,"vs current model", y.lab))
    

  ggsave(file.png, plot = gg)
  
}
