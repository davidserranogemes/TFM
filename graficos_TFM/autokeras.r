  library(ggplot2)
  route <- "../logs/AUTOKERAS/FINAL/"
  variants <- list.files(route)
  
  for(k in variants){
    dir.create(k)
  }
  
  files.list <- list.files(paste(route,variants, sep=""),full.names = TRUE)
  #Solo nos interesan los pares
  
  
  for (i in seq(from=2, to= length(files.list),by=2)) {
    data <- read.csv(files.list[i], header = FALSE,col.names = c("Epoch", "ACC"), sep = " ")
    data <- cbind(data,as.numeric(substr(data$Epoch,6,100000)))
    data <- data[,c(3,2)] 
    data <- cbind(data,numeric(nrow(data)))
    colnames(data) <- c("Epoch","ACC","Model")
    
    model.aux <- c(diff(data$Epoch)==1,TRUE)
    curr.model <- 1
    for (j in seq_along(model.aux)) {
      if (!model.aux[j]) {
        curr.model <- curr.model+1
      }
      data$Model[j] <- curr.model
      
    }
    data$Model <- factor(data$Model)
    gg <- ggplot(data = data, aes(x=Epoch, y=ACC)) + geom_line(aes(colour=Model))
    
    file.png <- paste(".",substr(files.list[i],24,10000),sep="")
    file.png <- paste(gsub(".txt","",file.png),".png",sep="")
    ggsave(file.png, plot = gg)
    
  }
  
