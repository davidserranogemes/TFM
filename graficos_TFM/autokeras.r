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
      data$ACC <- as.numeric(as.character(data$ACC))
      data <- data[!is.na(data$ACC),]
      
      
      
      if(length(unique(data$Model))>10){ 
        #Limitamos la visualización de modelos porque si no la leyenda es demasiado grande
          gg <- ggplot(data = data[as.numeric(data$Model)<11,], aes(x=Epoch, y=ACC)) + geom_line(aes(colour=Model))
          gg$labels$colour <- paste("Model (Limitado.\nNumero real: ",length(unique(data$Model)),")",sep = "" )
          
            }else{
        gg <- ggplot(data = data, aes(x=Epoch, y=ACC)) + geom_line(aes(colour=Model))  
      }
      
      
      file.png <- paste(".",substr(files.list[i],24,10000),sep="")
      file.png <- paste(gsub(".txt","",file.png),".png",sep="")
      ggsave(file.png, plot = gg)
      
      
      #Evolucion
      data.improve <- data.frame(sapply(1:length(unique(data$Model)),function(x) max(data[data$Model==x,]$ACC)))
      data.improve <- cbind(1:nrow(data.improve),data.improve)
      data.improve <- cbind(data.improve,cummax(data.improve[,2]))
      colnames(data.improve) <- c("Model", "ACC","MAX_ACC")
      
      gg <- ggplot(data.improve,aes(Model)) + 
        geom_line(aes(y = data.improve[,2]), colour= "red") + 
        geom_line(aes(y = data.improve[,3]), colour= "green")+
        xlab("Model") +
        ylab("ACC")
      
      
      file.png <- paste(".",substr(files.list[i],24,10000),sep="")
      file.png <- paste(gsub(".txt","",file.png),"improvement_log.png",sep="")
      ggsave(file.png, plot = gg)
      
    }
    
