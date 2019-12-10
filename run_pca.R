#load libraries
library(pca3d)


#load letter-recognition data
data <- read.table("datasets/letter-recognition.data", sep=',')

#set num of rows and cols in data
n_row <- nrow(data) 
n_col <- ncol(data) 

x <- as.matrix(data[,2:n_col]) 	# features
y <- as.matrix(data[,1]) 		# labels

#apply PCA
PCA <- princomp(x, cor=TRUE)

#draw screeplot
screeplot(PCA, npcs=n_col-1 , type='lines')

#draw 3d pca clusters
p3 <- pca3d(PCA, group=y, show.ellipses=FALSE, legend="topleft")
snapshotPCA3d(file="plots/pca3d.png")