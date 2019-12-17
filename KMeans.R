
DATASET_NAME = "LETTERRECOGNITION"
DATA_CENTER = TRUE
DATA_SCALE = TRUE
if(DATASET_NAME == "LETTERRECOGNITION")
{
  data <- read.table("datasets/letter-recognition.data", sep=',')
  
  #set num of rows and cols in data
  n_row <- nrow(data) 
  n_col <- ncol(data) 
  
  x <- as.matrix(data[,2:n_col]) 	# features
  labels <- as.matrix(data[,1]) 
  
  # x = x[1:100, ]
  # labels = labels[1:100]
  
  
  print(x)
  dataset = x
  
  class_labels = c(labels)
  label_colors = c(labels)
  classes = unique(label_colors)
  print(classes)
  for (i in 1:length(label_colors))
  {
    for(a in 1:length(classes))
    {
      if(label_colors[i] == classes[a])
      {
        label_colors[i] = a
        class_labels[i] = a
      }
    }
    
  }
}


as.numeric(Sys.time())-> t
set.seed(25)
library("Rfast")
dataset = scale(dataset, center = DATA_CENTER, scale = DATA_SCALE)
summary(dataset)
head(dataset)

# plot(dataset, col = label_colors, main = paste("Dataset center=",DATA_CENTER," scale=", DATA_CENTER))
kmeans_result = kmeans(dataset, 26,nstart=1, iter.max = 500)
print(kmeans_result$centers)
plot(dataset, col=kmeans_result$cluster, main="Kmeans Applied")
points(kmeans_result$centers, col = "orange", pch=16, cex=3)

# methods : ward.D, ward.D2, single, complete, average, median
# dist method : "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"
hierarchical_clustering_result <- hclust(dist(dataset,method = "euclidean"), method="ward")
tree_cut = cutree(hierarchical_clustering_result, k = 26)
plot(dataset, col=tree_cut, main="Hierarchical clustering Applied on Letter Recognition")
  
plot(hierarchical_clustering_result, main="Hierarchical clustering Applied on Letter Recognition")











