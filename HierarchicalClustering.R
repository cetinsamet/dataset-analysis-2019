
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
  
  x = x[1:100, ]
  labels = labels[1:100]
  
  
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

# data generation has constant value to obtain the same data
as.numeric(Sys.time())-> t
set.seed((t - floor(t)) * 1e8 -> seed)

dataset = scale(dataset, center = DATA_CENTER, scale = DATA_SCALE)


# to show the clusters properly we need to project data a plotable dimension...
data_pca = prcomp(dataset, center = DATA_CENTER, scale = DATA_SCALE, retx=TRUE)

plot(data_pca$x[, 1:2], col=label_colors, main="Original Projected Data")

# where the hierarchical clustering will be run, in original space or projected space...

cluster_data = dataset # data_pca$x[, 1:2] # dataset #  # dataset # data_pca$x[, 1:2]

# methods : ward.D, ward.D2, single, complete, average, median
# dist method : "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"
hierarchical_clustering_result <- hclust(dist(cluster_data,method = "euclidean"), method="average")
tree_cut = cutree(hierarchical_clustering_result, k = 3)
# show clusters on projected data...
plot(data_pca$x[, 1:2], col=tree_cut, main="Hierarchical clustering Applied Linkage(Average)")
plot(hierarchical_clustering_result, main=paste("Hierarchical clustering Applied Linkage(Average) (", ncol(cluster_data), "D)"))



hierarchical_clustering_result <- hclust(dist(cluster_data,method = "euclidean"), method="single")
tree_cut = cutree(hierarchical_clustering_result, k = 3)
# show clusters on projected data...
plot(data_pca$x[, 1:2], col=tree_cut, main="Hierarchical clustering Applied Linkage(Single)")
plot(hierarchical_clustering_result, main=paste("Hierarchical clustering Applied Linkage(Single) (", ncol(cluster_data), "D)"))


hierarchical_clustering_result <- hclust(dist(cluster_data,method = "euclidean"), method="complete")
tree_cut = cutree(hierarchical_clustering_result, k = 3)
# show clusters on projected data...
plot(data_pca$x[, 1:2], col=tree_cut, main="Hierarchical clustering Applied Linkage(Complete)")
plot(hierarchical_clustering_result, main=paste("Hierarchical clustering Applied Linkage(Complete) (", ncol(cluster_data), "D)"))



hierarchical_clustering_result <- hclust(dist(cluster_data,method = "euclidean"), method="ward")
tree_cut = cutree(hierarchical_clustering_result, k = 3)
plot(data_pca$x[, 1:2], col=tree_cut, main="Hierarchical clustering Applied Linkage(Ward)")
plot(hierarchical_clustering_result, main=paste("Hierarchical clustering Applied Linkage(Ward) (", ncol(cluster_data), "D)"))

# Part 3


set.seed(120)
result_matrix = matrix(rep(0, 20), nrow = 1)
for(experiment in 1:100)
{
  error_vector = c()
  for (k in 1:20)
  {
    as.numeric(Sys.time())-> t
    set.seed((t - floor(t)) * 1e8 -> seed)
    
    temp_kmeans = kmeans(cluster_data, k,nstart=20)
    error_vector = c(error_vector, temp_kmeans$tot.withinss)
  }
  
  result_matrix = rbind(result_matrix, error_vector)
  
}
error_vector = colMeans(result_matrix)

plot(x=1:20, type="b", y=error_vector, main="Total within class error and K")

result_matrix = matrix(rep(0, 30), nrow = 1)
for(experiment in 1:100)
{
  error_vector = c()
  for (nstart in 1:30)
  {
    as.numeric(Sys.time())-> t
    set.seed((t - floor(t)) * 1e8 -> seed)
    
    temp_kmeans = kmeans(cluster_data, 3,nstart=nstart)
    error_vector = c(error_vector, temp_kmeans$tot.withinss)
  }
  
  result_matrix = rbind(result_matrix, error_vector)
  
}
error_vector = colMeans(result_matrix)

plot(x=1:30, type="b", y=error_vector, main="Total within class error and nstart")

result_matrix = matrix(rep(0, 30), nrow = 1)
for(experiment in 1:100)
{
  error_vector = c()
  for (itermax in 1:30)
  {
    as.numeric(Sys.time())-> t
    set.seed((t - floor(t)) * 1e8 -> seed)
    
    temp_kmeans = kmeans(cluster_data, 3,nstart=10, iter.max = itermax)
    error_vector = c(error_vector, temp_kmeans$tot.withinss)
  }
  
  result_matrix = rbind(result_matrix, error_vector)
  
}
error_vector = colMeans(result_matrix)

plot(x=1:30, type="b", y=error_vector, main="Total within class error and itermax")

kmeans_result = kmeans(cluster_data, 3,nstart=20)
plot(data_pca$x[, 1:2], col=kmeans_result$cluster, main="Kmeans Applied to Letter Recognition")
# points(kmeans_result$centers, col = "orange", pch=16, cex=3)




