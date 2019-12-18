print(seq(1, 20, 2))
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
  int_classes = c()
  print(classes)
  for (i in 1:length(label_colors))
  {
    for(a in 1:length(classes))
    {
      # print(paste("label color : ", label_colors[a], " classes : ", classes[a], "\n"))
      if(label_colors[i] == classes[a])
      {
        label_colors[i] = a
        class_labels[i] = a
        int_classes = c(int_classes, a)
      }
    }
    
  }
}
print(class_labels[1: 20])
print(int_classes[1: 20])
print(label_colors[1:200])
class_labels = int_classes

# scale data
dataset = scale(dataset, center = DATA_CENTER, scale = DATA_SCALE)


# to show the clusters properly we need to project data a plotable dimension...
data_pca = prcomp(dataset, center = DATA_CENTER, scale = DATA_SCALE, retx=TRUE)

plot(dataset, col=label_colors, main=paste(DATASET_NAME, " Dataset"))

plot(data_pca$x[, 1:2], col=label_colors, main=paste("Original Projected Data ", DATASET_NAME))

# where the hierarchical clustering will be run, in original space or projected space...

cluster_data = dataset # data_pca$x[, 1:2] # dataset #  # dataset # data_pca$x[, 1:2]

# Part 1

kmeans_result = kmeans(cluster_data, 26,nstart=25, iter.max = 500)
print(kmeans_result$centers)
plot(data_pca$x[, 1:2], col=kmeans_result$cluster, main=paste("Kmeans Applied on", DATASET_NAME))
# plot(dataset, col=kmeans_result$cluster, main=paste("Kmeans Applied on", DATASET_NAME))
# points(kmeans_result$centers, col = "orange", pch=16, cex=3)

# methods : ward.D, ward.D2, single, complete, average, median
# dist method : "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"
hierarchical_clustering_result <- hclust(dist(cluster_data,method = "euclidean"), method="ward")
tree_cut = cutree(hierarchical_clustering_result, k = 26)
plot(data_pca$x[, 1:2], col=tree_cut, main="Hierarchical clustering Applied Linkage(Ward)")
# plot(dataset, col=tree_cut, main="Hierarchical clustering Applied Linkage(Ward)")

library(clValid)
library(fossil)
library(clusterSim)

# The corrected Rand index provides a measure for assessing the similarity between two partitions, adjusted for chance. Its range is -1 (no agreement) to 1 (perfect agreement). Agreement between the specie types and the cluster solution is 0.62 using Rand index and 0.748 using Meila’s VI
print(class_labels)
print(kmeans_result$cluster)
kmeans_rand_index = rand.index(class_labels, kmeans_result$cluster)
print(kmeans_rand_index)

hc_rand_index = rand.index(class_labels, tree_cut)
print(hc_rand_index)
print(paste("Kmeans(k=26) rand index value ", kmeans_rand_index))
print(paste("Hierarchical Clustering (Ward) rand index value : ", hc_rand_index))



# Part 2
print("**************Part 2**************")
# install.packages("fossil")
# install.packages("clValid")
# install.packages("clusterSim")



# to hold each runs values, later will be averaged...
dunn_matrix = matrix(rep(0, 9), ncol=9)
db_matrix = matrix(rep(0, 9), ncol=9)
silhoutte_matrix = matrix(rep(0, 9), ncol=9)

for (i in 1:5)
{
  silhouette_list = c()
  dunn_index_list = c()
  db_index_list = c()
  
  
  for (k_value in seq(2, 30, 2))
  {
    print(paste("For k = ", k_value))
    kmeans_result = kmeans(cluster_data, k_value,nstart=25, iter.max = 500)
    
    # original data proximity matrix...
    distance = dist(cluster_data,method = "euclidean")
    
    
    # The Dunn Index is the ratio of the smallest distance between observations 
    # not in the same cluster to the largest intra-cluster distance. 
    # The Dunn Index has a value between zero and infinity, and should be maximized. 
    # If the data set contains compact and well-separated clusters, the diameter of the clusters is expected to be small and the distance between the clusters is expected to be large. Thus, Dunn index should be maximized.
    
    print("dunn index")
    k_means_dunn_index = dunn(distance, kmeans_result$cluster)
    
    # Davies Bouldin index
    print("davies bouldin index")
    
    k_means_davies_bouldin_index = index.DB(cluster_data, kmeans_result$cluster, distance, centrotypes="medoids")
    k_means_davies_bouldin_index = k_means_davies_bouldin_index$DB
    # print(paste("Dunn index =", k_means_dunn_index))
    # print(paste("Davies Bouldin index : ", k_means_davies_bouldin_index))
    db_index_list = c(db_index_list, k_means_davies_bouldin_index)
    dunn_index_list = c(dunn_index_list, k_means_dunn_index)
    
    #Observations with a large s(i)(almost 1) are very well clustered, 
    # a small s(i) (around 0) means that the observation lies between two clusters, 
    # and observations with a negative  s(i)
    # are probably placed in the wrong cluster.
    print("silhouette index")
    
    silhouette_index = silhouette(kmeans_result$cluster, distance,)
    # print(silhouette_index)
    # plot(silhouette_index, cex.names=0.6)
    summary_silhoutte = summary(silhouette_index)
    # print(summary_silhoutte)
    silhouette_list = c(silhouette_list, summary_silhoutte$avg.width)
  }
  
  silhoutte_matrix<- rbind(silhoutte_matrix, silhouette_list)
  db_matrix<- rbind(db_matrix, db_index_list)
  dunn_matrix<- rbind(dunn_matrix, dunn_index_list)
}

silhoutte_list = colMeans(silhoutte_matrix)
dunn_index_list = colMeans(dunn_matrix)
db_index_list = colMeans(db_matrix)

plot(2:10, silhouette_list, type="b", xlab="K", ylab="Average Silhoutte", main=paste("Average Silhoutte and K Graph of ", DATASET_NAME))
plot(2:10, dunn_index_list, type="b",xlab="K", ylab="Dunn index", main=paste("Dunn index and K Graph of ", DATASET_NAME))
plot(2:10, db_index_list, type="b",xlab="K", ylab="DB Index", main=paste("Davies Bouldin index and K Graph of ", DATASET_NAME))

# Part 3
print("Part 3 - Analysis for Silhoutte index for HC clustering (Ward)...")
hc_silhouette_index = silhouette(tree_cut, distance)
print(hc_silhouette_index)
plot(hc_silhouette_index, cex.names=0.6, main=paste("Silhouette plot of ", DATASET_NAME))
hc_summary_silhoutte = summary(hc_silhouette_index)
print(hc_summary_silhoutte)



# Find the objects with negative silhouette
negative_indices <- which(hc_silhouette_index[, 'sil_width'] < 0)
print(negative_indices)
negative_objects = hc_silhouette_index[negative_indices, ,  drop = FALSE]

negative_objects = cbind(negative_indices, negative_objects)
# final_negative_matrix = matrix(negative_indices, hc_silhouette_index[negative_indices, ,  drop = FALSE], ncol=4)
print(negative_objects)