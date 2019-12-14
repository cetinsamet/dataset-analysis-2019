#load letter-recognition data
library(datasets)
DATASET_NAME = "LETTERRECOGNITION"
DATA_CENTER = TRUE
DATA_SCALE = TRUE
# wellseparated2D , wellseparated3D , wellseparated,   
# notseparable2D , notseparable3D , notseparable
# noisy2D , noisy3D, noisy
# wine , swissroll
if(DATASET_NAME == "LETTERRECOGNITION")
{
  
  data <- read.table("datasets/letter-recognition.data", sep=',')
  
  #set num of rows and cols in data
  n_row <- nrow(data) 
  n_col <- ncol(data) 
  
  x <- as.matrix(data[,2:n_col]) 	# features
  labels <- as.matrix(data[,1]) 
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


# print(summary(dataset))
data_pca = prcomp(dataset, center = DATA_CENTER, scale = DATA_SCALE, retx=TRUE)
print(paste("PCA is being applied on ",DATASET_NAME))
# number of samples 
data_count = nrow(dataset)
# column means
col_mean = colMeans(dataset)
print(paste("Col Mean of ",DATASET_NAME, " set"))
print(col_mean)
# install.packages("Rfast")
library("Rfast")
# list of each column's standard deviation...
col_stds = sqrt(colVars(dataset))

# print(col_mean)
# print("Data feature variance")
# print(colVars(dataset))
print(paste("Data feature standard deviation of ",DATASET_NAME, " set"))
print(col_stds)

# data is centered...
print(paste(DATASET_NAME," Raw data\n"))
print(dataset[1:5,])
centered_data_matrix = dataset

if(DATA_CENTER == TRUE)
{
  centered_data_matrix = sweep(dataset, 2, col_mean, FUN = "-")
  col_mean = colMeans(centered_data_matrix)
}
if(DATA_SCALE == TRUE)
{
  centered_data_matrix = sweep(centered_data_matrix, 2, col_stds, FUN="/")
}





# print(scale(dataset, center=TRUE, scale=TRUE))
pca_distance = dist(centered_data_matrix)
# covariace matrix = xT*x*/(1/N)
covariance = t(centered_data_matrix) %*% centered_data_matrix / data_count
# print(covariance)
# eigen vectors and values of covariance matrix...
eigen_decompisition = eigen(covariance)
eigen_values = eigen_decompisition$values
eigen_vectors = eigen_decompisition$vectors
eigen_values_sum = sum(eigen_values)

print(paste("Eigen Values of ",DATASET_NAME," set"))
print(eigen_values)

print(paste("Eigen Vectors of ",DATASET_NAME," set"))
print(eigen_vectors)




# scree plot of eigenvalues..
plot(eigen_values, type="b", xaxp=c(1, ncol(dataset), 1), ylab = "Eigen values", xlab="Eigenvalue index", main=paste("Scree Plot of Eigen Values on ",DATASET_NAME," set"))
# proportion of variance graph...
plot(cumsum(eigen_values/eigen_values_sum), type="b", xlab="Eigenvalue index", ylab="Proportion of variance", main=paste("Proportion of Variance on ",DATASET_NAME," set"))

# for projecting data features into new space...
# each feature corresponds to a dimension in original space...
basis_vectors <- data.frame()
basis_vector_names = colnames(dataset)
for(i in 1:ncol(dataset))
{
  vec =  rep(0, ncol(dataset))
  vec[i] = 1.0
  basis_vectors <- rbind(basis_vectors, c(vec))
}
basis_vectors = as.matrix(basis_vectors)

checkMatch <- function(matrix1, matrix2)
{
  eps = 0.0000001
  for(y in 1:nrow(matrix1))
  {
    for(x in 1:ncol(matrix1))
    {
      if((matrix1[y, x]-eps<matrix2[y, x]) && (matrix2[y, x] < matrix1[y, x]+eps))
      {
        
      }
      else return(FALSE)
    }
  }
  return(TRUE)
}
# svd and eigen eigenvector results does not match multiple -1
projection_matrix2D <- eigen_decompisition$vectors[, 1:2]
if(ncol(eigen_decompisition$vectors) >=3 )
{
  projection_matrix3D <- eigen_decompisition$vectors[, 1:3]
}
projection_matrix2D = as.matrix(projection_matrix2D)
pca_projection_matrix = as.matrix(data_pca$rotation[, 1:2])
if(checkMatch(projection_matrix2D, pca_projection_matrix) == FALSE)
{
  
  print("eigen and svd results do not match svd results will be used... ")
  projection_matrix2D = as.matrix(data_pca$rotation[, 1:2])
  if(ncol(eigen_decompisition$vectors) >=3 )
  {
    projection_matrix3D = as.matrix(data_pca$rotation[, 1:3])
  }
}


library("MASS")
# PCA Projections...
projected_data2D = centered_data_matrix %*% projection_matrix2D

if(ncol(eigen_decompisition$vectors) >=3 )
{
  projected_data3D = centered_data_matrix %*% projection_matrix3D
}

mean_projected_data2D = colMeans(projected_data2D)
if(ncol(eigen_decompisition$vectors) >=3 )
{
  mean_projected_data3D = colMeans(projected_data3D)
}
projected_data_covariance2D = colVars(projected_data2D)
# 2D projection
projected_basis_vectors2D = basis_vectors %*% projection_matrix2D

plot(data_pca$x[, 1:2], main=paste("Library Projected Data of ",DATASET_NAME, " set"), col=class_labels)
print(data_pca$x[, 1:2])

for (i in 1:nrow(projected_basis_vectors2D) )
{
  x = projected_basis_vectors2D[i, 1]*eigen_values[i]*4
  
  y = projected_basis_vectors2D[i, 2]*eigen_values[i]*4
  # print(x)
  # print(y)
  
  arrows(mean_projected_data2D[1],mean_projected_data2D[2], x, y)
  text(x= x, y=y, label=basis_vector_names[i])
}
print(projected_basis_vectors2D)

# 3d projection
if(ncol(eigen_decompisition$vectors) >=3 )
{
  projected_basis_vectors3D = basis_vectors %*% projection_matrix3D
  library("plot3D")
  library("rgl")
  transformed_x = projected_data3D[, 1]
  transformed_y = projected_data3D[, 2]
  transformed_z = projected_data3D[, 3]
  
  # increase the magnitudes of vectors for better visualization
  x1 = projected_basis_vectors3D[, 1]*10
  y1 = projected_basis_vectors3D[, 2]*10
  z1 = projected_basis_vectors3D[, 3]*10
  
  theta = 20
  
  plot3d(transformed_x, transformed_y, transformed_z,  colvar = NULL, 
         bty='g',
         xlab = "PCA1", 
         ylab = "PCA2",
         zlab = "PCA3",
         clab = "Type",
         theta =theta,
         phi = 20,
         pch = 20,
         size = 10, main=paste("PCA 3D projection of ",DATASET_NAME," set"), col=label_colors)
  
  for(i in 1:nrow(projected_basis_vectors3D))
  {
    arrow3d(p0=c(mean_projected_data3D[1],mean_projected_data3D[2],
                 mean_projected_data3D[3]), p1=c(x1[i], y1[i], z1[i]))
  }
  text3d(x1, y1, z1, basis_vector_names,
         colvar = x1^2, colkey = FALSE, add=TRUE, plot=FALSE)
}
print("Successfull.....")