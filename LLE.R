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







library("MASS")
kn = 5

# mean center, scale with variance
dataset = scale(dataset, scale = DATA_SCALE, center=DATA_CENTER)
distance_matrix = dist(dataset, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)

library("lle")
lle_map = lle(dataset, m=2, k=kn)
print(lle_map)
plot(lle_map$Y, main=paste("LLE (2D) on ",DATASET_NAME," set (kn=",kn,")"), col=class_labels)
shepard_lle = Shepard(distance_matrix, lle_map$Y, 2)
plot(shepard_lle$x, shepard_lle$y, main=paste("Shepard diagram with LLE (2D) on ",DATASET_NAME," set (kn=",kn,")"))




# 3D Projection

lle_map3D = lle(dataset, m=3, k=kn)
print(lle_map3D)
shepard_lle3D = Shepard(distance_matrix, lle_map3D$Y, 2)
plot(shepard_lle3D$x, shepard_lle3D$y, main=paste("Shepard diagram with LLE (3D) on ",DATASET_NAME," set (kn=",kn,")"))



transformed_x = lle_map3D$Y[, 1]
transformed_y = lle_map3D$Y[, 2]
transformed_z = lle_map3D$Y[, 3]
theta = 20
library("plot3D")
library("rgl")
plot3d(transformed_x, transformed_y, transformed_z,  colvar = NULL, 
       bty='g',
       xlab = "X", 
       ylab = "Y",
       zlab = "Y",
       clab = "Type",
       theta =theta,
       phi = 20,
       pch = 20,
       size = 10, main=paste("LLE 3D projection of ",DATASET_NAME," set (kn=",kn,")"), col=label_colors)

print("Succesfull.........")
