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

# 2D projection

# mean center, scale with variance
dataset = scale(dataset, scale = DATA_SCALE, center=DATA_CENTER)


distance_matrix = dist(dataset, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)
# print(distance_matrix)
# torgerson mds
# classical_mds = cmdscale(distance_matrix, k = 2, eig = FALSE, add = FALSE, x.ret = FALSE)
# print(classical_mds)

# plot(classical_mds, main=paste("Torgerson MDS (2D) for ",DATASET_NAME," set"), col=class_labels)
library("MASS")
# sammond metric mds

isomap_start= prcomp(dataset, center = FALSE, scale = FALSE, retx=TRUE)
starting_options = list(cmdscale(distance_matrix, k=2), matrix(rnorm(nrow(dataset)*2), ncol=2), 
                        as.matrix(isomap_start$x[, 1:2]))
starting_option_labels = list("Torgerson MDS", "Randomized", "PCA")

best_2d_config = NULL
best_2d_index = NULL
min_error = 100000000
best_config_learning = NULL
best_config_steps = NULL
for(start_config_index in 1:1)
{
  trace_result =  capture.output (sammon(distance_matrix, y = as.matrix(starting_options[[start_config_index]]), k = 2, niter = 100,
                                         magic = 0.05, tol = 1e-4, trace = TRUE))
    
  result = sammon(distance_matrix, y = as.matrix(starting_options[[start_config_index]]), k = 2, niter = 100,
                  magic = 0.05, tol = 1e-4, trace = TRUE)
  
  print(paste("Config ", start_config_index," : ", starting_option_labels[[start_config_index]], " Error : ", result$stress))
  
  # print(trace_result[1:5])
  iteration_steps = c(0)
  vec = c()
  l = trace_result[1]  
  index1 = regexpr(':', l)
  index2 = nchar(l)
  val1 = substr(l, start=index1+1, stop=index2-1)
  # print(val1)
  val1 = as.double(val1)
  vec = c(vec, val1)
  
  for(i in 2:length(trace_result))
  {
    l = trace_result[i]  
    # print(paste("Trace ", i, " ", l))
    if(regexpr('points', l) != -1)
    {
      break
    }
    
    index1 = regexpr(':', l)
    
    if(regexpr("magic", l) != -1)
    {
      index2 = regexpr(',', l)
      
    }
    else
    {
      index2 = nchar(l)
    }
    
    val1 = substr(l, start=index1+2, stop=index2-1)
    # print(paste("val1 : ", val1))
    val1 = as.double(val1)
    # print(val1)
    vec = c(vec, val1)
    
    s = regexpr('after', l)
    m = regexpr('iters:', l)
    it = substr(l, start=s+6, stop=m-1)
    iteration_steps = c(iteration_steps, as.integer(it))
    
  }
  # print(trace_result)
  if(result$stress < min_error)
  {
    min_error = result$stress
    best_2d_config = result
    best_2d_index = start_config_index
    best_config_learning = vec
    best_config_steps = iteration_steps
  }
}
print(best_config_learning)
plot(best_config_steps, best_config_learning, type="b", ylab = "Streess", xlab="Iteration", main=paste("SAMMON MDS (2D) Best with ",starting_option_labels[[best_2d_index]], " on ",DATASET_NAME," set"))

print(paste("Best index 2D : ", best_2d_index))
print(paste("Res : ", starting_option_labels[[best_2d_index]]))
# print(best_2d_config)
plot(best_2d_config$points, main=paste("SAMMON MDS (2D) for ",DATASET_NAME," set with ",starting_option_labels[[best_2d_index]], " start"), col=class_labels)
# print(result)
shepard_sammon_mds = Shepard(distance_matrix, best_2d_config$points, 2)

plot(shepard_sammon_mds$x, shepard_sammon_mds$y, main=paste("Shepard diagram with SAMMON MDS (2D) on ",DATASET_NAME," set with ",starting_option_labels[[best_2d_index]], " start"))

# 3d projection 


isomap_start3D = prcomp(dataset, center = FALSE, scale = FALSE, retx=TRUE)

starting_options3D = list(cmdscale(distance_matrix, k=3), matrix(rnorm(nrow(dataset)*3), ncol=3), 
                          isomap_start3D$x[, 1:3])
starting_option_labels3D = list("Torgerson MDS", "Randomized", "PCA")




min_error = 10000000

best_config_learning = NULL
best_config_steps = NULL
for(start_config_index in 1:1)
{
  
  iteration_steps = c()
  
  trace_result = capture.output(sammon(distance_matrix, y = starting_options3D[[start_config_index]], k = 3, niter = 100,
                                       magic = 0.05, tol = 1e-4, trace = TRUE))
  
  result3D = sammon(distance_matrix, y = starting_options3D[[start_config_index]], k = 3, niter = 100,
                    magic = 0.05, tol = 1e-4, trace = TRUE)
  print(paste("Config ",start_config_index," : ", starting_option_labels3D[[start_config_index]], " Error : ", result3D$stress))
  
  iteration_steps = c(iteration_steps, 0)
  
  vec = c()
  l = trace_result[1]  
  index1 = regexpr(':', l)
  index2 = nchar(l)
  val1 = substr(l, start=index1+1, stop=index2-1)
  # print(val1)
  val1 = as.double(val1)
  vec = c(vec, val1)
  for(i in 2:length(trace_result))
  {
    l = trace_result[i]  
    if(regexpr('points', l) != -1)
    {
      break
    }
    index1 = regexpr(':', l)
    if(regexpr("magic", l) != -1)
    {
      index2 = regexpr(',', l)
    }
    else
    {
      index2 = nchar(l)
    }
    val1 = substr(l, start=index1+1, stop=index2-1)
    # print(val1)
    val1 = as.double(val1)
    # print(val1)
    vec = c(vec, val1)
    s = regexpr('after', l)
    m = regexpr('iters:', l)
    it = substr(l, start=s+6, stop=m-1)
    iteration_steps = c(iteration_steps, as.integer(it))
  }
  if(result3D$stress < min_error)
  {
    min_error = result3D$stress
    best_3d_config = result3D
    best_3d_index = start_config_index
    best_config_learning = vec
    best_config_steps = iteration_steps
  }
}
print(best_config_learning)
plot(best_config_steps, best_config_learning, type="b", ylab = "Streess", xlab="Iteration", main=paste("SAMMON MDS (3D) Best with ",starting_option_labels[[best_3d_index]], " on ",DATASET_NAME," set"))

print(paste("Best index 3D : ", best_3d_index))
# print(sammon_mds3D)
shepard_sammon_mds3D = Shepard(distance_matrix, best_3d_config$points, 2)

plot(shepard_sammon_mds3D$x, shepard_sammon_mds3D$y, main=paste("Shepard diagram with SAMMON MDS (3D) on ",DATASET_NAME," set with ",starting_option_labels3D[[best_3d_index]], " start"))
transformed_x = best_3d_config$points[, 1]
transformed_y = best_3d_config$points[, 2]
transformed_z = best_3d_config$points[, 3]
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
       size = 10, main=paste("MDS (SAMMON) 3D projection of ",DATASET_NAME," set with ",starting_option_labels3D[[best_3d_index]], " start"), col=label_colors)
print("Successfull.....")
