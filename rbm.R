#data 
movies_df <- read.csv("movies.csv")
names(movies_df) <- c("MovieID", "Title","genres")

movies_df$MovieID <- as.numeric(movies_df$MovieID)
movies_df$id_order <- 1:nrow(movies_df)

ratings_df <- read.csv("ratings.csv", stringsAsFactors = F)
colnames(ratings_df) <- c("UserID","MovieID","Rating","Timestamp")

merged_df <- merge(movies_df, ratings_df, by = "MovieID", all = FALSE)
merged_df[,c("Timestamp","Title","Genres")] <- NULL
merged_df$rating_per <- merged_df$Rating/5



#평점표 만들기 
Users_count <- 3000
Movies_count <- length(unique(movies_df$MovieID))
train_set <- matrix(0,nrow=Users_count,ncol=Movies_count)
for(i in 1:Users_count){
  Full_data_set <- merged_df[merged_df$UserID %in% i,]
  train_set[i,Full_data_set$id_order] <- Full_data_set$rating_per
}


library(tensorflow)
np <- import("numpy")
tf$reset_default_graph()
sess <- tf$InteractiveSession()

Hidden_count = 25 
num_input = nrow(movies_df)
vb <- tf$placeholder(tf$float32, shape = shape(num_input))    #Number of unique movies
hb <- tf$placeholder(tf$float32, shape = shape(Hidden_count))   #Number of features we're going to learn
W <- tf$placeholder(tf$float32, shape = shape(num_input, Hidden_count))

#Phase 1: Input Processing
v0 = tf$placeholder(tf$float32,shape= shape(NULL, num_input))
prob_h0= tf$nn$sigmoid(tf$matmul(v0, W) + hb)
h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0))))
#Phase 2: Reconstruction
prob_v1 = tf$nn$sigmoid(tf$matmul(h0, tf$transpose(W)) + vb) 
v1 = tf$nn$relu(tf$sign(prob_v1 - tf$random_uniform(tf$shape(prob_v1))))
h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)

# RBM Parameters and functions
#Learning rate
alpha = 1.2
#Create the gradients
w_pos_grad = tf$matmul(tf$transpose(v0), h0)
w_neg_grad = tf$matmul(tf$transpose(v1), h1)
#Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(v0)[1])
#Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf$reduce_mean(v0 - v1)
update_hb = hb + alpha * tf$reduce_mean(h0 - h1)

# Mean Absolute Error Function.
err = v0 - v1
err_sum = tf$reduce_mean(err * err)

# Initialise variables (current and previous)
cur_w = tf$Variable(tf$zeros(shape = shape(num_input, Hidden_count), dtype=tf$float32))
cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
cur_hb = tf$Variable(tf$zeros(shape = shape(Hidden_count), dtype=tf$float32))
prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, Hidden_count), stddev=0.01, dtype=tf$float32))
prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
prv_hb = tf$Variable(tf$zeros(shape = shape(Hidden_count), dtype=tf$float32)) 

# Start tensorflow session
sess$run(tf$global_variables_initializer())
output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(v0=train_set,
                                                                          W = prv_w$eval(),
                                                                          vb = prv_vb$eval(),
                                                                          hb = prv_hb$eval()))
prv_w <- output[[1]] 
prv_vb <- output[[2]]
prv_hb <-  output[[3]]
sess$run(err_sum, feed_dict=dict(v0=train_set, W= prv_w, vb= prv_vb, hb= prv_hb))

# Train RBM
epochs= 200
errors <- list()
weights <- list()

set.seed(1234)
for(ep in 1:epochs){
  for(i in seq(0,(dim(train_set)[1]-1500),1500)){
    batchX <- train_set[(i+1):(i+300),]
    output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(v0=batchX,
                                                                              W = prv_w,
                                                                              vb = prv_vb,
                                                                              hb = prv_hb))
    prv_w <- output[[1]] 
    prv_vb <- output[[2]]
    prv_hb <-  output[[3]]
    if(i%%1000 == 0){
      errors <- c(errors,sess$run(err_sum, feed_dict=dict(v0=batchX, W= prv_w, vb= prv_vb, hb= prv_hb)))
      weights <- c(weights,output[[1]])
      cat(i , " : ")
    }
  }
  cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
}

# Plot reconstruction error
error_vec <- unlist(errors)
plot(error_vec,xlab="배치 수",ylab="MSRE",main="재구성 평균제곱오차")
error_vec


# Recommendation
#Selecting the input user
inputUser = as.matrix(t(train_set[300,]))
names(inputUser) <- movies_df$id_order


# Plot the top genre movies
top_rated_movies <- movies_df[as.numeric(names(inputUser)[order(inputUser,decreasing = TRUE)]),]$Title
top_rated_genres <- movies_df[as.numeric(names(inputUser)[order(inputUser,decreasing = TRUE)]),]$genres





#Feeding in the user and reconstructing the input
hh0 = tf$nn$sigmoid(tf$matmul(v0, W) + hb)
vv1 = tf$nn$sigmoid(tf$matmul(hh0, tf$transpose(W)) + vb)
feed = sess$run(hh0, feed_dict=dict( v0= inputUser, W= prv_w, hb= prv_hb))
rec = sess$run(vv1, feed_dict=dict( hh0= feed, W= prv_w, vb= prv_vb))
names(rec) <- movies_df$id_order
top_recom_movies
# Select all recommended movies
top_recom_movies <- movies_df[as.numeric(names(rec)[order(rec,decreasing = TRUE)]),]$Title[1:10]
top_recom_genres <- movies_df[as.numeric(names(rec)[order(rec,decreasing = TRUE)]),]$genres
top_recom_movies
