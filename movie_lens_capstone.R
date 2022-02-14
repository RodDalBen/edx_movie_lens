# 'PH125.9x Data Science: MovieLens Capstone Project'
# author: "Rodrigo Dal Ben de Souza"
# date: "December 28, 2021 - Last update: February 13, 2022"

#Here we install packages to be used during the project.  

# R version: 4.1.2
# packages - version
install.packages("tidyverse") # v1.3.1
install.packages("patchwork") # v1.1.1
install.packages("data.table") # v1.14.2
install.packages("caret") # v6.0-90
install.packages("here") # v1.0.1

# Here we load the packages, create a vector with color blind friendly palette (`color_blind_colors`), and save our target RMSE. 
library(tidyverse)
library(patchwork)
library(data.table)
library(caret)
library(here)
color_blind_colors <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
rmse_target <- 0.86490

## Loading dataset
# Here we use the code provided in the course to load and prepare the MovieLens 10M dataset.

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- caret::createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- 
  temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Exploration & Visualization
# Here we explore our data with summary statistics and visualizations. The Training subset is stored under the `edx` dataframe, the Test subset is stored under the `validation` dataframe. We will preserve our Test subset (`validation`) for model evaluation. So most summary statistics and visualizations will be made with the Training (`edx`) subset. 
# unique movies & ratings
summ_movies <- 
  edx %>% 
  group_by(title) %>% 
  summarise(n_ratings = n(),
            m_ratings = mean(rating),
            sd_ratings = sd(rating)
  ) %>% 
  arrange(desc(n_ratings))

# unique users & ratings
summ_users <- 
  edx %>% 
  group_by(userId) %>% 
  summarise(n_ratings = n(),
            m_ratings = mean(rating),
            sd_ratings = sd(rating)
  ) %>% 
  arrange(desc(n_ratings))

# number of ratings distributions
edx %>% 
  ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.5, 
                 fill = color_blind_colors[4], 
                 color = "black",
                 alpha = 0.8) +
  labs(title = "Raw ratings",
       x = "Number of stars (0-5)", 
       y = "Number of movies") +
  theme_bw()

# ratings distributions by movie
hist_summary_ratings <- 
  summ_movies %>% 
  ggplot(aes(x = m_ratings)) +
  geom_histogram(binwidth = 0.1, color = "grey", size = 0, alpha = 0.9) +
  geom_vline(xintercept = mean(summ_movies$m_ratings), color = color_blind_colors[2]) +
  geom_vline(xintercept = median(summ_movies$m_ratings), color = color_blind_colors[4]) +
  labs(title = "By movie",
       x = "Mean Number of stars (0-5)",
       y = "Number of movies") +
  theme_bw()

# ratings distributions by user
hist_user_ratings <- 
  summ_users %>% 
  ggplot(aes(x = n_ratings)) +
  geom_histogram(bins = 50, 
                 fill = color_blind_colors[4], 
                 color = "black") +
  scale_x_log10() +
  labs(title = "By user",
       x = "Log number of ratings",
       y = "Number of users") +
  theme_bw()

hist_summary_ratings + hist_user_ratings + plot_annotation(tag_levels = 'A', title = "Ratings distributions")

# ratings by movie  
ratings_movies <- 
  summ_movies %>% 
  ggplot(aes(x = n_ratings, y = m_ratings)) +
  geom_point(alpha = 0.2) +
  geom_smooth(alpha = 0.5, color = color_blind_colors[2]) +
  labs(title = "Number of ratings vs. average ratings by movies",
       x = "Number of ratings", 
       y = "Average rating") +
  theme_bw()

ratings_users <- 
  summ_users %>% 
  ggplot(aes(x = n_ratings, y = m_ratings)) +
  geom_point(alpha = 0.2) +
  geom_smooth(alpha = 0.5, color = color_blind_colors[2]) +
  labs(title = "Number of ratings vs. average ratings by users",
       x = "Number of ratings", 
       y = "Average rating") +
  theme_bw()

ratings_movies / ratings_users

# Results
  
# Here we *fit* the models and *evaluate* the fit using RMSE for each model. We will add the RMSE for each model in a table to facilitate comparison across models.

## Model 01: Baseline
# calculate the mean (Training subset)
mu <- mean(edx$rating)

# calculate the RMSE (Test subset)
rmse_baseline <- caret::RMSE(validation$rating, mu)

# create table with RMSE
rmse_scores <- tibble(Model = "Baseline", 
                      RMSE = rmse_baseline,
                      `Below Target RMSE` = if_else(RMSE < rmse_target, "Yes", "No"))

rmse_scores %>% knitr::kable()

## Model 02: User effects  
# calculate average deviance for each user
user_avg <- 
  edx %>% 
  group_by(userId) %>% 
  summarise(b_u = mean(rating - mu)) 

# predict rating
pred_rating <- 
  mu + validation %>% 
  left_join(user_avg, by = "userId") %>% 
  pull(b_u)

# save rmse
rmse_user_effect <- caret::RMSE(pred_rating, validation$rating)
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "User effects",
                                RMSE = rmse_user_effect, 
                                `Below Target RMSE` = if_else(RMSE < rmse_target, "Yes", "No")))

rmse_scores %>% knitr::kable()

## Model 03: Movie effects
# calculate average deviance for each user
movie_avg <- 
  edx %>% 
  group_by(movieId) %>% 
  summarise(b_m = mean(rating - mu)) 

# predict rating
pred_rating <- 
  mu + validation %>% 
  left_join(movie_avg, by = "movieId") %>% 
  pull(b_m)

# save rmse
rmse_movie_effect <- caret::RMSE(pred_rating, validation$rating)
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "Movie effects",
                                RMSE = rmse_movie_effect,
                                `Below Target RMSE` = if_else(RMSE < rmse_target, "Yes", "No")))

rmse_scores %>% knitr::kable()

## Model 04: Movie and User effects
# predict rating
pred_rating <- 
  validation %>% 
  left_join(movie_avg, by = "movieId") %>% 
  left_join(user_avg, by = "userId") %>% 
  mutate(pred = mu + b_m + b_u) %>% 
  pull(pred)

# save rmse
rmse_user_movie_effect <- caret::RMSE(pred_rating, validation$rating)
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "User and movie effects",
                                RMSE = rmse_user_movie_effect,
                                `Below Target RMSE` = if_else(RMSE < rmse_target, "Yes", "No")))

rmse_scores %>% knitr::kable()

## Model 05: Regularization dealing with bias
# create lambda
lambdas <- seq(0, 10, 0.25)

# find lambda that minimizes errors
rmses <- sapply(lambdas, function(l){

  # mu already calculated
  
  # calculate user effects
  b_movie <- 
    edx %>%
    group_by(movieId) %>%
    summarise(b_movie = sum(rating - mu)/(n()+l))
  
  # calculate movie effects
  b_user <- 
    edx %>%
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarise(b_user = sum(rating - b_movie - mu)/(n()+l))
  
  # predict ratings using the test dataset
  predicted_ratings <- 
    validation %>% 
    left_join(b_user, by = "userId") %>% 
    left_join(b_movie, by = "movieId") %>% 
    mutate(pred = mu + b_user + b_movie) %>% 
    pull(pred)
  
  # return rmses based on lambdas
  return(caret::RMSE(predicted_ratings, validation$rating)) 
  })

# plot
qplot(x = lambdas, y = rmses) + theme_bw()

# find min rmse and best tunning parameter
min_rmse <- min(rmses)
lambda <- lambdas[which.min(rmses)]

# save rmse
rmse_scores <- bind_rows(rmse_scores, 
                         tibble(Model = "User and movie effects regularized",
                                RMSE = min_rmse,
                                `Below Target RMSE` = if_else(RMSE < rmse_target, "Yes", "No"))
                         )

rmse_scores %>% knitr::kable()

# THE END
