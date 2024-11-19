# IntroML Lab
# 2024-11-19
# EDG rtemis.org | Stathis.Gennatas@ucsf.edu

# Packages ----
install.packages("rtemis", repos = c("https://egenn.r-universe.dev", "https://cloud.r-project.org"))
install.packages(c(
  "data.table", "R6", "plyr", "progressr", "future.apply",
  "glmnet", "rpart", "ranger", "lightgbm", "plotly"
))
# If more dependencies are needed, the functions bellow will let you know

# Load package
library(rtemis)

# Data ----
# Source: https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfInstances=between_1000_10000&qualities.NumberOfFeatures=gte_0&qualities.NumberOfClasses=lte_1&tags.tag=medicine&id=43672
# Download and unzip
# Read data (change this to your local path)
dat <- read("~/icloud/Data/OpenML/43672_HeartDisease/dataset.arff")

# Preprocess ----
# Convert target to factor in-place
dat[, target := factor(target, levels = c(1, 0), labels = c("CVD", "Control"))]
check_data(dat)
str(dat)
head(dat)
nunique_perfeat(dat)
dat <- preprocess(dat, len2factor = 4)

# Model ----
## LASSO ----
# This will automatically tune the regularization parameter lambda.
dat_lasso <- train_cv(
  dat,
  alg = "glmnet",
  train.params = list(alpha = 1)
)

## CART ----
dat_cart <- train_cv(
  dat,
  alg = "cart"
)

## Random Forest ----
dat_rf <- train_cv(
  dat,
  alg = "ranger"
)

## LightGBM ----
# This automatically tunes the number of trees, which is very important.
# There are many more hyperparameters you can tune.
dat_lightgbm <- train_cv(
  dat,
  alg = "lightgbm"
)

## Present results ----
# Plot training and testing performance of all models
present(dat_lasso, dat_cart, dat_rf, dat_lightgbm)

# Describe the gradient boosting model
describe(dat_lightgbm)
