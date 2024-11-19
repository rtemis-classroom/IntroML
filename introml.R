# IntroML Lab
# 2024-11-19
# EDG rtemis.org

# Packages ----
install.packages("rtemis", repos = c("https://egenn.r-universe.dev", "https://cloud.r-project.org"))
install.packages(c(
  "data.table", "R6", "plyr", "progressr", "future.apply",
  "glmnet", "rpart", "ranger", "lightgbm", "plotly"
))

# Load package
library(rtemis)

# Data ----
# Source: https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfInstances=between_1000_10000&qualities.NumberOfFeatures=gte_0&qualities.NumberOfClasses=lte_1&tags.tag=medicine&id=43672
# Read data
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

dat_lightrf <- train_cv(
  dat,
  alg = "lightrf"
)

## LightGBM ----
dat_lightgbm <- train_cv(
  dat,
  alg = "lightgbm"
)

## Present results ----
present(dat_lasso, dat_cart, dat_rf, dat_lightgbm)
describe(dat_lightgbm)
