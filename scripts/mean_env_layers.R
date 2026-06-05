library(terra)

tas7 <- rast("C:/Users/maria/Documents/PhD/data/climate_data/tas_2024_7.tif")
tas8 <- rast("C:/Users/maria/Documents/PhD/data/climate_data/tas_2024_8.tif")
tas9 <- rast("C:/Users/maria/Documents/PhD/data/climate_data/tas_2024_9.tif")

mean_tas <- (tas7 + tas8 + tas9) / 2

writeRaster(mean_tas, 
            "C:/Users/maria/Documents/Chapter1/bayesian_model/data/predictors/tas_mean.tif", 
            overwrite = TRUE)

pr7 <- rast("C:/Users/maria/Documents/PhD/data/climate_data/pr_2024_7.tif")
pr8 <- rast("C:/Users/maria/Documents/PhD/data/climate_data/pr_2024_8.tif")
pr9 <- rast("C:/Users/maria/Documents/PhD/data/climate_data/pr_2024_9.tif")

mean_pr <- (pr7 + pr8 + pr9) / 2

writeRaster(mean_pr, 
            "C:/Users/maria/Documents/Chapter1/bayesian_model/data/predictors/pr_mean.tif", 
            overwrite = TRUE)
