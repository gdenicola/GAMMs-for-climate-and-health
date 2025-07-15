##Generalized Additive (Mixed) Models tutorial for Climate-Health applications
##By Giacomo De Nicola

# A practical tutorial for Generalized Additive (Mixed) Models with the mgcv package.
# This script covers simple GAMs, GAMs with binomial data, and a full GAMM
# with random effects, a smooth time trend, and linear covariates.

# Load the key packages
library(mgcv)
library(ggplot2)

# Set a seed for reproducibility
set.seed(42)

################################################################################
# Part 1: A Simple GAM - Non-linear Temperature Effect (as before)
################################################################################

# --- 1a. Create toy data ---
n <- 200
temp <- runif(n, 0, 35) 
pollution_true_effect <- -0.1 * (temp - 18)^2 + 30
noise <- rnorm(n, 0, 5)
pollution <- pollution_true_effect + noise
df <- data.frame(temp, pollution)
#View(df)

# This scatter plot helps visualize the raw relationship before we fit a model.
ggplot(df, aes(x = temp, y = pollution)) + 
  geom_point(alpha = 0.6, color = "dodgerblue") +  # Added color for clarity
  theme_minimal() +
  labs(
    title = "Raw Data: Pollution vs. Temperature",
    subtitle = "This is the pattern the GAM will try to capture.",
    x = "Temperature (°C)",
    y = "Pollution Level"
  )

# --- 1b. Fit and plot the simple GAM ---
model1 <- gam(pollution ~ s(temp, bs = "ps", k = 15), data = df)
summary(model1)
plot(model1, shade = TRUE, shade.col = "lightblue")


################################################################################
# Part 2: A Generalized GAM (Binomial)
################################################################################

# --- 2a. Create toy data ---
n_binom <- 300
soil_moisture <- runif(n_binom, 0, 1)
prob_outbreak <- plogis(-2 + 7 * soil_moisture - 5 * soil_moisture^2)
malaria_event <- rbinom(n_binom, 1, prob_outbreak)
df_malaria <- data.frame(soil_moisture, malaria_event)
#View(df_malaria)

# Data creation is the same as above
# library(ggplot2)
# df_malaria <- ...

# --- Plot the proportions within bins ---
ggplot(df_malaria, aes(x = soil_moisture, y = malaria_event)) +
  stat_summary_bin(fun = "mean", geom = "bar", bins = 15, fill = "darkgreen", alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "Proportion of Malaria Events vs. Soil Moisture",
    subtitle = "Data grouped into 15 bins.",
    x = "Soil Moisture",
    y = "Proportion of Cases with Malaria Event"
  )

# --- 2b. Fit the binomial GAM ---
model2 <- gam(malaria_event ~ s(soil_moisture, bs = "ps", k = 10), 
              data = df_malaria, 
              family = binomial)
summary(model2)
plot(model2, shade = TRUE, shade.col = 'lightgreen')


################################################################################
# Part 3: Adding Random Effects (GAMM)
################################################################################

# --- 3a. Add a grouping variable (clinic ID) ---
num_clinics <- 50
df_malaria$clinic_id <- as.factor(rep(1:num_clinics, each = n_binom / num_clinics))
clinic_effects <- rnorm(num_clinics, 0, 1.5)
df_malaria$logit_prob_with_re <- -2 + 7 * df_malaria$soil_moisture + clinic_effects[df_malaria$clinic_id]
df_malaria$malaria_event_re <- rbinom(n_binom, 1, plogis(df_malaria$logit_prob_with_re))

# --- 3b. Fit the GAMM ---
model3_gamm <- bam(malaria_event_re ~ s(soil_moisture, bs = "ps") + 
                     s(clinic_id, bs = "re"),
                   data = df_malaria,
                   family = binomial,
                   discrete = TRUE)
summary(model3_gamm)
plot(model3_gamm, pages = 1, shade = TRUE)


################################################################################
# Part 4: The Full Picture - A GAMM with Smooth Time, Linear Covariates, 
#         and Random Effects
################################################################################
# This example models malaria *counts* instead of just presence or absence.

# --- 4a. Create more complex, realistic panel data ---
n_clinics <- 50
n_months <- 48 # 4 years of data
total_obs <- n_clinics * n_months

# Create the basic structure
full_df <- data.frame(
  clinic_id = as.factor(rep(1:n_clinics, each = n_months)),
  time = rep(1:n_months, times = n_clinics)
)

# Create covariates for each clinic (these don't change over time)
clinic_data <- data.frame(
  clinic_id = as.factor(1:n_clinics),
  # A linear predictor: e.g., socio-economic status of the clinic's area
  wealth_index = runif(n_clinics, -2, 2),
  # Population for each clinic to use in the offset
  population = round(runif(n_clinics, 500, 3000))
)

# Merge them together
full_df <- merge(full_df, clinic_data, by = "clinic_id")

# Create a time-varying covariate (soil moisture with seasonality)
full_df$soil_moisture <- sin(2 * pi * full_df$time / 12) + rnorm(total_obs, 0, 0.5)
View(full_df)
# --- 4b. Define the "True" relationship to generate data ---
# This is the "secret" model we are trying to recover.

# Define random intercepts for each clinic
clinic_re <- rnorm(n_clinics, mean = 0, sd = 0.5)

# The log of the expected rate is a sum of linear and smooth effects
log_rate <- -3 +  # Intercept
  (0.04 * full_df$time - 0.0005 * full_df$time^2) + # A smooth, non-linear time trend
  (1.2 * full_df$soil_moisture) + # A smooth effect of moisture
  (-0.4 * full_df$wealth_index) + # A linear effect of wealth
  clinic_re[full_df$clinic_id] # The random effect for the clinic

# Calculate expected counts = rate * population
# log(counts) = log(rate) + log(population)
# The offset is log(population)
expected_counts <- exp(log_rate) * full_df$population

# Generate final counts using a negative binomial distribution (handles overdispersion)
full_df$malaria_cases <- rnbinom(total_obs, mu = expected_counts, size = 5)


# --- 4c. Fit the full, realistic GAMM ---
# This model includes all the different component types.

# Note the formula structure:
# - s(time, ...): smooth term for the non-linear trend
# - s(soil_moisture, ...): smooth term for the environmental driver
# - wealth_index: a standard linear term (NO 's()' wrapper)
# - s(clinic_id, ...): the random effect term
# - offset(log(population)): accounts for population size differences
model4_full <- bam(malaria_cases ~ 
                     s(time, bs = "ps", k = 20) + 
                     s(soil_moisture, bs = "ps", k = 10) +
                     wealth_index +
                     s(clinic_id, bs = "re") +
                     offset(log(population)),
                   data = full_df,
                   family = nb(), # Negative Binomial for overdispersed counts
                   discrete = TRUE)

# --- 4d. Interpret the comprehensive model output ---
summary(model4_full)
# You will now see two distinct sections in the output:
#
# 1. PARAMETRIC COEFFICIENTS:
#    This section is for your LINEAR terms. Here you will find 'wealth_index'.
#    The 'Estimate' is the standard β coefficient. You interpret it
#    just like in a regular GLM: "For a one-unit increase in wealth_index,
#    the log of the expected malaria count decreases by 0.38, holding all
#    other variables constant."
#
# 2. APPROXIMATE SIGNIFICANCE OF SMOOTHS:
#    This section is for ALL terms wrapped in s(). You will see s(time),
#    s(soil_moisture), and the random effect s(clinic_id). You interpret
#    these visually using the plots.

# --- 4e. Plot the effects from the full model ---
plot(model4_full, shade = TRUE)

# How to read the plots:
# - s(time): Shows the long-term trend in malaria risk after accounting for
#   everything else. We can see it increases and then levels off.
# - s(soil_moisture): Shows the non-linear relationship with moisture.
# - s(clinic_id): Shows the estimated random intercepts for each clinic.
#   Some clinics have consistently higher or lower risk than the average.
#   Note that these clinic-specific random effects are not interesting to
#   interpret, but it is important to control for them to prevent bias in the
#   estimates. 
# - There is NO PLOT for 'wealth_index' because its effect is assumed to be
#   strictly linear and is fully described by its single coefficient in the summary.