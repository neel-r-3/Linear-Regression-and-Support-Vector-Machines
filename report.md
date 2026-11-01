# Q1. Bike Rental Demand Prediction with Linear Regression

All code is in [`q1.ipynb`](q1.ipynb). Plots are in the [`figures/`](figures/) folder.
To run it, open the notebook from inside the `q1/` folder and run all cells. It reads `data/hour.csv` and `data/bike_temp_comfort_zone.csv`.

No library linear regression is used. I only use NumPy for matrix operations (matrix multiply, solving a linear system, and `lstsq` where the task allows it).

---

## Data and split

- Dataset: `hour.csv` from the UCI Bike Sharing Dataset. It has 17,379 rows, one per hour.
- Target: `cnt`, the number of rentals in that hour.
- I never use `casual` or `registered`, because they add up to `cnt`.
- `weathersit = 4` (heavy rain or snow) appears in only 3 rows. A column for it would be almost all zeros, so I merged it into `weathersit = 3` (light rain or snow).

**Split:** random 80 : 10 : 10, with random seed **42**.

**What a random split means here:** the data is a time series, so a random split puts most validation hours right next to training hours from the same day. Because of this, the validation error shows how well the model fills in missing hours, not how well it predicts the future, so it is a bit too optimistic.

---

## 1.1 Feature Engineering

### Problem with plain integers

A linear model treats an integer column as "each step up adds the same amount". That is wrong for these columns:

| Column | Why a plain integer is wrong | What I did |
|---|---|---|
| `hr` | Rentals do not go steadily up or down during the day. There are two rush-hour peaks on working days. Also 23:00 and 00:00 are neighbours, but the numbers are far apart. | One-hot (23 columns) |
| `mnth` | Rentals rise in spring, stay high in summer and autumn, and drop in winter. December and January are neighbours. | One-hot (11 columns) |
| `weekday` | Days are labels, not amounts. Sunday (0) and Saturday (6) are both weekend days but far apart as numbers. | One-hot (6 columns) |
| `season` | 4 labels with no real order. | One-hot (3 columns) |
| `weathersit` | Worse weather means fewer riders, but the drop from "clear" to "mist" is not the same size as from "mist" to "rain". | One-hot (2 columns) |

In every one-hot encoding I drop the first level. The intercept already covers that level, and keeping all levels would make the columns add up to the intercept column (see 1.3.1).

Other columns:

- `yr`, `holiday`: already 0/1, kept as they are.
- `workingday`: **not used as its own column**. It is exactly "weekday is Monday to Friday, and not a holiday", so it can be built from the weekday columns and `holiday`, and adds nothing new. I checked this on every row. I still use it inside interactions.
- `temp`, `hum`, `windspeed`: kept as numbers and standardized (see 1.2.2).
- `atemp`: **dropped**. It is the "feels like" temperature, and its correlation with `temp` is 0.988, so it is almost a copy.

### The five interaction terms

An interaction lets the effect of one feature change depending on another feature.

1. **weathersit × hr**: rain removes more riders at 17:00, when many people would ride, than at 03:00, when almost nobody rides.
2. **temp × hr**: temperature matters most for leisure rides (midday, evening). Commuters ride at rush hour anyway.
3. **hum × hr**: high humidity is most unpleasant in hot afternoons and matters little at night.
4. **workingday × weathersit**: people going to work must travel even in bad weather, while weekend riders can stay home.
5. **workingday × hr**: working days and non-working days have a completely different daily shape. On working days there are peaks at about 08:00 (~480 rentals) and 17:00 (~525). On weekends there is one wide peak around midday (~370). This is day type × time of day, not weather. I first tried weekday × season here, but it gave no improvement, so I replaced it.

### Raw hour vs one-hot hour

Two models, the same in every way except how hour is stored. In the raw model, the interactions with hour also use the raw number.

| Hour encoding | Columns | Train RMSE | **Validation RMSE** |
|---|---|---|---|
| `hr` as one integer | 36 | 137.82 | **136.39** |
| `hr` one-hot | 168 | 68.81 | **68.56** |

**What the gap tells us:** with `hr` as one number, the model must assume that rentals change by the same amount every hour, so the daily pattern can only be a straight line going up or down. It also forces working days and weekends to differ only by a straight-line tilt. That cannot show the morning and evening peaks or the low at night. One-hot removes this assumption and cuts validation RMSE in half.

### How much each interaction helps

| Model | Columns | Train RMSE | Validation RMSE |
|---|---|---|---|
| All five interactions | 168 | 68.81 | 68.56 |
| No interactions | 51 | 102.18 | 100.59 |
| Without weathersit × hr | 122 | 69.98 | 68.90 |
| Without temp × hr | 145 | 75.48 | 75.06 |
| Without hum × hr | 145 | 68.94 | 68.72 |
| Without workingday × weathersit | 166 | 68.81 | 68.54 |
| Without workingday × hr | 145 | 95.22 | 93.83 |
| Only temp × hr and workingday × hr | 97 | 70.49 | 69.76 |

- workingday × hr helps the most, followed by temp × hr.
- weathersit × hr and hum × hr help a little.
- workingday × weathersit does not really help, but it only costs 2 columns, so I kept it.

### Final design matrix

**168 columns** (167 features + 1 intercept), with 13,903 training rows.

---

## 1.2 Dataset Preprocessing and Analysis

### 1.2.1 Train/val/test split

| Set | Rows |
|---|---|
| Train | **13,903** |
| Validation | **1,737** |
| Test | **1,739** |

### 1.2.2 Feature normalization

**Standardization:** for `temp`, `hum` and `windspeed` I use z = (x − mean) / std. The mean and std come from the **training set only**, and I reuse them for validation and test. Interactions are built from the standardized values. The 0/1 columns are left as they are.

| Feature | Train mean (as given) | Train std (as given) | After standardizing |
|---|---|---|---|
| temp | 0.498 | 0.192 | mean 0, std 1 |
| hum | 0.627 | 0.193 | mean 0, std 1 |
| windspeed | 0.190 | 0.123 | mean 0, std 1 |

The dataset already gives these columns scaled into [0, 1]. So "without standardization" means using them as given: small numbers, but not centred at 0.

**Gradient descent setup:**

- Loss is the training MSE, (1/n)·‖Xw − y‖², with gradient (2/n)·Xᵀ(Xw − y).
- It starts from w = 0 and uses all training rows in every step.
- **Stable** means the loss is still finite after 2,000 steps and lower than at the start.
- **Converged** means the training MSE is within 0.1% of the closed-form best value (4,734.53).

**Which learning rates are stable:**

| Learning rate | Standardized | Not standardized |
|---|---|---|
| 0.01 | stable | stable |
| 0.1 | stable | stable |
| 0.2 | stable | stable |
| 0.3 | stable | stable |
| 0.4 | stable | **diverged** |
| 0.5 | stable | diverged |
| 0.6 | **diverged** | diverged |
| 1.0 | diverged | diverged |

- Largest stable learning rate with standardization: **0.5**
- Largest stable learning rate without standardization: **0.3**

**Iterations to converge at a learning rate that is stable for both:**

| Learning rate | Standardized | Not standardized | Ratio |
|---|---|---|---|
| **0.3** | **4,355** | **29,085** | 6.7× |
| 0.1 | 13,064 | 87,254 | 6.7× |

**What I observe and why:**

| | Standardized | Not standardized |
|---|---|---|
| Largest eigenvalue of XᵀX/n | 1.903 | 2.587 |
| Smallest eigenvalue of XᵀX/n | 9.9e-5 | 2.7e-5 |
| Condition number | 1.93e4 | 9.43e4 |
| Stability limit 1/λmax | 0.526 | 0.387 |

- Gradient descent is only stable if the learning rate is below 1 / (largest eigenvalue). The limits above (0.526 and 0.387) match the table of stable learning rates.
- Without standardization, the three columns are all positive numbers with a mean far from 0. So each interaction like temp × (hr = 17) becomes almost a copy of the (hr = 17) column. Their correlation is **0.933**, compared with **−0.004** after standardizing.
- Near-copy columns make the loss surface long and thin. The largest eigenvalue gets bigger, so we need a smaller learning rate. The smallest eigenvalue gets smaller, so progress in that direction is slower. That is why it needs about 6.7 times more steps.
- Both versions reach the **same** best MSE (4,734.53). Standardizing only changes how fast gradient descent gets there, not where it ends.

**Loss vs iteration for different learning rates (standardized features):**

![Gradient descent loss vs iteration](figures/gd_loss_vs_lr.png)

| Learning rate | Steps run | Final MSE | Final MSE / best MSE |
|---|---|---|---|
| 1e-5 | 20,000 | 33,392 | 7.05 |
| 1e-4 | 20,000 | 16,621 | 3.51 |
| 1e-3 | 20,000 | 7,951 | 1.68 |
| 1e-2 | 20,000 | 4,934 | 1.04 |
| 0.1 | 20,000 | 4,737 | 1.001 |
| 1.0 | 10 (stopped) | 4.4e12 | diverged |

- **Very small learning rate (1e-5):** the loss is almost flat for the first ~1,000 steps. After 20,000 steps it is still 7 times the best value. The steps are simply too small.
- **Very large learning rate (1.0):** this is above the limit of 0.526. Each step jumps past the minimum and lands further away than before, so the loss explodes. It goes from about 69,000 to more than 1e12 in 10 steps, and I stop the run.
- **In the middle:** 0.1 reaches the best value (within 0.1%). 1e-2 follows the same path about 10 times slower.

### 1.2.3 Visualization

All plots use the training set only.

**1. Distribution of the target**

![Distribution of cnt](figures/cnt_distribution.png)

Rentals per hour are strongly right-skewed (median 142, mean 190, max 976), and 17% of hours (2,333 of 13,903) have fewer than 20 rentals, so big errors will mostly come from busy hours and the model may predict below zero at night.

**2. Hour of day vs rentals, for working and non-working days**

![Mean rentals by hour](figures/cnt_by_hour_workingday.png)

Working days have two sharp peaks (about 480 at 08:00 and 525 at 17:00), while weekends have one wide midday peak (about 370), which is why hour must be one-hot and why workingday × hr is so useful.

**3. Temperature vs rentals**

![Rentals vs temperature](figures/cnt_vs_temp.png)

Rentals go up with temperature (correlation 0.40) up to about temp = 0.8 (around 33 °C) and then go down a little, so the relationship is not a straight line.

**4. Correlation heatmap of all 167 engineered features and `cnt`**

![Correlation heatmap](figures/feature_correlation_heatmap.png)

Most features are close to uncorrelated. Each interaction is correlated with its parent column (for example workingday × (hr = k) with (hr = k) at about 0.82), and no single feature has a correlation with `cnt` above 0.40 (temp), so a good model needs many features together.

---

## 1.3 Linear Regression from Scratch

### 1.3.1 Closed-form solution

Normal equation: setting the gradient of ‖Xw − y‖² to zero gives **XᵀX w = Xᵀy**. I solve this linear system directly with `np.linalg.solve`.

**Condition numbers before solving (final 168-column matrix):**

| | Value |
|---|---|
| cond(X) | **138.8** |
| cond(XᵀX) | **1.93e4** (= 138.8², as expected) |
| **Training RMSE** | **68.81** |

This is **not** very large. We lose at most 4 to 5 of the 16 digits a computer keeps. The matrix is in good shape because 1.1 already avoided repeated information: one level is dropped from each one-hot group, `workingday` is not used alone, and `atemp` is dropped. What is left comes from rare 0/1 columns, such as rain at one specific hour, that are non-zero in only a few dozen rows.

**What happens with a redundant representation.** To show where a very large condition number comes from, I rebuilt the matrix in the naive way: keep **all** one-hot levels and add `workingday` as its own column.

| Matrix | Columns | Rank | cond(X) | cond(XᵀX) |
|---|---|---|---|---|
| **Redundant** (all levels + workingday) | 174 | 168 | **2.02e16** | **2.01e17** |
| **Corrected** (final, from 1.1) | 168 | 168 | **138.8** | **1.93e4** |
| Corrected + atemp | 169 | 169 | 150 | 2.24e4 |

**Diagnosis.** The redundant matrix has 174 columns but rank only 168, so 6 columns carry no new information. I found the 6 exact reasons, and each one holds with error 0 on every row:

1. The 4 season columns add up to the intercept column.
2. The 12 month columns add up to the intercept column.
3. The 7 weekday columns add up to the intercept column.
4. The 3 weather columns add up to the intercept column.
5. The 24 hour columns add up to the intercept column.
6. `workingday` = (weekday 1 + … + weekday 5) − `holiday`.

Because of this, XᵀX cannot be inverted, and there is no single best answer. The condition number of about 1e16 to 1e17 is just the largest value the computer can show.

- `np.linalg.solve` still returns *one* of the many answers. Training RMSE is 68.81, but the weights are arbitrary (the largest is 489.4).
- An explicit matrix inverse gives nonsense: training RMSE **24,279**.

**Correction:** drop one level from each one-hot group and do not use `workingday` as its own column. This gives the final 168-column matrix, with cond(X) = 138.8 and cond(XᵀX) = 1.93e4. Adding back `atemp`, which is only *almost* a copy of `temp`, raises them only a little (150 and 2.24e4).

### 1.3.2 Gradient descent

I used the gradient descent from 1.2.2 on the same matrix, with learning rate **0.5** (the largest stable one), starting from w = 0, and compared its weights with the closed-form weights.

| Iterations | Largest weight difference | Relative difference | Train RMSE (GD) |
|---|---|---|---|
| 1,000 | 56.0 | 0.235 | 69.0289 |
| 10,000 | 18.1 | 0.0783 | 68.8139 |
| 50,000 | 0.350 | 1.5e-3 | 68.8079 |
| 100,000 | 2.5e-3 | 1.1e-5 | 68.8079 |
| 200,000 | 1.3e-7 | 5.5e-10 | 68.8079 |
| 300,000 | **5.5e-11** | **2.3e-13** | 68.8079 |

**The correctness check passes.** After 300,000 steps, the weights match the closed-form weights up to a difference of 5.5e-11, which is the limit of computer precision. For example, the largest weight (workingday × hr = 8) is 428.714305 in both.

**Why so many steps?** The loss is almost at its best after about 10,000 steps, but some weights are still off by up to 18. Those weights belong to rare columns (rain at a particular hour). They affect very few rows, so the gradient pushes them very slowly: in that direction the error only shrinks by a factor of e about every 10,000 steps. Checking the weights is therefore a stricter test than checking the loss.

---

## 1.4 Polynomial Features and the Bias-Variance Tradeoff

Dataset: `bike_temp_comfort_zone.csv`, using its own `split` column (400 train, 200 validation, 200 test).
For each degree d = 1 to 12, the features are temp, temp², …, tempᵈ. Each power is standardized with the training mean and std of that power, an intercept is added, and the model is fit with the closed-form normal equation.

| Degree | Train RMSE | Val RMSE | Largest \|coefficient\| | cond(XᵀX) |
|---|---|---|---|---|
| 1 | 117.287 | 121.770 | 73.3 | 1 |
| 2 | 79.083 | 74.988 | 404 | 60 |
| 3 | 63.348 | 60.557 | 701 | 2.3e3 |
| 4 | 60.978 | 57.773 | 2,690 | 9.2e4 |
| **5** | 60.297 | **56.720** | 4,020 | 3.2e6 |
| **6** | 60.292 | **56.714** | 1,880 | 1.0e8 |
| **7** | 60.290 | **56.691** | 7,350 | 3.6e9 |
| 8 | 60.235 | 57.172 | 216,000 | 1.2e11 |
| 9 | 60.133 | 57.527 | 1.74e6 | 4.2e12 |
| 10 | 59.810 | 58.160 | 1.63e7 | 1.3e14 |
| 11 | 59.659 | 59.117 | 6.69e7 | 4.5e15 |
| 12 | 61.116* | 58.822 | 5.34e8 | 7.8e16 |

\* See the note on degree 12 below.

![Train and validation error vs degree](figures/poly_error_vs_degree.png)

**Region where validation error is lowest: degrees 5 to 7.** Their validation RMSEs are within 0.1% of each other. Degree 4 (57.77) and degree 8 (57.17) are clearly worse.

- **Left side (degrees 1 to 4), underfitting / high bias.** Both errors are high and fall quickly together. A straight line cannot bend. A parabola can only make a symmetric hill, but the real curve rises slowly, peaks near temp = 0.65, and falls faster. The model is too simple.
- **Right side (degrees 8 to 12), overfitting / high variance.** Training error keeps going down a little, but validation error goes up (from 56.69 to about 59). The extra powers start fitting the noise in the 400 training points. The degree-12 curve wiggles, especially at the edges.

![Fitted curves at degree 1, 7 and 12](figures/poly_fits.png)

**Largest coefficient vs degree (log scale):**

![Largest coefficient vs degree](figures/poly_max_coef_vs_degree.png)

The largest coefficient grows from 73 at degree 1 to about 5e8 at degree 12. It jumps about 30 times between degrees 7 and 8, which is the same place where validation error starts to rise. High powers of temp look almost the same as each other, so the model builds its curve from huge positive and negative coefficients that cancel out. This makes the model very sensitive to small changes in the data.

**Test RMSE at the chosen degree (degree 7, lowest validation error in the region): 52.614.** The test set was used only for this one model.

**Note on degree 12:** cond(XᵀX) is 7.8e16, which is too large for the computer to handle accurately. The normal-equation fit gives training RMSE 61.12, which is *higher* than degree 11. That is impossible for a real best fit, because degree 12 can always copy degree 11. It is a rounding error. `np.linalg.lstsq` gives 59.39 at degree 12, and both methods agree for degrees 1 to 11, so the chosen region and the test result are not affected.

---

## 1.5 Regularization

**Ridge regression** minimizes ‖Xw − y‖² + λ·(sum of squared weights, not counting the intercept).
Setting the gradient to zero gives **(XᵀX + λD) w = Xᵀy**, where D is the identity matrix with a 0 in the intercept position, so the intercept is not shrunk. I solve this with `np.linalg.solve`. For λ = 0, I use the least-squares solver `np.linalg.lstsq`, as the task asks.

Preprocessing: the penalty treats all weights the same, so I standardize **every** feature column using the rows being fit, after dropping columns with zero variance. Both sweeps are scored on the same 1,737-row validation set.

### Full training set (13,903 rows, 168 columns, 0 columns dropped)

![Ridge on the full training set](figures/ridge_full.png)

| λ | Train RMSE | Val RMSE |
|---|---|---|
| 0 | 68.808 | 68.560 |
| 1e-2 | 68.808 | 68.560 |
| 1 | 68.808 | 68.558 |
| 10 | 68.810 | 68.541 |
| **100** | 68.954 | **68.530** |
| 1e3 | 72.825 | 71.848 |
| 1e4 | 96.931 | 96.239 |
| 1e5 | 149.000 | 150.547 |
| 1e6 | 176.911 | 179.315 |

**Best λ = 100.** Validation RMSE goes from 68.560 to 68.530, an improvement of only **0.031 (0.04%)**, which is basically nothing. Above λ = 1,000 the penalty removes real signal and both errors go up. At λ = 1e6 the model almost just predicts the average.

### Random 100-row subsample of the training set (seed 42)

28 columns were constant in these 100 rows, mostly rare interaction cells, so I dropped them. That leaves 139 features + 1 intercept = 140 columns for only 100 rows.

![Ridge on 100 rows](figures/ridge_subsample.png)

| λ | Train RMSE | Val RMSE |
|---|---|---|
| 0 | 0.00 | 324.26 |
| 1e-2 | 0.77 | 314.90 |
| 1 | 18.68 | 155.31 |
| **10** | 35.88 | **130.16** |
| 100 | 74.09 | 131.06 |
| 1e3 | 147.13 | 160.72 |
| 1e4 | 182.28 | 180.69 |
| 1e5 | 187.41 | 183.73 |
| 1e6 | 187.94 | 184.05 |

**Best λ = 10.** Validation RMSE goes from 324.26 to 130.16, an improvement of **194.10 (59.9%)**.

To check that this is not luck, I repeated it with 20 other random 100-row subsamples:

- The best λ was 100 in 16 cases, 10 in 3 cases, and 1e5 in 1 case.
- The improvement was always large: median 120.1, smallest 29.5, largest 915.7.

### Comparing the two improvements

| Training data | Best λ | Improvement in validation RMSE |
|---|---|---|
| Full (13,903 rows) | 100 | 0.031 (0.04%) |
| 100 rows | 10 | 194.10 (59.9%) |

**Why they are so different:** ridge adds a little **bias** (it pulls weights towards 0) in exchange for less **variance** (weights that change less when the training data changes). It only helps when variance is the bigger problem.

- **Full training set: almost no variance to remove.** There are about 83 rows for every weight, and the matrix is well-conditioned (cond(X) = 139). The normal least-squares weights are already stable, so shrinking them mainly adds bias.
- **100 rows: variance is huge.** There are more weights (140) than rows (100), so XᵀX cannot be inverted and many different weight vectors fit the training rows perfectly (train RMSE 0). The one we get fits the noise and fails on new data. Ridge makes the system solvable and shrinks the weights that the data barely supports, which removes a lot of variance.
- **The same λ is stronger when there is less data.** The error part of the loss is a sum over rows, so it grows with the number of rows, but the penalty part does not. λ = 100 is tiny compared with 13,903 rows of error, but strong compared with 100 rows.

Even the best 100-row model (130) is much worse than the full-data model (68.5). Regularization reduces overfitting, but it cannot replace having more data.

---

## 1.6 Evaluation

Model: ridge regression from 1.5, fit on the full training set with **λ = 100** (chosen by the full-training-set sweep), with the same training-set standardization. It is evaluated once on the 1,739-row test set. The metrics are implemented from scratch, and predictions are **not clipped**.

| Metric | Test set |
|---|---|
| MAE | **49.02** |
| RMSE | **67.03** |
| R² | **0.857** |
| Pearson correlation | **0.926** |

- The model explains about 86% of the variation in hourly rentals.
- Test RMSE (67.03) is close to training (68.95) and validation (68.53) RMSE, so the model is not overfitting.
- MAE is lower than RMSE because a few big errors at busy rush hours count extra in RMSE.

**Negative predictions:**

- **181 of 1,739 test predictions (10.4%) are below zero**, the lowest being −108.6, even though a rental count can never be negative.
- 161 of them are at night or early morning (00:00 to 07:59). The real counts in these rows are very small (median 9, maximum 77).
- This happens because a linear model just adds effects together. A low night-time value plus negative effects (rain, high humidity, winter, first year) can add up to less than zero.
- The model also assumes errors are the same size at every level of demand, but real counts vary little near zero and a lot at peak hours.
- For comparison only (not the reported numbers): if predictions were clipped at 0, the results would be MAE 45.42, RMSE 64.99, R² 0.866, Pearson 0.932. Clipping can only help, because the true value is never below 0.
- A better fix would be to predict log(cnt + 1) or to use a model made for counts (such as Poisson regression). Neither can give negative rentals.
