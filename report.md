# Q2. Radar Signal Classification with Support Vector Machines

All code is in [`q2.ipynb`](q2.ipynb). Plots are in the [`figures/`](figures/) folder.
To run it, open the notebook from inside the `q2/` folder and run all cells. It reads `data/ionosphere.data` and needs `numpy`, `pandas`, `matplotlib` and `cvxopt`.

**Rules I followed**

- **Positive class = "good" = +1**, negative class = "bad" = −1.
- **Random seed: 42.** One NumPy random generator makes both the train/test split and the folds.
- **Support vectors:** a point counts as a support vector if **αᵢ > 1e-5**.
- **No library SVM** (no `sklearn.svm`).
- **`cvxopt.solvers.qp`** is used only as a general QP solver. I wrote the dual problem, its inputs, and the code that pulls out the support vectors, weights and bias myself.
- **Metrics from scratch:** I wrote my own confusion matrix, accuracy, precision, recall and F1.
- **Hyperparameters:** every one (learning rate, kernel, degree, γ, C) is chosen with **5-fold stratified cross-validation** on the training set. CV accuracy is the mean over the 5 folds.
- **Test set:** not used until 2.5.

---

## 2.1 Preprocessing and Data Analysis

### 2.1.1 Inspecting the features

The data has 351 rows and 34 feature columns (`a01` to `a34`), with no missing values. I looked at the mean, std, min, quartiles, max and number of unique values of every column (full table in the notebook).

- **`a02` has no information.** It is 0 in all 351 rows (std 0, only 1 unique value), so it cannot tell any two rows apart. **I removed it.**
- **What would go wrong if I kept it:** standardizing divides by the std, which is 0, so every value of `a02` would become 0/0 = NaN. That NaN would spread into every dot product, every kernel value and the QP matrix P, so the solver would fail or every prediction would be NaN. Even without standardizing, it would only add 0 to every dot product and distance, so it is useless. It would also be counted wrongly in γ = 1/(number of features) in 2.3.2.
- **`a01` is kept.** It only takes the values 0 and 1, but it is useful: all 38 rows with `a01 = 0` are "bad".
- The other 32 columns are continuous numbers in [−1, 1] with std between 0.44 and 0.65.

**Feature columns left: 33.**

### 2.1.2 Train/test split and cross-validation folds

**Random seed: 42.**

**How I split:** inside each class separately, I shuffled the rows and put 20% (rounded) into the test set. This is a stratified 80:20 split.

| Set | Rows | Positive (good) | Negative (bad) | Positive share |
|---|---|---|---|---|
| All data | 351 | 225 | 126 | 0.641 |
| **Train** | **281** | **180** | **101** | 0.641 |
| **Test** | **70** | **45** | **25** | 0.643 |

**How I built the 5 folds:** inside each class, I shuffled the training rows again and cut them into 5 nearly equal parts. Fold *k* = part *k* of the good rows + part *k* of the bad rows. Each fold is used once for validation, while the other 4 are used for training.

| Fold | Rows | Good | Bad | Positive share |
|---|---|---|---|---|
| 1 | 57 | 36 | 21 | 0.632 |
| 2 | 56 | 36 | 20 | 0.643 |
| 3 | 56 | 36 | 20 | 0.643 |
| 4 | 56 | 36 | 20 | 0.643 |
| 5 | 56 | 36 | 20 | 0.643 |

Every fold has almost the same class balance as the whole training set.

### 2.1.3 Standardization

For each feature, z = (x − mean) / std, where the mean and std come from the **training set only**. I used the same mean and std for the test set. The CV folds are just subsets of the standardized training rows, so nothing is refit for each fold. Before this, the training std of the features ranged from 0.30 to 0.64.

**Why this matters for the kernels too, not just the linear model:**
- **The kernels in 2.3 compare points using distances and dot products.** The polynomial kernel uses xᵀx′ and the RBF kernel uses ‖x − x′‖², so a feature with a bigger spread would control every kernel value.
- **A fixed γ assumes equal scales.** A value like γ = 1/33 only means the same thing for every feature if all features are on the same scale.
- **Fitting on training rows only** keeps the test set unseen, so it cannot shape the kernel.

### 2.1.4 Visualization

**1. Class balance**

![Class balance](figures/class_balance.png)

There are 225 good (64.1%) and 126 bad (35.9%) returns, so always guessing "good" already gives 64.1% accuracy; that is the bar every model must beat, and accuracy alone can hide weakness on the "bad" class.

**2. Two features coloured by class (a03 vs a05)**

![a03 vs a05](figures/pair_a03_a05.png)

Good returns form one tight group in the top-right corner, while bad returns are spread over the whole square around it, which a single straight line cannot separate well.

**3. Correlation heatmap of the 33 kept features**

![Correlation heatmap](figures/feature_correlation.png)

The odd-numbered features (a03, a05, …) are fairly strongly correlated with each other (mean |r| = 0.45, up to 0.83 for a13–a15), while the even-numbered ones are only weakly linked to them (mean |r| = 0.15), so many features partly repeat the same information.

---

## 2.2 Linear SVM from Scratch

### 2.2.1 Primal formulation: subgradient descent

The soft-margin linear SVM minimizes

J(w, b) = ½‖w‖² + C · Σᵢ max(0, 1 − yᵢ(wᵀxᵢ + b)).

The hinge part has a corner, so I use a **subgradient**. Let V be the points with yᵢ(wᵀxᵢ + b) < 1 (inside the margin or wrong). Then:

- g_w = w − C · Σ_{i∈V} yᵢxᵢ
- g_b = −C · Σ_{i∈V} yᵢ

I **sum** the per-point subgradients over the batch (not average them), as the task asks. The batch is the whole training set. I start from w = 0, b = 0 and update w ← w − η·g_w, b ← b − η·g_b.

**Learning rates tried:**
- fixed η = 1e-4, 1e-3 and 1e-2;
- a decreasing schedule, **η_t = 1e-2 / √(t + 1)**.

For each C in {0.01, 0.1, 1, 10, 100} and each learning rate, I ran 10,000 iterations and measured 5-fold CV accuracy.

| C | 1e-4 | 1e-3 | 1e-2 | 1e-2/√(t+1) |
|---|---|---|---|---|
| 0.01 | 0.8793 | 0.8722 | 0.8722 | 0.8722 |
| **0.1** | 0.8757 | **0.8792** | 0.8721 | 0.8792 |
| 1 | 0.8686 | 0.8686 | 0.8615 | 0.8722 |
| 10 | 0.8581 | 0.8722 | 0.7653 | 0.8509 |
| 100 | 0.8474 | 0.8545 | 0.8013 | 0.8545 |

**What I observe with each learning rate.** "Swing" means how much the objective still moves over the last 1,000 steps, compared with its lowest value.

- **1e-4: stable but too slow.** At C = 0.1 the objective is still 7.223 after 10,000 steps, while the true best value is 7.1865 (from the QP in 2.2.2). Its top CV score at C = 0.01 (0.8793) comes from weights that have not finished growing, not from a better model.
- **1e-3: good at small C, unstable at large C.** At C = 0.1 it reaches the lowest objective of all (7.187) and hardly swings (0.04%). At C = 10 it swings by 98% of the objective, and at C = 100 by 32 times.
- **1e-2: too big.** Because the subgradients are summed, the step grows with C and with the number of margin violators. Even at C = 0.1 it stops above the best value (7.208) and swings 0.5%. At C = 1 it swings 8×, at C = 10 48×, and CV accuracy falls to 0.77.
- **Schedule 1e-2/√(t+1): fast start, never unstable, slow finish.** It starts as fast as 1e-2, but the shrinking steps stop the swinging at every C (at most 0.39 even at C = 100, against 201 for 1e-2). The cost is a slow finish: at C = 0.1 it ends at 7.191, a little behind 1e-3.

Once the optimizer has converged, CV accuracy hardly depends on the learning rate: 0.8793 and 0.8792 differ by much less than one validation point (1/56 ≈ 0.018). Larger C gives lower CV accuracy (about 0.85), because there is less regularization.

**My choice and why:**
- **C = 0.1:** the best CV accuracy among runs that actually converged (0.8792). The exact QP in 2.2.2 confirms it.
- **Fixed η = 1e-3:** at C = 0.1 it ties the schedule on CV accuracy (0.8792) but gets much closer to the true best solution (see 2.2.2). The schedule would be the safer choice if C were larger, where 1e-3 starts to swing.

**Hyperparameters used:**

| C | Learning rate | Iterations | Batch | Start |
|---|---|---|---|---|
| 0.1 | fixed 1e-3 | 10,000 | full training set (281 rows), subgradients summed | w = 0, b = 0 |

### 2.2.2 Dual formulation: quadratic program

**Derivation.** With slack variables ξᵢ, the primal is

minimize ½‖w‖² + C Σᵢ ξᵢ, such that yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ and ξᵢ ≥ 0.

I add multipliers αᵢ ≥ 0 (for the margin constraints) and μᵢ ≥ 0 (for ξᵢ ≥ 0). Setting the derivatives of the Lagrangian to zero gives:

- w = Σᵢ αᵢ yᵢ xᵢ
- Σᵢ αᵢ yᵢ = 0
- C − αᵢ − μᵢ = 0, which together with μᵢ ≥ 0 means **0 ≤ αᵢ ≤ C**

Putting these back gives the **dual**:

maximize Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ xᵢᵀxⱼ, such that 0 ≤ αᵢ ≤ C and Σᵢ αᵢ yᵢ = 0.

**Standard QP form.** `cvxopt.solvers.qp` solves: minimize ½xᵀPx + qᵀx such that Gx ≤ h and Ax = b. I multiply the dual objective by −1 to turn "maximize" into "minimize".

| Solver input | Value | What it is in my derivation |
|---|---|---|
| x | α (length n) | the multipliers, one per training point |
| P | Pᵢⱼ = yᵢ yⱼ xᵢᵀxⱼ (n × n) | the quadratic part of the (negated) dual; for kernels xᵢᵀxⱼ becomes K(xᵢ, xⱼ) |
| q | −1 (length n) | the linear part −Σᵢ αᵢ of the negated dual |
| G | [−I ; I] (2n × n) | stacks −αᵢ ≤ 0 and αᵢ ≤ C |
| h | [0 ; C·1] (length 2n) | the right-hand sides 0 and C |
| A | yᵀ (1 × n) | the equality constraint Σᵢ αᵢ yᵢ … |
| b | 0 | … = 0 |

**Getting the model out of the solution:**
- **Support vectors:** αᵢ > 1e-5.
- **Bias:** a point with 0 < αᵢ < C sits exactly on the margin, so b = yᵢ − Σⱼ αⱼ yⱼ K(xⱼ, xᵢ). I average this over all such margin support vectors.
- **Decision function:** f(x) = Σ over SVs of αᵢ yᵢ K(xᵢ, x) + b.
- **Linear kernel only:** w = Σᵢ αᵢ yᵢ xᵢ.

**Results at C = 0.1** (solver status: optimal):

| | Value |
|---|---|
| **Support vectors** | **96 of 281 = 34.2% of the training set** |
| … on the margin (0 < α < C) | 20 |
| … at the bound (α = C) | 76 |
| Bias b | 0.2836 |
| Dual objective | 7.1865 |
| Primal objective of the recovered (w, b) | 7.1865 (same, so the answer is correct) |
| Training accuracy | 0.9146 |
| 5-fold CV accuracy | 0.8792 |

I also refit the exact QP for every C to check my choice of C. CV accuracy was 0.8722, **0.8792**, 0.8722, 0.8509 and 0.8439 for C = 0.01, 0.1, 1, 10 and 100, so C = 0.1 is best.

**Comparing the QP weights with the subgradient descent weights.** The table shows the largest elementwise difference. The largest QP weight is 0.616 in size.

| Iterations | Fixed η = 1e-3 | Schedule 1e-2/√(t+1) |
|---|---|---|
| 100 | 0.356 | 0.288 |
| 1,000 | 0.096 | 0.158 |
| 10,000 | **0.0011** | 0.033 |

- **Fixed 1e-3:** the difference falls fast, to 0.2% of the largest weight after 10,000 steps (the bias difference falls from 0.076 to 0.0012). A fixed step can only get within some distance of the answer, and that distance shrinks as η shrinks. The objective plot in 2.2.3 shows this run reaching that floor after about 2,000 steps. At η = 1e-3 the floor is small, so the weights match the QP.
- **Schedule:** it is ahead at 100 steps because its first steps are 10× bigger, but it falls behind later. Its steps shrink like 1/√t, so its error also shrinks slowly (0.288 → 0.158 → 0.033). It never swings and is still improving, but it needs many more steps.

Both runs are clearly moving towards the QP weights, so the primal and dual code solve the same problem.

### 2.2.3 Convergence and comparison

![Objective vs iteration](figures/primal_objective.png)

The left plot shows the objective. The right plot shows how far each run still is from the QP's best value. All four use C = 0.1 and 10,000 steps.

- **All start at J = 28.1.** With w = 0, each of the 281 points has hinge loss 1, and 0.1 × 281 = 28.1.
- **Bigger steps start faster.** 1e-2 and the schedule are below 8 within 100 steps, 1e-3 gets there by about 1,000, and 1e-4 is about 10× slower again.
- **1e-2 stops improving after about 200 steps** and then jumps around at a gap of about 0.01. Subgradient descent does not always go downhill, so a fixed step keeps circling the corner of the hinge loss instead of settling.
- **1e-3 does the same, but about 10× lower.** It bounces around a gap of about 1e-3 from about 2,000 steps on. The floor of a fixed step shrinks with η.
- **The schedule is the only run with no floor.** It keeps going down smoothly (gap 0.0041 at 10,000 steps) and would pass 1e-3 with more steps.
- **1e-4 is smooth but slow:** still 0.036 away after 10,000 steps.

**Training time** on 281 rows × 33 features, best of several runs:

| Method | Time |
|---|---|
| Subgradient descent, 10,000 steps | about **0.20 s** |
| cvxopt QP | about **0.06 s** |

The QP is faster here, but **I expect subgradient descent to scale much better as the number of training examples n grows:**
- **Subgradient descent:** each step costs O(n·d) time and only needs the data itself, so the cost grows linearly with n. With mini-batches, each step can even be independent of n.
- **QP:** needs a dense n × n matrix P, which costs O(n²·d) time and O(n²) memory to build. Each interior-point step solves an n × n system, which costs O(n³). For n = 100,000, P alone would need 80 GB of memory.

---

## 2.3 Kernels

The kernel replaces the dot product xᵢᵀxⱼ in the matrix P and in the decision function. My `fit_dual_svm` function takes the kernel as an input, so nothing else changes.

**How I chose settings:** the degree (or γ) and C are chosen **together** by 5-fold CV over a grid with C ∈ {0.01, 0.1, 1, 10, 100}. Ties go to the smaller degree or γ, then the smaller C. The curves and boundary plots below keep C fixed at the chosen value, so only the degree or γ changes.

### 2.3.1 A case where linear fails

**Baseline:** always predicting the majority class of the training folds ("good") gives **5-fold CV accuracy 0.6406**.

**Search:** I did not guess. I fit the linear QP on **all 528 pairs** of the 33 features (C = 1, 5-fold CV). **234 pairs** do not beat the baseline. I then re-checked the 40 weakest pairs with C tuned over the full grid, and compared them with an RBF kernel, to find a pair where the problem is the *shape* of the classes rather than a useless pair.

**The pair I report: a21 & a28.**

| C | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| Linear SVM CV accuracy | **0.6406** | 0.637 | 0.619 | 0.619 | 0.616 |

**Best linear SVM CV accuracy: 0.6406** (C = 0.01), which is exactly the baseline. At that C the solver returns w = 0 and b = 1, so it just predicts "good" for every point. Every other C is worse than the baseline.

![Linear SVM on a21 and a28](figures/pair_linear_boundary.png)

In both plots the whole area is on the "good" side, so there is no boundary line inside the data.

**Why a straight line cannot do better:**
- The **bad** points are in two places: a small group in the **middle** (around a21 ≈ −0.5, a28 ≈ 0 to 0.5), and a scattered **outer ring** at the edges (a28 near ±1.5–2, a21 at its extremes).
- The **good** points fill the space **between** them.

A straight line splits the plane into two halves, so it can cut off part of the ring or the middle group, but never both. Any such cut also removes many good points. Every line loses about as many good points as it gains bad ones, so the best linear rule is "always say good".

**Kernels do much better on the same pair:**
- RBF (γ = 1, C = 100): **CV accuracy 0.8788**
- Polynomial (degree 5, C = 10): **CV accuracy 0.8788**

That is about **24 percentage points** better, because both kernels can draw closed, curved boundaries.

### 2.3.2 Polynomial kernel

K(x, x′) = (γ·xᵀx′ + coef0)^d

**Values used:**

| Setting | Value |
|---|---|
| γ on all 33 kept features | **1/33 ≈ 0.0303** |
| γ on the 2-feature pair (2 features used) | **1/2** |
| coef0 | **1** (everywhere) |
| Degree | swept from 1 to 5 |

**CV accuracy for every (degree, C) on all 33 features:**

| Degree | C = 0.01 | C = 0.1 | C = 1 | C = 10 | C = 100 |
|---|---|---|---|---|---|
| 1 | 0.6406 | 0.8330 | 0.8685 | 0.8722 | 0.8615 |
| **2** | 0.6406 | 0.8722 | 0.9005 | **0.9219** | 0.9079 |
| 3 | 0.7047 | 0.8863 | 0.9078 | 0.9113 | 0.9078 |
| 4 | 0.7439 | 0.8863 | 0.9041 | 0.9006 | 0.9078 |
| 5 | 0.7724 | 0.8863 | 0.8970 | 0.8898 | 0.8898 |

**Degree with the lowest CV error: 2** (C = 10, CV accuracy **0.9219**).

![Polynomial accuracy vs degree](figures/poly_accuracy_vs_degree.png)

At C = 10:

| Degree | 1 | **2** | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Training accuracy | 0.940 | 0.989 | 1.000 | 1.000 | 1.000 |
| CV accuracy | 0.872 | **0.922** | 0.911 | 0.901 | 0.890 |
| Support vectors | 85 | 69 | 61 | 62 | 67 |

- **Below degree 2 (degree 1): too simple.** It is basically linear (training accuracy 0.94, CV accuracy 0.87).
- **Above degree 2 (degrees 3–5): overfitting.** Training accuracy is 100%, but CV accuracy keeps falling (0.911, 0.901, 0.890).

**The boundary on the a21 & a28 pair** (C = 10 in every plot):

![Polynomial boundaries on the pair](figures/poly_pair_boundaries.png)

- **Degree 1 (below the best degree):** no boundary in the data area; everything is called "good" (CV accuracy 0.619, all 281 points are support vectors).
- **Degree 2 (the best degree on all features):** a smooth **horizontal band**. It calls the extreme a28 values "bad", so it catches the outer ring but not the middle bad group (CV accuracy 0.776).
- **Degrees 3–5 (above it):** the boundary gets more and more bendy.
  - Degree 3 bends the band.
  - Degree 4 cuts a **pocket** around the middle bad group.
  - Degree 5 adds a separate "bad" island in the top-right corner and wraps tightly around small groups of points.
  - CV accuracy: 0.779, 0.847, 0.879.

On this 2-feature pair, the extra bending still helps, because two features alone need a complicated boundary. On all 33 features a degree-2 boundary is enough, and higher degrees start to overfit.

### 2.3.3 RBF kernel

K(x, x′) = exp(−γ‖x − x′‖²), with γ ∈ {0.001, 0.01, 0.1, 1, 10, 100}.

**CV accuracy for every (γ, C) on all 33 features:**

| γ | C = 0.01 | C = 0.1 | C = 1 | C = 10 | C = 100 |
|---|---|---|---|---|---|
| 0.001 | 0.6406 | 0.6406 | 0.7615 | 0.8970 | 0.9290 |
| 0.01 | 0.6406 | 0.7225 | 0.9325 | 0.9575 | 0.9610 |
| **0.1** | 0.6406 | 0.6727 | 0.9538 | 0.9645 | **0.9680** |
| 1 | 0.6406 | 0.6406 | 0.6727 | 0.6727 | 0.6727 |
| 10 | 0.6406 | 0.6406 | 0.6406 | 0.6477 | 0.6477 |
| 100 | 0.6406 | 0.6406 | 0.6406 | 0.6406 | 0.6406 |

**Best γ: 0.1** (C = 100, CV accuracy **0.9680**).

![RBF accuracy and support vectors vs gamma](figures/rbf_accuracy_vs_gamma.png)

| γ | 0.001 | 0.01 | **0.1** | 1 | 10 | 100 |
|---|---|---|---|---|---|---|
| Training accuracy (C = 100) | 0.972 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CV accuracy (C = 100) | 0.929 | 0.961 | **0.968** | 0.673 | 0.648 | 0.641 |
| Support vectors (C = 100) | 79 | 55 | 149 | 250 | 281 | 281 |
| Support vectors (C = 1) | 198 | 111 | 152 | 247 | 281 | 281 |

**The boundary on the a21 & a28 pair** (C = 100 in every plot):

![RBF boundaries on the pair](figures/rbf_pair_boundaries.png)

**Very small γ (0.001): smooth, almost straight boundary.** The kernel hardly changes over the whole data range, so every point looks similar to every other one and the model behaves almost like a linear model.
- On the pair, γ = 0.001 draws no boundary at all (everything "good"), and γ = 0.01 draws two almost straight horizontal bands.
- On all features the model is a little too simple (training accuracy 0.972, CV accuracy 0.929).

**Very large γ (10, 100): the boundary breaks into small islands.** The kernel is so local that each point only "sees" itself.
- On the pair, γ = 100 draws a tiny circle around almost every single bad training point, and everything else is "good".
- On all features the model memorizes the training data (training accuracy 1.0), but CV accuracy drops to the baseline (0.648, 0.641): a new point is far from every training point and just gets the default answer.

**Number of support vectors at the extremes:**
- **Large γ:** it goes up to **all 281 training points** at γ = 10 and 100. The kernel matrix is almost the identity, so no point can be explained by its neighbours and every point must hold up its own label.
- **Small γ:** it depends on C. It is 79 at C = 100 but **198 at C = 1**, because the almost-straight boundary cannot separate the classes well and many points end up inside the margin.
- **In the middle:** the count is lowest at medium γ (55 at γ = 0.01, C = 100).

### 2.3.4 Comparative study

| Kernel | Setting | C | CV accuracy | Support vectors | Training time (one fit, full training set) |
|---|---|---|---|---|---|
| Linear | – | 0.1 | 0.8792 | 96 | ≈ 0.05 s |
| Polynomial | degree 2, γ = 1/33, coef0 = 1 | 10 | 0.9219 | 69 | ≈ 0.07 s |
| **RBF** | **γ = 0.1** | **100** | **0.9680** | 149 | ≈ 0.07 s |

The training times are all about the same. With only 281 rows, the time is mostly spent on the same 281 × 281 QP, whichever kernel fills the matrix.

**I would choose the RBF kernel.** It has the highest CV accuracy by a clear margin: 4.6 points above polynomial and 8.9 points above linear.

**Why it wins on Ionosphere:**
- **Good returns follow one pattern.** A "good" return is a real echo from structure in the ionosphere, so its 17 pulse measurements follow a steady pattern. Section 2.1.4 shows this: good returns form one tight group in the a03/a05 plot, and the odd-numbered features are strongly linked.
- **Bad returns are scattered.** A "bad" return passes straight through, so its values look like noise and are spread in every direction around that group.
- **So the real question is "Is this return close to the typical good pattern?"** That means drawing a closed area around one group, not cutting the space in half.
- **RBF does exactly that.** Its decision is a weighted sum of similarities exp(−γ‖x − xᵢ‖²) to the support vectors, so it naturally draws closed, local boundaries around a group.
- **The others can't.** A linear kernel can only cut the space in half. A degree-2 polynomial can make rough ellipse-like shapes, which is why it comes second.

---

## 2.4 The Regularization Parameter C

I used the chosen kernel (**RBF, γ = 0.1**) and swept C over {0.01, 0.1, 1, 10, 100}.

![Error and support vectors vs C](figures/c_sweep.png)

| C | 0.01 | 0.1 | 1 | 10 | **100** |
|---|---|---|---|---|---|
| Training error | 0.359 | 0.064 | 0.014 | 0.000 | 0.000 |
| CV error | 0.359 | 0.327 | 0.046 | 0.036 | **0.032** |
| Support vectors | 207 | 208 | 152 | 151 | 149 |
| … at the bound (α = C) | 194 | 194 | 30 | 0 | 0 |

**Best C: 100** (CV error **0.0320**, no tie). Compared with the smallest C (0.01, CV error 0.3594), this is an improvement of **0.327, which is 32.7 percentage points** of CV error.
- **Where the gain comes from:** most of it happens between C = 0.1 and C = 1.
- **After C = 1:** CV error only goes down about 1.4 more points, which is roughly 4 validation points.
- **Edge of the grid:** C = 100 is the largest value in the sweep, but the curve is already flat there.

**What C controls.** C is the price paid for each margin violation (hinge loss), compared with keeping a wide margin. It is also the largest value any α can take.

- **Small C: the model underfits.** Violations are cheap and every α is capped at a tiny number, so the decision function is almost constant.
  - At C = 0.01 the model says "good" for everything: training and CV error are both 0.359, the majority-class error.
  - At C = 0.1 it starts to fit the training set (training error 0.064) but still acts almost like the baseline on new data (CV error 0.327).
- **Large C: the model fits the training data more closely.** Violations are expensive. Training error reaches 0 from C = 10, and CV error falls to about 0.03. With γ = 0.1 the boundary is smooth enough that fitting the training set exactly does not yet hurt CV error, so there is no overfitting within this sweep.

**What happens to the number of support vectors as C grows:** it goes **down**, from about 207 to 149.
- **Small C:** the margin is wide, almost every point is inside it, and those points are stuck at α = C (194 of 207).
- **Growing C:** the margin gets narrower, fewer points break it, and the points at the bound disappear completely (0 from C = 10).
- **What stays:** about 150 points on the margin, which define the boundary. The count levels off instead of going to zero, because a curved boundary in 33 dimensions needs many points to hold it in place.

---

## 2.5 Evaluation

**C for each SVM,** tuned separately by 5-fold CV with the other settings fixed:

| Model | C = 0.01 | C = 0.1 | C = 1 | C = 10 | C = 100 | **C used** |
|---|---|---|---|---|---|---|
| Linear SVM | 0.8722 | **0.8792** | 0.8722 | 0.8509 | 0.8439 | **0.1** |
| Polynomial SVM (degree 2) | 0.6406 | 0.8722 | 0.9005 | **0.9219** | 0.9079 | **10** |
| RBF SVM (γ = 0.1) | 0.6406 | 0.6727 | 0.9538 | 0.9645 | **0.9680** | **100** |

Each model was refit on the full training set and tested **once** on the 70 test rows (45 good, 25 bad). The trivial classifier always predicts the training majority class, "good".

**Test results** (precision, recall and F1 are for the positive class, "good"):

| Model | Accuracy | Precision | Recall | F1 | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|
| Linear SVM (C = 0.1) | 0.8857 | 0.8776 | 0.9556 | 0.9149 | 43 | 2 | 6 | 19 |
| Polynomial SVM (degree 2, C = 10) | 0.9000 | 0.9130 | 0.9333 | 0.9231 | 42 | 3 | 4 | 21 |
| **RBF SVM (γ = 0.1, C = 100)** | **0.9286** | **0.9762** | 0.9111 | **0.9425** | 41 | 4 | 1 | 24 |
| Trivial (always "good") | 0.6429 | 0.6429 | 1.0000 | 0.7826 | 45 | 0 | 25 | 0 |

- **RBF is the best** on accuracy, precision and F1. It calls only **1** of the 25 bad returns "good", while the linear SVM lets **6** through (the pattern studied in 2.6). The price is slightly lower recall: RBF misses 4 good returns, linear misses 2.
- **The order on the test set is the same as in CV:** RBF > polynomial > linear > trivial.
- **RBF's test accuracy (0.929) is a bit lower than its CV accuracy (0.968).** This is normal noise with only 70 test rows: one row is worth 1.4 percentage points, and the usual uncertainty of an accuracy near 0.93 on 70 rows is about 3 points.

**What the trivial baseline's accuracy hides:**
- **Its numbers look fine:** 64.3% accuracy, recall 1.0 and F1 0.78.
- **But they are free:** "good" is both the majority class (64% of the data, see 2.1.2) and the positive class, so always saying "good" gets these numbers without looking at the data.
- **The confusion matrix shows the truth:** **TN = 0 and FP = 25**. It never finds a single bad return and uses no information from the radar signal at all.
- **So judge accuracy against 64%, not 50%.** A model is only useful if it finds the smaller "bad" class, which needs the confusion matrix or per-class precision and recall, not accuracy alone.

---

## 2.6 Support Vectors and Error Analysis

The **best overall model** is the RBF SVM (γ = 0.1, C = 100): it has the highest CV accuracy (0.968), and the test results agree. The **linear SVM from 2.2.2** is the QP linear SVM with C = 0.1. The hinge loss of a training point is max(0, 1 − yᵢ f(xᵢ)). A point counts as having non-zero hinge loss if this is above **1e-3**.

**Support vectors of the best model:** **149 of 281 = 53.0% of the training set.** All 149 are on the margin (0 < α < C), and none are at the bound.

**What that suggests about the fit:**
- **It is a flexible, local fit.** More than half the training points are needed to describe the boundary. With γ = 0.1 each support vector only affects the area near it, so the boundary is built piece by piece from many points.
- **No training point is inside the margin.** The training set is separated perfectly, which on its own could mean overfitting. But the good CV accuracy (0.968) and test accuracy (0.929) show that the fit still works on new data.
- **For comparison,** the linear SVM uses fewer support vectors (96, 34%), but 76 of them break the margin. It is simpler, but it cannot separate the classes.

**Training points with non-zero hinge loss (hinge > 1e-3):**

| Model | Hinge > 1e-3 | … correct but inside the margin | … misclassified | Largest hinge | Total hinge |
|---|---|---|---|---|---|
| RBF (best model), C = 100 | **0** | 0 | 0 | 0.0001 | 0.006 |
| Linear SVM (2.2.2), C = 0.1 | **76** | 52 | 24 | 3.37 | 60.2 |

**Why they differ:**
- **RBF bends around every point.** With a large C it pays almost no hinge loss.
- **The linear SVM has to accept errors.** It can only use a flat boundary and is strongly regularized (C = 0.1), so it accepts 76 margin violations to keep a wide margin.
- **Zero training loss is not automatically good,** since memorizing also gives zero. Here the CV and test results show RBF is really better.

### The linear SVM's worst points

I sorted the training points by their hinge loss under the linear SVM and inspected the largest ones. The first 5:

| Row | True class | Linear score f(x) | Hinge loss | RBF y·f(x) | a01 | a03 | a05 | Pulse magnitude spread |
|---|---|---|---|---|---|---|---|---|
| 126 | bad | +2.37 | 3.37 | 1.0 | 1 | 0.60 | 0.83 | 0.23 |
| 87 | bad | +1.69 | 2.69 | 1.0 | 1 | 0.73 | 0.89 | 0.31 |
| 85 | bad | +1.55 | 2.55 | 1.0 | 1 | 0.39 | 0.98 | 0.30 |
| 148 | bad | +1.18 | 2.18 | 1.0 | 1 | 0.85 | 1.00 | 0.24 |
| 116 | bad | +1.14 | 2.14 | 1.0 | 1 | 1.00 | 1.00 | 0.23 |

**Group comparison** (medians):

| Group | Rows | a03 | a05 | Distance to good-class centre | Mean pulse magnitude | Pulse magnitude spread | Values exactly 0 or ±1 |
|---|---|---|---|---|---|---|---|
| Top 15 by hinge loss (linear) | 15 | 0.853 | 0.997 | 4.70 | 0.842 | **0.274** | 7 |
| Other bad returns | 86 | 0.137 | 0.000 | 7.55 | 0.744 | 0.419 | 23 |
| Good returns | 180 | 0.905 | 0.868 | 3.49 | 0.843 | **0.092** | 2 |

(Pulse magnitude = √(a_odd² + a_even²) for each of the 16 pulses a03/a04, …, a33/a34.)

**Is there a pattern? Yes.**
- **They are all bad returns called "good".** All 15 of the largest hinge losses are bad returns (17 of the top 20), and the linear SVM gives every one of them a **positive score**. It is confidently calling them "good", with scores up to +2.4.
- **On the features the linear model uses, they look like good returns.**
  - They have `a01 = 1` and strong first pulses (a03 ≈ 0.85, a05 ≈ 1.00), just like good returns and unlike other bad ones.
  - They sit close to the good-class centre (4.7, against 3.5 for good and 7.5 for other bad returns).
  - Their average pulse strength (0.84) is the same as that of good returns.
- **Where they differ is steadiness across the 16 pulses.** How much their pulse strength changes from pulse to pulse is **0.27, three times that of good returns (0.09)**. They are strong but uneven signals. Several also have many values stuck exactly at 0 or ±1.
- **The RBF model gets all of them right** (y·f(x) = 1.0, exactly on its margin). The information needed to separate them exists, but one flat boundary over the raw features cannot use it.

**Proposed change (not implemented): give the linear SVM pulse-strength features.**
1. For each of the 16 pulses, add the magnitude |zₖ| = √(a₂ₖ₊₁² + a₂ₖ₊₂²) of the complex value stored in the pair (a03, a04), …, (a33, a34).
2. Also add one summary feature: **how much these magnitudes vary across the pulses** (their standard deviation).
3. Standardize the new features with training statistics, like the others.

A linear model cannot compute a square root of squares or a spread across columns by itself, but it can give them a weight once they are handed to it as features.

**What I would expect to happen:**
- **The look-alike bad returns get pushed towards "bad".** The spread feature separates them from good returns (0.27 vs 0.09), so many of the 15 largest hinge losses should drop, and the linear model should let fewer bad returns through on the test set (now 6).
- **Linear CV accuracy should rise.** It should go from 0.879 towards the kernel models (0.92–0.97), while the model stays linear, fast and easy to read: its weights would directly say "uneven pulse strength means bad".
- **It would probably not fully reach RBF,** because some of these points look like good returns on almost every feature. It should still close a big part of the gap.
