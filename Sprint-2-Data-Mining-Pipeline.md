# 📖 Lecture: Sprint_2_complete

---

# 1. Lecture Overview

This lecture walks through the **first four stages of the Data Mining Pipeline** in detail, focusing on how raw data is prepared and how models are evaluated *before* real modelling begins.

## Main Topics

1. **The Data Mining Pipeline** — the repeating loop: Problem understanding → Explore data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring.
2. **Step 1 – Determine Sample Size** (how much data do we need).
3. **Step 2 – Divide data into Train / Test / Validation** sets, hyperparameter tuning, inference.
4. **Step 3 – Preprocessing:**
   - Feature Engineering
   - Encoding (One-Hot Encoding)
   - Handling Missing Data
   - Handling Imbalanced Data
   - Feature Scaling / Normalization / Standardization
   - Dimensionality Reduction (Curse of Dimensionality, **PCA**, **t-SNE**, UMAP)
5. **Step 4 – Determine Validation Metrics:**
   - Confusion Matrix (TP, FP, TN, FN)
   - Precision, Recall (Sensitivity), Accuracy, Specificity
   - F1-Score, ROC / AUC
   - Bayes' Theorem for evaluation

## Learning Objectives

- Understand **each stage** of the data mining pipeline and its order.
- Know **why preprocessing matters** and how each preprocessing step works.
- Understand **dimensionality reduction** and the difference between **PCA (linear, deterministic, global)** and **t-SNE (probabilistic, stochastic, local)**.
- Be able to **read a confusion matrix** and compute Precision, Recall, Accuracy, Specificity, F1, and interpret ROC/AUC.
- Develop the **mindset of an ML practitioner**: dimensionality, transformations, hyperparameter tuning, optimization (gradient descent), KL divergence.

## Important Takeaways

- **"It is not always about accuracy."** The right validation metric is *problem-specific* and needs *domain knowledge*.
- **Split data BEFORE preprocessing / scaling** to avoid data leakage.
- **PCA = deterministic, linear, keeps global structure, fast, fights noise.** **t-SNE = stochastic, probabilistic, keeps local structure, slow, poor with noise.**
- **Perplexity** is the key t-SNE hyperparameter; a middle value (~30) is usually best.
- In medical/pathology examples, **False Negatives are dangerous** (missing a cancer) → **Recall matters most**.

---

# 2. Topic-wise Notes

## 2.1 The Data Mining Pipeline

### Definition
The data mining pipeline is the step-by-step, repeating process of turning a business problem into a working, monitored model. It loops back to the data at every stage.

### Simple Example
A hospital wants to predict whether tissue is cancerous. They first understand the problem, explore patient data, build a model, share results with doctors, deploy it, and keep monitoring it in production.

### Why is it Important?
It gives structure to any data project so no critical step (like preprocessing or monitoring) is skipped. The exam often asks you to **put the stages in the correct order**.

### Key Points
- Stages: **Problem understanding → Explore data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring → (loop / End)**.
- Every stage reads from and writes back to the central **data store** (shown as a database cylinder in the diagram).
- This lecture zooms into the **preprocessing sub-pipeline** (Step 3) and **validation metrics** (Step 4).
- Sub-pipeline order: **1. Determine sample size → 2. Divide data to train/test/validate → 3. Preprocessing → 4. Determine validation metrics**.

### Common Mistakes
- Thinking the pipeline is a straight line — it is a **loop** that keeps returning to the data.
- Forgetting **monitoring** after deployment (models drift over time).

### Diagram Explanation
The "staircase" diagram shows a person climbing steps labelled: *start → Problem understanding → Explore data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring → End*, with arrows flowing to/from a central database. It represents the iterative nature of data mining.
- **Exam question:** "Order the stages of the data mining pipeline." / "Which stage comes after Modelling & Analysis?" (→ Communicate results).

### Important Keywords
- **Pipeline** – ordered sequence of data-mining steps.
- **Operationalize** – putting a model into real production use.
- **Monitoring** – checking model performance over time.

### Memory Tips
- **"P-E-M-C-D-T"**: *Problem, Explore, Model, Communicate, Deploy, Test.* → "**P**lease **E**xplore **M**y **C**ool **D**ata **T**hing."

---

## 2.2 Step 1 — Determine Sample Size

### Definition
Deciding how many data samples you need to train a reliable model before collecting or splitting data.

### Simple Example
Before training a spam filter you decide you need at least, say, 10,000 emails so the model sees enough examples of both spam and non-spam.

### Why is it Important?
Too little data → the model can't learn. Too much → wasted time/cost. It is the **first** step of the sub-pipeline.

### Key Points
- Depends on problem complexity and number of features.
- More features usually require **more samples** (linked to the **curse of dimensionality**).

### Common Mistakes
- Assuming "more data is always the answer" without considering quality and balance.

### Memory Tips
- Sample size is **Step 1** — you can't split or preprocess data you haven't collected yet.

---

## 2.3 Step 2 — Divide Data into Train / Test / Validation

### Definition
Splitting the dataset into separate parts: one to **train** the model, one to **tune** it (validation), and one to give an **unbiased final test**.

### Simple Example
From 1,000 labelled images: use 700 to train, 150 to validate/tune settings, and keep 150 hidden as the final test the model has never seen.

### Why is it Important?
Testing on data the model already saw gives falsely high scores (**overfitting**). Separate sets give an honest estimate of real-world performance.

### Key Points
- **Training set** – the model learns patterns / parameters from it.
- **Validation set** – used to **tune hyperparameters** and pick the best model.
- **Test set** – used only **once** at the end for an unbiased performance estimate.
- Split the data **before** scaling/preprocessing to prevent **data leakage**.
- After tuning on validation, the model is used for **inference** (predicting on new, unseen data).

### Common Mistakes
- **Confusing validation and test sets.** Validation = tuning; Test = final unbiased check.
- Preprocessing (e.g. computing scaling stats) on the whole dataset **before** splitting → **data leakage**.

### Comparison

| Set | Purpose | How often used | Sees labels for learning? |
|-----|---------|----------------|---------------------------|
| **Training** | Learn model parameters | Repeatedly | Yes |
| **Validation** | Tune hyperparameters, model selection | Repeatedly during tuning | Used to check, not to learn params |
| **Test** | Final unbiased evaluation | **Once**, at the end | No |

### Important Keywords
- **Overfitting** – model memorizes training data, fails on new data.
- **Data leakage** – test information sneaks into training, inflating scores.
- **Hyperparameter tuning** – choosing settings (not learned from data) like learning rate, tree depth, perplexity.
- **Inference** – using the trained model to predict on new data.

### Memory Tips
- **"Train to learn, Validate to tune, Test to judge."**

---

## 2.4 Step 3 — Preprocessing (Overview)

### Definition
Cleaning and transforming raw data into a form a model can learn from effectively.

### Simple Example
Before feeding customer data to a model: fill missing ages, turn "Male/Female" into numbers, scale income so it doesn't dominate age, and drop useless columns.

### Why is it Important?
"Garbage in, garbage out." Most real-world model quality comes from good preprocessing, not fancy algorithms.

### Key Points — the preprocessing sub-steps (in the green box)
1. **Engineer features**
2. **Handle missing data**
3. **Handle imbalanced data**
4. **Feature Scaling / Normalization**
5. **Dimensionality reduction**

### Memory Tips
- **"E-M-I-S-D"**: *Engineer, Missing, Imbalanced, Scale, Dimensionality.*

---

## 2.5 Feature Engineering

### Definition
Creating, transforming, or selecting input variables (features) so the model can learn better.

### Simple Example
From a "date of birth" column, create a new "age" feature; from "height" and "weight", create "BMI".

### Why is it Important?
Good features often improve results more than a better algorithm. It injects **domain knowledge** into the data.

### Key Points
- Includes creating new features, combining features, and selecting the useful ones.
- Requires **domain knowledge**.

### Common Mistakes
- Creating features using information not available at prediction time (leakage).

### Important Keywords
- **Feature** – an input column/variable used by the model.

### Memory Tips
- "Feature engineering = **teaching the model what to look at**."

---

## 2.6 Encoding — One-Hot Encoding

### Definition
Turning categorical (text) values into numbers a model can use. **One-Hot Encoding** makes a new binary (0/1) column for each category.

### Simple Example
A "Color" column with {Red, Green, Blue} becomes three columns: `Color_Red`, `Color_Green`, `Color_Blue`. A red row → (1, 0, 0).

### Why is it Important?
Most algorithms need numbers, not text. One-hot avoids implying a false order between categories.

### Key Points
- One new **binary column per category**.
- Prevents the model from assuming an order (e.g. Blue > Red) that doesn't exist.
- Best for **nominal** (unordered) categories.
- Downside: many categories → **many new columns** (high dimensionality).

### Common Mistakes
- Using plain integer/label encoding (Red=1, Green=2, Blue=3) for **unordered** categories → model wrongly assumes order/magnitude.

### Comparison

| Aspect | One-Hot Encoding | Label (Integer) Encoding |
|--------|------------------|--------------------------|
| Output | One 0/1 column per category | Single integer per category |
| Implies order? | **No** | **Yes** (can mislead) |
| Best for | Nominal (unordered) data | Ordinal (ordered) data |
| Dimensionality | Increases (many columns) | Stays the same |

### Important Keywords
- **Categorical variable** – a non-numeric category.
- **Nominal** – categories with no order. **Ordinal** – categories with order.

### Memory Tips
- **One-Hot = "one 1, rest 0"** across the category columns.

---

## 2.7 Handling Missing Data

### Definition
Deciding what to do with empty/absent values in the dataset.

### Simple Example
If some patients have no recorded blood pressure, you might fill the blanks with the average blood pressure (imputation) or drop those rows.

### Why is it Important?
Many algorithms crash or give biased results with missing values. Correct handling keeps the data usable and unbiased.

### Key Points
- Two broad options: **remove** (drop rows/columns) or **impute** (fill in values).
- Common imputations: **mean, median, mode**, or model-based.
- Choice depends on **how much** is missing and **why** it is missing.

### Common Mistakes
- Dropping too much data (losing information) or filling blindly with the mean (distorts distribution).
- Computing imputation values from the **whole** dataset before splitting → leakage.

### Important Keywords
- **Imputation** – filling missing values with estimates.

### Memory Tips
- **"Drop it or fill it."**

---

## 2.8 Handling Imbalanced Data

### Definition
Fixing datasets where one class has far more examples than another.

### Simple Example
In fraud detection, 99% of transactions are legit and 1% are fraud. A model can get 99% accuracy by always saying "legit" — but catches **zero** fraud.

### Why is it Important?
Imbalance makes **accuracy misleading** and makes the model ignore the rare (often most important) class. This is exactly why **accuracy alone is not enough**.

### Key Points
- Techniques: **oversampling** the minority (e.g. SMOTE), **undersampling** the majority, or using **class weights**.
- Directly connected to choosing the right **validation metric** (Precision/Recall/F1 over accuracy).

### Common Mistakes
- Trusting **accuracy** on imbalanced data.
- Resampling **before** splitting → leakage into the test set.

### Memory Tips
- "Imbalanced data → **don't trust accuracy**, look at Recall/F1."

---

## 2.9 Feature Scaling / Normalization / Standardization

### Definition
Putting all numeric features on a comparable scale so no single large-valued feature dominates.

### Simple Example
"Income" (0–100,000) and "Age" (0–100): without scaling, income dominates distance calculations. Scaling puts both on a similar range.

### Why is it Important?
Distance-based and gradient-based algorithms (KNN, SVM, PCA, neural nets, gradient descent) are sensitive to feature scale. **PCA especially requires scaling.**

### Key Points
- **Normalization (Min-Max):** rescales values to a fixed range, usually **[0, 1]**.
- **Standardization (Z-score):** rescales to **mean = 0, standard deviation = 1**.
- Always **fit the scaler on training data only**, then apply to validation/test.

### Common Mistakes
- Confusing normalization with standardization.
- Fitting the scaler on the full dataset (leakage).
- Forgetting to scale before PCA.

### Comparison

| Aspect | Normalization (Min-Max) | Standardization (Z-score) |
|--------|-------------------------|---------------------------|
| Output range | Fixed, e.g. **[0, 1]** | Mean 0, **SD 1** (unbounded) |
| Formula idea | (x − min) / (max − min) | (x − mean) / std |
| Sensitive to outliers? | **Yes** (min/max shift) | Less so |
| When to use | Bounded data, image pixels | When data is ~normal / for PCA |

### Formula Explanation

**Normalization (Min-Max):**  x' = (x − min) / (max − min)
- **Variables:** x = value, min/max = smallest/largest of that feature.
- **Meaning/Intuition:** squeezes every value into 0–1.
- **When used:** features with known bounds; neural network inputs.
- **Interpretation:** 0 = smallest, 1 = largest.
- **Common mistake:** outliers stretch min/max and squash the rest.

**Standardization (Z-score):**  z = (x − μ) / σ
- **Variables:** μ = mean, σ = standard deviation.
- **Meaning/Intuition:** how many standard deviations a value is from the mean.
- **When used:** PCA, algorithms assuming centered data.
- **Interpretation:** z = 0 is average; z = +2 is well above average.
- **Common mistake:** applying to data with heavy outliers without care.

### Memory Tips
- **Normalization → "Range 0–1."** **Standardization → "Standard normal, mean 0."**

---

## 2.10 Curse of Dimensionality

### Definition
As the number of features (dimensions) grows, data becomes sparse and distances between points become almost equal, making learning harder.

### Simple Example
Finding a friend on a 1D line is easy, on a 2D field harder, in a huge multi-story building (3D+) very hard — data "spreads out" and neighbors get far away in high dimensions.

### Why is it Important?
It motivates **dimensionality reduction** (PCA, t-SNE). More features are not always better — they can hurt performance and need far more samples.

### Key Points
- High dimensions → **sparse data**, distances lose meaning, models overfit.
- Requires **exponentially more data** as dimensions grow.
- Solution: **dimensionality reduction**.

### Common Mistakes
- Believing "more features always help."

### Memory Tips
- **"More dimensions → data gets lonely (sparse) and far apart."**

---

## 2.11 Dimensionality Reduction (General)

### Definition
Reducing the number of features while keeping the important information/structure of the data.

### Simple Example
Compressing a 784-pixel image dataset (Fashion MNIST) down to 2 dimensions so it can be plotted and clusters can be seen.

### Why is it Important?
Fights the curse of dimensionality, speeds up training, removes noise, and enables **visualization** in 2D/3D.

### Key Points
- Two main methods in this lecture: **PCA** and **t-SNE** (plus **UMAP** as advanced).
- Goals: reduce complexity, preserve structure, enable visualization.

### Memory Tips
- "Dimensionality reduction = **fewer columns, same story**."

---

## 2.12 PCA — Principal Component Analysis

### Definition
A **linear, deterministic** technique that finds new axes (principal components) capturing the **maximum variance** in the data, then keeps only the top few.

### Simple Example
A stretched cloud of 2D points is best described by one diagonal line (PC1). Projecting onto that line keeps most of the spread while reducing 2D → 1D.

### Why is it Important?
It is the classic method for **dimensionality reduction and feature transformation**. It preserves **global structure** and helps against **noisy data**. Very common exam topic vs t-SNE.

### Key Points
- Finds **principal components** = directions of maximum **variance**.
- Components are **orthogonal** (uncorrelated) and **ordered** by how much variance they explain.
- **PC1** captures the most variance, **PC2** the next most, etc.
- **Linear transformation** → distances stay meaningful.
- **Deterministic** — same input always gives the same output.
- **Fast** and **helps against noisy data**.
- Preserves **GLOBAL** structure.
- **Requires feature scaling** before use.
- Choose number of components via **explained variance** (e.g. keep enough PCs to explain ~95%).

### Common Mistakes
- Forgetting to **scale** features first.
- Thinking PCA is for visualization only — it's mainly **feature transformation / reduction**.
- Assuming PCA can capture **non-linear** structure (it can't — it's linear).

### Comparison — see the dedicated PCA vs t-SNE table in 2.14.

### Formula Explanation (conceptual — as taught)
- PCA maximizes **variance** along new axes and uses the **covariance** structure of the data.
- **Explained variance ratio:** the fraction of total variance captured by each principal component.
- **Intuition:** keep the directions where the data varies most; drop directions that are mostly noise.
- **Interpretation:** high explained variance with few components = good compression.
> The lecture does not derive the eigenvector math; know it *conceptually*.

### Diagram Explanation
- **PCA on digits (0–9) scatter plot:** colored clusters **overlap and look messy**, but the overall geometry is meaningful.
  - **Labels:** each color = a digit; axes = PC1 and PC2.
  - **Point:** PCA "**Preserves global variance**; distances between far-apart points are meaningful; good for understanding overall geometry," but "**clusters often overlap**."
  - **Exam question:** "Which plot is PCA vs t-SNE?" → the **messier, overlapping** one is PCA; the one with **clean separated blobs** is t-SNE.

### Important Keywords
- **Principal Component (PC)** – new axis of maximum variance.
- **Variance** – spread of the data.
- **Orthogonal** – at right angles / uncorrelated.
- **Explained variance** – how much information a PC keeps.
- **Linear / Deterministic** – straight-line transform; same result every run.

### Memory Tips
- **PCA = "Preserves Components by Amount of variance"**, linear, global, fast.

---

## 2.13 t-SNE — t-Distributed Stochastic Neighbor Embedding

### Definition
A **non-linear, probabilistic, stochastic** technique that reduces dimensions (usually to 2D) for **visualization** by keeping similar points close together (**local structure**).

### Simple Example
Taking 784-dimensional handwritten digit images and plotting them in 2D so all the "3"s form one tight blob, all the "7"s another, etc.

### Why is it Important?
It creates **clear, visually intuitive clusters** — great for exploring high-dimensional data. Heavily contrasted with PCA in exams.

### Key Points
- Purpose: **data visualization in low dimensions (typically 2D)**.
- **Stochastic** (random init) → different runs can give different maps.
- **Probabilistic (local similarity) embedding** — models similarity as probability.
- **Slow**, and **can't cope with noisy data**.
- Preserves **LOCAL** structure only; **global distances are NOT meaningful** (cluster spacing can mislead).
- Uses a **Gaussian** distribution in the original high-dim space and a **t-Student** distribution in the low-dim space.
- Optimized with **Gradient Descent** to minimize **KL Divergence** between distributions P (original) and Q (low-dim).
- **Perplexity** is the key hyperparameter.

### How t-SNE Works (step by step)
1. Measure distances from a **point of interest** to all others in the original high-dim space.
2. Convert distances to **joint probabilities**: close points → **high** joint probability; far points → **low** joint probability. This builds a **symmetric similarity matrix (P)** using a **Gaussian**.
3. Randomly scatter points into the lower dimension; model similarities with a **t-Student** distribution → matrix **Q**.
4. **Iterate:** some points **attract**, some **repel** (attractive & repulsive forces).
5. Use **Gradient Descent** to minimize **KL Divergence** so Q matches P as closely as possible — preserving neighborhood relationships.
6. Stop when distances **converge**.

### Common Mistakes
- Reading **global** meaning into t-SNE plots (distances between clusters are **not** reliable).
- Expecting the **same plot every run** (it's stochastic).
- Using it on very noisy data.
- Picking a bad **perplexity** (too low → fragments; too high → one blob).

### Formula Explanation — KL Divergence

**KL(P ‖ Q) = Σ_(i≠j) p_ij · log( p_ij / q_ij )**

- **Variables:** P = similarity distribution in the **original** high-dim space; Q = similarity distribution in the **low-dim** map; p_ij, q_ij = joint probabilities that points i and j are neighbors.
- **Meaning:** measures the **difference between two probability distributions**.
- **Intuition:** t-SNE moves points until the low-dim similarities Q look like the original similarities P (low KL = good match).
- **When used:** the **loss function** t-SNE minimizes via gradient descent.
- **Interpretation:** **KL = 0** means P and Q are identical (perfect preservation). Larger KL = worse.
- **Common mistake:** thinking KL is symmetric — KL(P‖Q) ≠ KL(Q‖P).

### Diagram Explanations
- **Gaussian "point of interest" diagram:** shows a bell curve centered on a point; near points get **high similarity scores**, far points **low**. Illustrates converting distance → probability.
- **P / Q similarity matrices:** P (original, Gaussian) has clean block structure; Q (random start, t-student) is messy; after many iterations Q's blocks match P's — showing convergence.
- **1D scatter animation:** points put in random order, then attract/repel over **iterations** until distances converge into 3 clean groups.
- **Exam question:** "What does t-SNE minimize?" → **KL Divergence** (via **gradient descent**). "Which distribution in low dim?" → **t-Student**.

### Important Keywords
- **Stochastic** – involves randomness (different runs differ).
- **Joint probability / similarity matrix** – how similar two points are.
- **Gaussian** – bell curve used in original space.
- **t-Student distribution** – heavy-tailed distribution used in low-dim space.
- **KL Divergence** – difference between distributions (the loss).
- **Gradient Descent** – optimization used to minimize the loss.
- **Attractive / Repulsive forces** – pull similar points together, push dissimilar apart.

### Memory Tips
- **t-SNE = "tiny neighborhoods"** → **Local**, stochastic, slow, great pictures.
- **"P is the truth, Q is the sketch — KL makes the sketch match the truth."**

---

## 2.14 Perplexity (t-SNE Hyperparameter)

### Definition
Perplexity controls **how many neighbors each point considers** when t-SNE estimates similarities — balancing focus on **local vs global** structure.

### Simple Example
With 50 points per cluster: perplexity = 2 breaks clusters into fragments (too local), perplexity = 100 merges everything into one blob (too global), perplexity = 30 gives two clean clusters (just right 😎).

### Why is it Important?
Wrong perplexity ruins the t-SNE plot. It's the main knob you tune, and a common quiz.

### Key Points
- Controls the **balance between local and global** structure.
- Influences how t-SNE defines **"neighborhoods"** when computing similarities.
- **Too small** → fragmented, misleading micro-clusters.
- **Too large** → everything blurs into one blob.
- A **medium value (~30)** is usually a good default.

### Common Mistakes
- Thinking "higher perplexity = better." Both extremes are bad.

### Diagram Explanation
- **Perplexity circles (2, 4, 5):** larger perplexity draws a bigger "neighborhood circle" that starts overlapping other clusters — showing how it defines neighborhoods.
- **Quiz plot (Perplexity 2 / 30 / 100):** 2 → 🙁 fragmented; 30 → 😎 clean two clusters; 100 → 🙁 one merged blob. **Answer: ~30 is good.**

### Memory Tips
- **"Goldilocks perplexity"** — not too small, not too big, ~**30** is just right.

---

## 2.15 PCA vs t-SNE (and UMAP)

### Definition
A comparison of the two main dimensionality-reduction methods; **UMAP** is a newer alternative.

### Why is it Important?
This comparison table is one of the **highest-probability exam items** in the whole lecture.

### Comparison — PCA vs t-SNE (from the lecture table)

| Aspect | **PCA** | **t-SNE** |
|--------|---------|-----------|
| **Purpose** | Dimensionality reduction & feature transformation | Data visualization in low dimensions (typically 2D) |
| **Type of algorithm** | **Deterministic** | **Stochastic** |
| **Approach** | **Linear** transformation (distance makes sense) | **Probabilistic** (local similarity) embedding |
| **Speed** | **Fast** | **Slow** |
| **Noisy data** | **Helps** against noisy data | **Can't cope** with noisy data |
| **Structure preserved** | **Global** | **Local only** |

### "So what?" (key insight)
- Sometimes it's useful to **combine** them: **apply PCA first, then t-SNE** — best of both worlds (PCA reduces/denoises, t-SNE visualizes).

### PCA vs t-SNE — on the digits plots
- **PCA:** preserves **global variance**; far-apart distances are meaningful; good overall geometry; but clusters **overlap / look messy**.
- **t-SNE:** preserves **local similarity**; excellent at grouping similar digits (all "3"s together); **clear cluster separation**; but **cluster spacing can be misleading** (clusters may look equally far apart even when they aren't).

### UMAP (Uniform Manifold Approximation and Projection)
- Advanced tool to explore **after** PCA and t-SNE.
- Another powerful method for **visualizing and understanding** high-dimensional data (shown on 784-dim Fashion MNIST).

### Common Mistakes
- Saying t-SNE preserves global structure (it does **not**).
- Saying PCA is stochastic (it's **deterministic**).
- Reading cluster distances in a t-SNE plot as meaningful.

### Memory Tips
- **PCA = D-L-F-G-N**: *Deterministic, Linear, Fast, Global, fights Noise.*
- **t-SNE = S-P-S-L-N**: *Stochastic, Probabilistic, Slow, Local, bad with Noise.*
- **"PCA sees the forest (global); t-SNE sees the trees (local)."**

---

## 2.16 Step 4 — Determine Validation Metrics (Overview)

### Definition
Choosing the right numbers to measure how good a model is — after preprocessing.

### Simple Example
For cancer detection you care most about **not missing sick patients** (Recall), not just overall accuracy.

### Why is it Important?
The lecture stresses: metrics are **problem-specific**, need **domain knowledge**, are **methodology-specific**, and **"it is not always about accuracy."**

### Key Points
- Metric choice depends on the problem and domain.
- Built from the **confusion matrix** (TP, FP, TN, FN).

### Memory Tips
- **"Not always about accuracy"** — the exam loves this line.

---

## 2.17 Confusion Matrix

### Definition
A 2×2 table comparing the model's **predictions** against the **truth**, splitting results into TP, FP, TN, FN.

### Simple Example
Classifying tissue as **malignant (Positive)** or **benign (Negative)**: a malignant tumor correctly flagged = **TP**; a malignant tumor missed = **FN**; a benign flagged as malignant = **FP**.

### Why is it Important?
Every other metric (precision, recall, accuracy, specificity, F1) is computed from it. Reading it correctly is essential.

### Key Points / Layout (rows = Truth, columns = Predicted)

| | **Predicted Positive** | **Predicted Negative** |
|---|---|---|
| **Truth Positive** | **True Positive (TP)** | **False Negative (FN)** |
| **Truth Negative** | **False Positive (FP)** | **True Negative (TN)** |

- **TP** – malignant correctly classified malignant.
- **FN** – malignant wrongly classified benign (**dangerous — missed cancer**).
- **FP** – benign wrongly classified malignant (false alarm).
- **TN** – benign correctly classified benign.

### Common Mistakes
- **Swapping FP and FN.** Trick: the **second word** (Positive/Negative) is what the model **predicted**; **True/False** says whether it was right.
- Mixing up which axis is truth vs prediction.

### Diagram Explanation
- **Two-circle Venn diagram:** the **circle = all cancer patients** (relevant/positive). Left region = negatives predicted, right = positives predicted. Inside circle & predicted positive = **TP**; inside circle but predicted negative = **FN**; outside circle predicted positive = **FP**; outside circle predicted negative = **TN**.
- **Exam question:** "Label TP/FP/TN/FN on the confusion matrix." / "Which cell represents a missed cancer?" → **FN**.

### Important Keywords
- **TP, FP, TN, FN** – the four confusion-matrix cells.
- **Positive class** – the class of interest (e.g. malignant).

### Memory Tips
- **First word = correctness (True/False); second word = prediction (Positive/Negative).**
- **FN = "missed it"; FP = "false alarm."**

---

## 2.18 Precision

### Definition
Of all the items the model **predicted positive**, what fraction were actually positive.

### Simple Example
The model flags 10 tumors as malignant; 8 truly are. Precision = 8/10 = 0.8.

### Why is it Important?
High precision = few **false alarms**. Matters when false positives are costly (e.g. unnecessary surgery, spam filter deleting real mail).

### Formula Explanation

**Precision = TP / (TP + FP)**

- **Variables:** TP = true positives, FP = false positives.
- **Meaning:** "What fraction of our positive predictions was correct?"
- **Intuition:** *"How many selected items are relevant?"*
- **When used:** when **false positives** are expensive.
- **Interpretation:** 1.0 = every positive prediction was right.
- **Common mistake:** confusing it with recall — precision looks at **predicted** positives (the column), recall at **actual** positives (the row).

### Key Points
- Denominator uses **predicted positives** (TP + FP).
- Also called *positive predictive value*.

### Memory Tips
- **Precision → "P for Predicted positives."** How many selected are relevant.

---

## 2.19 Recall (Sensitivity / True Positive Rate)

### Definition
Of all the **actually positive** items, what fraction the model correctly found.

### Simple Example
There are 20 real malignant tumors; the model catches 16. Recall = 16/20 = 0.8.

### Why is it Important?
High recall = few **misses (FN)**. Critical in medicine — **missing a cancer (FN) is dangerous**. The most emphasized metric for the pathology example.

### Formula Explanation

**Recall = TP / (TP + FN)**

- **Variables:** TP = true positives, FN = false negatives.
- **Meaning:** "What fraction of actually positive examples did we find?"
- **Intuition:** *"How many relevant items are selected?"*
- **When used:** when **missing positives (false negatives)** is costly (cancer, fraud).
- **Interpretation:** 1.0 = caught every real positive.
- **Common mistake:** confusing with precision; recall's denominator is **actual positives** (TP + FN).

### Key Points
- Same as **Sensitivity** and **True Positive Rate (TPR)**.
- Denominator = **actual positives** (TP + FN).

### Comparison — Precision vs Recall

| Aspect | **Precision** | **Recall (Sensitivity)** |
|--------|---------------|--------------------------|
| Formula | TP / (TP + FP) | TP / (TP + FN) |
| Question | "How many selected are relevant?" | "How many relevant are selected?" |
| Focuses on | **False Positives** (false alarms) | **False Negatives** (misses) |
| Use when | False alarms costly | Missing positives costly (cancer) |

### Memory Tips
- **Recall → "R for Real/actual positives."** How many relevant we recalled.
- **"Recall catches the cancer"** (avoids FN).

---

## 2.20 Accuracy

### Definition
The fraction of **all** predictions that were correct.

### Simple Example
Out of 100 samples the model gets 90 right (positives and negatives). Accuracy = 90/100 = 0.9.

### Why is it Important?
The most intuitive metric — but **misleading on imbalanced data**. The lecture repeatedly warns "**it is not always about accuracy**."

### Formula Explanation

**Accuracy = (TP + TN) / (TP + TN + FP + FN)**

- **Variables:** all four confusion-matrix cells.
- **Meaning:** "What fraction of our predictions was correct?"
- **Intuition:** correct predictions ÷ total predictions.
- **When used:** balanced datasets.
- **Interpretation:** 1.0 = every prediction right.
- **Common mistake:** trusting accuracy on **imbalanced** data (99% "legit" example) — it hides poor minority-class performance.

### Common Mistakes
- Reporting only accuracy on imbalanced problems.

### Memory Tips
- **Accuracy = "All correct over All"** (TP+TN on top, everything on bottom).

---

## 2.21 Specificity (True Negative Rate)

### Definition
Of all the **actually negative** items, what fraction the model correctly labelled negative.

### Simple Example
Of 50 truly benign tumors, the model correctly calls 45 benign. Specificity = 45/50 = 0.9.

### Why is it Important?
The "mirror image" of recall — measures how well the model avoids **false alarms** among the negatives.

### Formula Explanation

**Specificity = TN / (TN + FP)**

- **Variables:** TN = true negatives, FP = false positives.
- **Meaning:** fraction of actual negatives correctly identified.
- **Intuition:** recall **for the negative class**.
- **When used:** when correctly identifying negatives matters.
- **Interpretation:** 1.0 = no false alarms.
- **Common mistake:** confusing with recall — specificity is about **negatives** (TN + FP), recall about positives (TP + FN).

### Comparison — Recall (Sensitivity) vs Specificity

| Aspect | **Recall / Sensitivity (TPR)** | **Specificity (TNR)** |
|--------|--------------------------------|------------------------|
| Formula | TP / (TP + FN) | TN / (TN + FP) |
| Focuses on | Positive class | Negative class |
| Answers | "Caught all the positives?" | "Cleared all the negatives?" |
| Error avoided | False Negatives | False Positives |

### Memory Tips
- **"Se-N-sitivity → positives; Spe-cificity → negatives."**
- Specificity = **T**rue **N**egative **R**ate = "**TN** on top."

---

## 2.22 F1-Score

### Definition
A single score that balances **Precision** and **Recall** using their **harmonic mean**.

### Simple Example
Precision = 0.8, Recall = 0.6 → F1 = 2·(0.8·0.6)/(0.8+0.6) ≈ 0.69. A model must do well on **both** to get a high F1.

### Why is it Important?
Useful when you need a **balance** of precision and recall, especially on **imbalanced** data where accuracy fails.

### Formula Explanation

**F1 = 2 · (Precision · Recall) / (Precision + Recall)**

- **Variables:** Precision and Recall.
- **Meaning:** harmonic mean of precision and recall.
- **Intuition:** punishes extreme imbalance — you can't get a high F1 by being great at one and terrible at the other.
- **When used:** imbalanced data; when both FP and FN matter.
- **Interpretation:** 1.0 = perfect precision and recall; near 0 if either is very low.
- **Common mistake:** using the plain average instead of the **harmonic** mean.

### Memory Tips
- **F1 = "balance of P and R"** → harmonic mean; low if either is low.

---

## 2.23 ROC Curve and AUC

### Definition
The **ROC curve** plots **True Positive Rate (Recall)** vs **False Positive Rate** across all thresholds; **AUC** is the area under it.

### Simple Example
Sliding the decision threshold from strict to lenient traces the ROC curve. AUC = 0.5 is random guessing; AUC = 1.0 is a perfect classifier.

### Why is it Important?
Shows performance across **all thresholds**, independent of one cutoff — great for comparing models.

### Formula / Axes Explanation
- **Y-axis:** **True Positive Rate (TPR) = Recall = TP/(TP+FN)**.
- **X-axis:** **False Positive Rate (FPR) = FP/(FP+TN) = 1 − Specificity**.
- **AUC (Area Under Curve):** single number summarizing the curve.
- **Interpretation:** **AUC = 1.0** perfect; **0.5** = random (diagonal line); higher = better. A curve hugging the **top-left corner** is best.
- **Common mistake:** thinking the X-axis is precision — it's **False Positive Rate (1 − specificity)**.

### Diagram Explanation
- **ROC curve:** curve bowing toward the **top-left** = good; diagonal = random. Area beneath = AUC.
- **Exam question:** "What does AUC = 0.5 mean?" → random guessing. "What are the ROC axes?" → TPR vs FPR.

### Memory Tips
- **ROC = TPR vs FPR;** **AUC closer to 1 = better; 0.5 = coin flip.**

---

## 2.24 Bayes' Theorem (Evaluation Context)

### Definition
A rule to update the probability of an event given new evidence — used to reason about test results (e.g. probability of disease given a positive test).

### Simple Example
A test is positive; Bayes' theorem combines the test's accuracy with how rare the disease is to find the **actual** chance the patient is sick — often surprisingly low for rare diseases.

### Why is it Important?
Links confusion-matrix quantities (sensitivity, specificity, prevalence) to real-world probability; appears as a **quiz** in the lecture.

### Formula Explanation

**P(A | B) = [ P(B | A) · P(A) ] / P(B)**

- **Variables:** P(A) = prior probability, P(B|A) = likelihood, P(A|B) = posterior (updated) probability.
- **Meaning:** updates belief in A after seeing evidence B.
- **Intuition:** rare conditions + imperfect tests → many false positives, so a positive test may not mean high disease probability.
- **When used:** reasoning about test reliability, base rates.
- **Common mistake:** ignoring the **base rate / prior** (base-rate fallacy).

### Memory Tips
- **"Posterior ∝ Likelihood × Prior."**

---

# 3. Topic-wise Exam Practice

## Pipeline & Sample Size

**MCQ 1.** Which is the correct order of the data mining pipeline?
- a) Explore → Problem understanding → Model → Deploy
- b) **Problem understanding → Explore data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring** ✅
- c) Model → Explore → Deploy → Problem understanding
- d) Deploy → Monitor → Explore → Model
**Answer: b.**

**MCQ 2.** What is the correct order of the preprocessing sub-pipeline?
- a) **Determine sample size → Divide train/test/validate → Preprocessing → Determine validation metrics** ✅
- b) Preprocessing → Sample size → Metrics → Split
- c) Split → Metrics → Sample size → Preprocess
**Answer: a.**

**True/False.** The data mining pipeline is a one-way straight line. → **False** (it loops back to the data).

**Fill in the blank.** The very first step of the sub-pipeline is to determine the ______. → **sample size**

**One-word.** The stage after "Modelling & Analysis"? → **Communicate (results)**

**Scenario.** You deployed a fraud model 6 months ago and accuracy is dropping. Which stage handles this? → **Test and Monitoring.**

## Train / Test / Validation

**MCQ 3.** The validation set is used to:
- a) Give the final unbiased score
- b) **Tune hyperparameters** ✅
- c) Train the model weights
- d) Deploy the model
**Answer: b.**

**True/False.** You should scale/normalize data before splitting into train and test. → **False** (split first, then fit the scaler on train only — otherwise data leakage).

**Fill in the blank.** Adjusting settings like tree depth or learning rate is called ______ tuning. → **hyperparameter**

**One-word.** Using a trained model to predict on new data is called ______. → **Inference**

**Scenario.** Your model scores 99% on training but 60% on test data. What happened? → **Overfitting.**

## Feature Engineering & Encoding

**MCQ 4.** One-Hot Encoding is used for:
- a) Numeric continuous data
- b) **Categorical data** ✅
- c) Images only
- d) Missing values
**Answer: b.**

**True/False.** One-Hot Encoding creates one new binary column per category. → **True.**

**Fill in the blank.** Creating "age group" from "age" is an example of ______ ______. → **feature engineering**

**Scenario.** A "Color" column has values Red, Green, Blue. After one-hot encoding, how many columns? → **3 columns** (Color_Red, Color_Green, Color_Blue).

**Common-mistake question.** Why not just label-encode colors as 1,2,3? → It **implies a false order/magnitude** (Blue > Red), which one-hot avoids.

## Missing & Imbalanced Data

**MCQ 5.** Replacing missing values with the column mean is called:
- a) Dropping
- b) **Imputation** ✅
- c) Scaling
- d) Encoding
**Answer: b.**

**True/False.** In an imbalanced dataset, high accuracy always means a good model. → **False** (a 99%-negative dataset gives 99% accuracy by predicting "negative" every time).

**Fill in the blank.** SMOTE is a technique to handle ______ data. → **imbalanced**

**One-word.** Removing rows/columns with missing values is called ______. → **Dropping** (deletion).

**Scenario.** 990 healthy vs 10 sick patients. Model predicts "healthy" always → 99% accuracy. Problem? → **Class imbalance**; use recall/F1 and resampling (oversample minority / undersample majority / SMOTE).

## Scaling / Normalization / Standardization

**MCQ 6.** Standardization transforms data to have:
- a) Range [0,1]
- b) **Mean 0 and standard deviation 1** ✅
- c) Only positive values
- d) Integer values
**Answer: b.**

**MCQ 7.** Min-Max normalization scales values to:
- a) **Range [0, 1]** ✅
- b) Mean 0
- c) [-∞, +∞]
**Answer: a.**

**True/False.** Distance-based algorithms (KNN, K-Means, PCA) are sensitive to feature scale. → **True.**

**Fill in the blank.** Standardization uses the formula z = (x − μ) / ______. → **σ (standard deviation)**

**Comparison prompt.** When prefer standardization over normalization? → When data has **outliers** or is roughly **Gaussian**; normalization is better when you need a **bounded [0,1]** range.

## Curse of Dimensionality & Dimensionality Reduction

**MCQ 8.** The curse of dimensionality means that as dimensions grow:
- a) Data becomes denser
- b) **Data becomes sparse and distances lose meaning** ✅
- c) Models always improve
**Answer: b.**

**True/False.** Dimensionality reduction can help fight noise and speed up training. → **True.**

**Fill in the blank.** Reducing the number of features while keeping structure is called ______ ______. → **dimensionality reduction**

**Scenario.** You have 784 pixel features per image and training is slow + overfitting. What technique helps? → **Dimensionality reduction** (e.g., PCA, then t-SNE for visualization).

## PCA

**MCQ 9.** PCA is best described as:
- a) A stochastic, probabilistic method
- b) **A deterministic, linear transformation preserving global variance** ✅
- c) A classification algorithm
**Answer: b.**

**True/False.** PCA components are ordered by the amount of variance they capture. → **True.**

**Fill in the blank.** PCA finds new axes called ______ ______ that capture maximum variance. → **principal components**

**One-word.** PCA preserves which type of structure? → **Global.**

**Scenario.** You want to keep 95% of the variance with fewer features. Which method? → **PCA** (choose the number of components covering 95% variance).

## t-SNE

**MCQ 10.** t-SNE is mainly used for:
- a) Feature transformation for models
- b) **Visualizing high-dimensional data in 2D/3D** ✅
- c) Handling missing data
**Answer: b.**

**MCQ 11.** t-SNE minimizes which quantity using gradient descent?
- a) Mean squared error
- b) **KL (Kullback–Leibler) Divergence** ✅
- c) Variance
**Answer: b.**

**True/False.** t-SNE is deterministic and gives the same result every run. → **False** (it is **stochastic**).

**True/False.** In t-SNE, the distance *between* clusters is globally meaningful. → **False** (only **local** similarity is preserved; cluster spacing can mislead).

**Fill in the blank.** t-SNE models similarities with a ______ distribution in high-D and a ______ distribution in low-D. → **Gaussian**, **t-Student (Student-t)**

**One-word.** The hyperparameter controlling how many neighbors each point considers? → **Perplexity.**

**Scenario.** With 50 points per cluster, perplexity=2 gives fragmented blobs and perplexity=100 gives one messy blob. Best choice? → **~30** (a middle value).

## PCA vs t-SNE

**MCQ 12.** Which is FAST and helps against noisy data?
- a) t-SNE
- b) **PCA** ✅
**Answer: b.**

**MCQ 13.** Which preserves LOCAL structure only?
- a) PCA
- b) **t-SNE** ✅
**Answer: b.**

**Matching.** Match method → property: PCA → {deterministic, linear, global, fast, noise-robust}; t-SNE → {stochastic, probabilistic, local, slow, noise-sensitive}.

**Scenario.** Best practice for very high-D data? → Apply **PCA first, then t-SNE** to get the best of both.

## Confusion Matrix & Metrics

**MCQ 14.** A malignant tumor predicted as benign is a:
- a) True Positive
- b) **False Negative** ✅
- c) False Positive
- d) True Negative
**Answer: b.**

**MCQ 15.** Precision = ?
- a) TP / (TP + FN)
- b) **TP / (TP + FP)** ✅
- c) (TP + TN) / total
**Answer: b.**

**MCQ 16.** Recall = ?
- a) **TP / (TP + FN)** ✅
- b) TP / (TP + FP)
- c) TN / (TN + FP)
**Answer: a.**

**MCQ 17.** Specificity = ?
- a) TP / (TP + FN)
- b) **TN / (TN + FP)** ✅
- c) TP / (TP + FP)
**Answer: b.**

**True/False.** Accuracy is a good metric for highly imbalanced data. → **False.**

**Fill in the blank.** The harmonic mean of precision and recall is the ______. → **F1-score**

**One-word.** Recall is also called ______ (or True Positive Rate). → **Sensitivity**

**Formula interpretation.** F1 = 2 · (P·R)/(P+R). If P=1.0 and R=0.0, F1 = ? → **0** (harmonic mean punishes imbalance).

**Diagram interpretation.** In the "Relevant vs Selected items" Venn diagram, the green "True Positives" region divided by the whole *selected* set = **Precision**; divided by the whole *relevant* set = **Recall.**

**Scenario (cancer).** Which metric matters most when missing a cancer (FN) is deadly? → **Recall (Sensitivity).**

**Scenario (spam).** Which metric matters most when flagging a good email as spam (FP) is costly? → **Precision.**

## ROC / AUC

**MCQ 18.** The ROC curve plots:
- a) Precision vs Recall
- b) **True Positive Rate vs False Positive Rate** ✅
- c) Accuracy vs Threshold
**Answer: b.**

**True/False.** AUC = 1.0 is a perfect classifier; AUC = 0.5 is random guessing. → **True.**

**Fill in the blank.** AUC stands for Area Under the ______. → **(ROC) Curve**


---

# 4. Flashcards

**Q: What are the stages of the data mining pipeline?**
→ Problem understanding → Explore data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring (loops back to data).

**Q: What are the 4 steps of the preprocessing sub-pipeline?**
→ 1) Determine sample size, 2) Divide train/test/validate, 3) Preprocessing, 4) Determine validation metrics.

**Q: What are the 3 data splits and their jobs?**
→ Train (learn weights), Validation (tune hyperparameters), Test (final unbiased score).

**Q: What is a hyperparameter?**
→ A setting chosen before training (e.g., tree depth, learning rate, perplexity), not learned from data.

**Q: What is inference?**
→ Using a trained model to make predictions on new, unseen data.

**Q: What is feature engineering?**
→ Creating or transforming features from raw data to help the model learn better.

**Q: What is One-Hot Encoding?**
→ Converting each category into its own binary (0/1) column.

**Q: Why not label-encode nominal categories as 1,2,3?**
→ It implies a false order/magnitude between categories.

**Q: What is imputation?**
→ Filling missing values (e.g., with mean, median, or mode).

**Q: What is imbalanced data?**
→ When one class hugely outnumbers others; accuracy becomes misleading.

**Q: Name techniques for imbalanced data.**
→ Oversampling the minority (e.g., SMOTE), undersampling the majority, class weights.

**Q: Normalization (Min-Max) formula and range?**
→ x' = (x − min) / (max − min); range [0, 1].

**Q: Standardization (Z-score) formula and result?**
→ z = (x − μ) / σ; mean 0, standard deviation 1.

**Q: What is the curse of dimensionality?**
→ As features increase, data becomes sparse and distances lose meaning, hurting models.

**Q: What is PCA?**
→ A deterministic, linear dimensionality-reduction method that keeps directions of maximum variance (global structure).

**Q: What is t-SNE?**
→ A stochastic, probabilistic technique that visualizes high-dimensional data in 2D/3D, preserving local neighborhood structure.

**Q: What distribution does t-SNE use in high vs low dimensions?**
→ Gaussian in the original (high-D) space, t-Student in the lowered space.

**Q: What does t-SNE minimize, and how?**
→ KL divergence between distributions P and Q, using gradient descent.

**Q: KL divergence formula?**
→ KL(P‖Q) = Σ (i≠j) p_ij · log(p_ij / q_ij).

**Q: What is perplexity?**
→ A t-SNE hyperparameter controlling how many neighbors each point considers (local vs global balance).

**Q: Good perplexity value?**
→ A middle value (~30). Too low (2) fragments clusters; too high (100) merges them.

**Q: PCA vs t-SNE — one line each?**
→ PCA: deterministic, linear, fast, global, fights noise. t-SNE: stochastic, probabilistic, slow, local, poor with noise.

**Q: Can PCA and t-SNE be combined?**
→ Yes — apply PCA first, then t-SNE, for the best of both.

**Q: What is UMAP?**
→ Uniform Manifold Approximation and Projection; another dimensionality-reduction/visualization tool (successor to t-SNE).

**Q: Confusion matrix — 4 cells?**
→ TP (correct positive), FN (missed positive), FP (false alarm), TN (correct negative).

**Q: Precision formula and meaning?**
→ TP / (TP + FP); of predicted positives, how many were correct.

**Q: Recall formula and meaning?**
→ TP / (TP + FN); of actual positives, how many we found. (aka Sensitivity / TPR)

**Q: Accuracy formula?**
→ (TP + TN) / (TP + TN + FP + FN).

**Q: Specificity formula and meaning?**
→ TN / (TN + FP); of actual negatives, how many we correctly identified. (True Negative Rate)

**Q: F1-Score formula and use?**
→ 2 · (Precision · Recall) / (Precision + Recall); harmonic mean, good for imbalanced data.

**Q: What does ROC plot?**
→ True Positive Rate (Recall) vs False Positive Rate (1 − Specificity) at different thresholds.

**Q: AUC = 1.0 vs 0.5?**
→ 1.0 = perfect classifier; 0.5 = random guessing.

**Q: In cancer detection, which is more dangerous, FN or FP?**
→ FN (a real cancer classified as benign) — so Recall/Sensitivity matters most.

**Q: Should you scale before or after splitting data?**
→ After splitting — fit scaler on train only to avoid data leakage.

**Q: "It is not always about ______."**
→ accuracy (metric choice is problem-specific + needs domain knowledge).

---

# 5. Rapid Revision

- **Pipeline:** Problem → Explore → Model → Communicate → Deploy → Test/Monitor (loops).
- **Sub-pipeline:** Sample size → Split → Preprocess → Validation metrics.
- **Splits:** Train (learn) / Validation (tune) / Test (final).
- **Overfitting** = great on train, poor on test.
- **Hyperparameter tuning** uses the validation set; **inference** = predict on new data.
- **Feature engineering** = create better features from raw data.
- **One-Hot Encoding** = one binary column per category (avoids fake ordering).
- **Missing data:** drop or impute (mean/median/mode).
- **Imbalanced data:** accuracy misleads → use recall/F1 + resampling (SMOTE).
- **Normalization** → [0,1]; **Standardization** → mean 0, std 1.
- **Scale AFTER splitting** (avoid leakage). Distance-based methods need scaling.
- **Curse of dimensionality:** more features → sparse data, meaningless distances.
- **PCA:** deterministic, linear, global, fast, fights noise; keeps max-variance components.
- **t-SNE:** stochastic, probabilistic, local, slow, poor with noise; for 2D visualization.
- **t-SNE internals:** Gaussian (high-D) → t-Student (low-D), minimize **KL divergence** via **gradient descent**.
- **Perplexity ≈ 30** is usually good (balances local/global).
- **PCA + t-SNE** can be combined (PCA first).
- **Confusion matrix:** TP, FN, FP, TN.
- **Precision** = TP/(TP+FP); **Recall** = TP/(TP+FN); **Accuracy** = (TP+TN)/all; **Specificity** = TN/(TN+FP).
- **F1** = harmonic mean of precision & recall (best for imbalance).
- **ROC** = TPR vs FPR; **AUC** = 1 perfect, 0.5 random.
- **Cancer example:** FN is dangerous → maximize Recall.
- **"It is not always about accuracy."**

---

# 6. High Probability Exam Questions

### ★★★★★ Very High
- **Read a confusion matrix and compute Precision, Recall, Accuracy, Specificity.** (Formulas + interpretation.)
- **Compare PCA vs t-SNE** (deterministic/stochastic, linear/probabilistic, global/local, fast/slow, noise handling).
- **Define Precision vs Recall** and say which matters in a cancer/medical scenario (→ Recall).
- **What does t-SNE minimize?** → KL divergence via gradient descent (Gaussian→t-Student).
- **Order the data mining pipeline / preprocessing sub-pipeline.**
- **Perplexity:** what it is + which value is good (~30) using the quiz figure.

### ★★★★ High
- **F1-Score:** formula, why it's used for imbalanced data.
- **Normalization vs Standardization** (formulas + ranges).
- **Why split before scaling** (data leakage).
- **One-Hot Encoding:** purpose + number of columns produced.
- **Curse of dimensionality** definition + why reduction helps.
- **ROC/AUC interpretation** (AUC = 1 perfect, 0.5 random).
- **Identify which plot is PCA vs t-SNE** (t-SNE = clean separated clusters; PCA = overlapping/messy but globally meaningful).
- **Handling imbalanced data** techniques (SMOTE, over/undersampling).

### ★★★ Medium
- **Bayes' theorem** in evaluation context (base rate / posterior).
- **UMAP** — what it is (next step after PCA/t-SNE).
- **Sensitivity vs Specificity** distinction.
- **Handling missing data** (drop vs impute).
- **Inference vs training** definitions.
- **Why "it is not always about accuracy."**

---

# 7. Lecture Cheat Sheet

### Pipelines
- **Main:** Problem understanding → Explore → Modelling & Analysis → Communicate → Deploy → Test & Monitoring (loop).
- **Sub:** ① Sample size → ② Train/Test/Validate split → ③ Preprocessing → ④ Validation metrics.
- **Preprocessing box:** Engineer features · Handle missing data · Handle imbalanced data · Feature scaling/normalization · Dimensionality reduction.

### Key Definitions
- **Hyperparameter:** setting fixed before training (tuned on validation set).
- **Inference:** predicting on new data with a trained model.
- **Feature engineering:** building better features from raw data.
- **One-Hot Encoding:** one binary column per category.
- **Imputation:** filling missing values.
- **Imbalanced data:** one class dominates; accuracy misleads.
- **Curse of dimensionality:** many features → sparse data, meaningless distances.
- **PCA:** deterministic linear reduction keeping max-variance components (global).
- **t-SNE:** stochastic probabilistic 2D visualization keeping local neighborhoods.
- **Perplexity:** t-SNE neighbor-count hyperparameter (~30 good).

### Formulas
| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) |
| Recall (Sensitivity, TPR) | TP / (TP + FN) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Specificity (TNR) | TN / (TN + FP) |
| F1-Score | 2 · (P · R) / (P + R) |
| Normalization (Min-Max) | (x − min) / (max − min) → [0,1] |
| Standardization (Z-score) | (x − μ) / σ → mean 0, std 1 |
| KL Divergence | Σ(i≠j) p_ij · log(p_ij / q_ij) |
| Bayes | P(A\|B) = P(B\|A)·P(A) / P(B) |

### Comparisons at a glance
| | Left | Right |
|---|---|---|
| **PCA vs t-SNE** | deterministic, linear, global, fast, fights noise | stochastic, probabilistic, local, slow, poor w/ noise |
| **Precision vs Recall** | correct among predicted-positive | found among actual-positive |
| **Recall vs Specificity** | TP among actual positives | TN among actual negatives |
| **Normalize vs Standardize** | [0,1] range | mean 0, std 1 |

### Confusion Matrix (memorize layout)
| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

### Visualizations
- **PCA plot:** overlapping/messy clusters, globally meaningful distances.
- **t-SNE plot:** clean, separated clusters; between-cluster distances NOT meaningful.
- **ROC curve:** TPR vs FPR; AUC=1 perfect, 0.5 random.

### Frequently Confused
- **Precision ≠ Recall** · **Accuracy misleads on imbalanced data** · **FN vs FP** (FN dangerous in medicine) · **PCA global vs t-SNE local** · **Normalization vs Standardization** · **Validation vs Test set.**

---

# 8. Lecture Summary

## Top 50 Most Probable Exam Questions
1. Order the stages of the data mining pipeline.
2. Order the 4 preprocessing sub-pipeline steps.
3. Define train, validation, and test sets.
4. What is the validation set used for? (hyperparameter tuning)
5. Define a hyperparameter with an example.
6. What is inference?
7. What is overfitting?
8. Why split data before scaling? (data leakage)
9. What is feature engineering?
10. What is One-Hot Encoding and why use it?
11. How many columns does one-hot encoding create for 3 categories? (3)
12. Why not label-encode nominal categories as 1,2,3?
13. What is imputation?
14. Name ways to handle missing data.
15. What is imbalanced data?
16. Why is accuracy misleading on imbalanced data?
17. Name techniques for imbalanced data (SMOTE, over/undersample).
18. Normalization formula and range.
19. Standardization formula and result.
20. Normalization vs Standardization — when to use each.
21. Which algorithms need feature scaling? (distance-based)
22. What is the curse of dimensionality?
23. Why does dimensionality reduction help?
24. What is PCA?
25. What structure does PCA preserve? (global)
26. Is PCA deterministic or stochastic? (deterministic)
27. What is a principal component?
28. What is t-SNE used for?
29. Is t-SNE deterministic or stochastic? (stochastic)
30. What distributions does t-SNE use? (Gaussian high-D, t-Student low-D)
31. What does t-SNE minimize? (KL divergence)
32. How does t-SNE optimize? (gradient descent, attract/repel)
33. Write the KL divergence formula.
34. What is perplexity?
35. What is a good perplexity value and why? (~30)
36. Compare PCA vs t-SNE (full table).
37. Can PCA and t-SNE be combined? How?
38. What is UMAP?
39. Given two plots, which is PCA and which is t-SNE?
40. Draw/label the confusion matrix.
41. Define TP, FP, TN, FN.
42. Precision formula and meaning.
43. Recall formula and meaning.
44. Accuracy formula.
45. Specificity formula and meaning.
46. F1-Score formula and when to use it.
47. What does ROC plot? What does AUC mean?
48. In cancer detection, which error is worse and which metric matters? (FN; Recall)
49. Recall vs Specificity difference.
50. Why is "it not always about accuracy"?

## Top 30 Definitions
1. **Data mining pipeline** – iterative process from problem to monitored model.
2. **Sample size** – amount of data needed to train reliably.
3. **Training set** – data used to learn model weights.
4. **Validation set** – data used to tune hyperparameters.
5. **Test set** – held-out data for final unbiased evaluation.
6. **Hyperparameter** – setting chosen before training.
7. **Inference** – predicting on new data.
8. **Overfitting** – fits training data but fails on new data.
9. **Feature engineering** – creating useful features from raw data.
10. **One-Hot Encoding** – one binary column per category.
11. **Imputation** – filling missing values.
12. **Imbalanced data** – one class dominates the dataset.
13. **SMOTE** – synthetic oversampling of the minority class.
14. **Normalization** – scale to [0,1].
15. **Standardization** – transform to mean 0, std 1.
16. **Curse of dimensionality** – sparsity + distance breakdown in high dimensions.
17. **Dimensionality reduction** – fewer features while keeping structure.
18. **PCA** – deterministic linear reduction keeping max variance.
19. **Principal component** – new axis capturing maximum variance.
20. **Variance (in PCA)** – spread of data captured by a component.
21. **t-SNE** – stochastic probabilistic 2D/3D visualization of local structure.
22. **Perplexity** – t-SNE hyperparameter controlling neighbor count.
23. **KL divergence** – measure of difference between two probability distributions.
24. **Gradient descent** – iterative optimization to minimize a loss.
25. **UMAP** – manifold-based reduction/visualization method.
26. **Confusion matrix** – table of TP/FP/TN/FN.
27. **Precision** – correct fraction among predicted positives.
28. **Recall / Sensitivity** – found fraction among actual positives.
29. **Specificity** – correct fraction among actual negatives.
30. **F1-Score** – harmonic mean of precision and recall.

## Top 30 Keywords
Pipeline · Sample size · Train/Test/Validation · Hyperparameter · Inference · Overfitting · Data leakage · Feature engineering · One-Hot Encoding · Imputation · Missing data · Imbalanced data · SMOTE · Normalization · Standardization · Feature scaling · Curse of dimensionality · Dimensionality reduction · PCA · Principal component · Variance · t-SNE · Perplexity · Gaussian · t-Student distribution · KL divergence · Gradient descent · UMAP · Confusion matrix · Precision/Recall/Accuracy/Specificity · F1 · ROC/AUC · Sensitivity.

## Top 20 Comparisons
1. PCA vs t-SNE
2. Precision vs Recall
3. Recall vs Specificity
4. Normalization vs Standardization
5. Training vs Validation vs Test set
6. Validation vs Test set
7. Deterministic vs Stochastic (PCA vs t-SNE)
8. Linear vs Probabilistic embedding
9. Global vs Local structure preservation
10. Gaussian (high-D) vs t-Student (low-D) in t-SNE
11. Accuracy vs F1 (balanced vs imbalanced data)
12. FN vs FP (which is worse in medicine)
13. Sensitivity vs Specificity
14. Oversampling vs Undersampling
15. Drop vs Impute (missing data)
16. Label encoding vs One-Hot encoding
17. High vs Low perplexity
18. PCA plot vs t-SNE plot (visual)
19. Fast vs Slow (PCA vs t-SNE)
20. Fights noise vs Poor with noise (PCA vs t-SNE)

## Top 20 Formulas
1. Precision = TP/(TP+FP)
2. Recall = TP/(TP+FN)
3. Accuracy = (TP+TN)/(TP+TN+FP+FN)
4. Specificity = TN/(TN+FP)
5. F1 = 2·P·R/(P+R)
6. FPR = FP/(FP+TN) = 1 − Specificity
7. TPR = Recall = TP/(TP+FN)
8. Normalization = (x−min)/(max−min)
9. Standardization = (x−μ)/σ
10. KL(P‖Q) = Σ p_ij·log(p_ij/q_ij)
11. Bayes: P(A|B) = P(B|A)·P(A)/P(B)
12. Error rate = 1 − Accuracy
13. Misclassification = (FP+FN)/total
14. Prevalence = (TP+FN)/total
15. Negative Predictive Value = TN/(TN+FN)
16. False Negative Rate = FN/(TP+FN)
17. Variance captured ratio (PCA) = component variance / total variance
18. z-score interpretation: how many σ from mean
19. Min-Max to arbitrary [a,b]: a+(x−min)(b−a)/(max−min)
20. Harmonic mean idea behind F1 (penalizes imbalance between P and R)

## Top 20 Diagrams / Images
1. Data mining pipeline staircase (stages + database loop).
2. Preprocessing sub-pipeline (1→2→3→4) box diagram.
3. Preprocessing inner box (5 tasks).
4. Train/test/validation split diagram.
5. One-Hot Encoding table example.
6. Missing data illustration.
7. Imbalanced data class distribution.
8. Normalization vs standardization scaling plots.
9. Curse of dimensionality (sparse points in high-D).
10. PCA projection to principal components.
11. PCA variance-explained illustration.
12. t-SNE Gaussian centered at point (similarity vs distance curve).
13. t-SNE P vs Q similarity matrices with iterate loop.
14. t-SNE KL divergence + P/Q matrices with "?" .
15. t-SNE 2D scatter attract/repel iterations.
16. Perplexity = 2 / 4 / 5 neighbor circles.
17. Quiz: perplexity 2/30/100 (30 = 😎 best).
18. Quiz: which plot is PCA vs t-SNE (MNIST digits).
19. Confusion matrix (TP/FN/FP/TN) + benign/malignant labels.
20. Precision/Recall Venn (relevant vs selected items) & ROC curve.

## 10-Minute Revision Sheet
- **Pipeline:** Problem→Explore→Model→Communicate→Deploy→Monitor (loops). Sub: Sample→Split→Preprocess→Metrics.
- **Split first, scale after** (leakage). Train=learn, Val=tune, Test=final.
- **Preprocess:** engineer features, encode (one-hot = 1 col/category), impute missing, fix imbalance (SMOTE), scale (Norm→[0,1], Std→mean0/std1), reduce dimensions.
- **Curse of dimensionality:** high-D → sparse, distances meaningless.
- **PCA:** deterministic, linear, global, fast, fights noise, max-variance components.
- **t-SNE:** stochastic, probabilistic, local, slow, poor with noise; Gaussian→t-Student; minimize **KL divergence** via **gradient descent**; **perplexity ~30**.
- **Combine:** PCA then t-SNE. **UMAP** = newer alternative.
- **Confusion matrix:** TP/FN/FP/TN. **Precision**=TP/(TP+FP), **Recall**=TP/(TP+FN), **Accuracy**=(TP+TN)/all, **Specificity**=TN/(TN+FP), **F1**=2PR/(P+R).
- **ROC** TPR vs FPR; **AUC** 1=perfect, 0.5=random.
- **Cancer:** FN worst → maximize Recall. **"Not always about accuracy."**


---

# 9. Examples Repository

**Confusion Matrix worked example.** A cancer model on 100 patients gives: TP=40, FN=10, FP=5, TN=45.
- Precision = 40/(40+5) = 40/45 ≈ **0.89**
- Recall = 40/(40+10) = 40/50 = **0.80**
- Accuracy = (40+45)/100 = **0.85**
- Specificity = 45/(45+5) = 45/50 = **0.90**
- F1 = 2·(0.89·0.80)/(0.89+0.80) ≈ **0.84**
- Interpretation: we miss 10 real cancers (FN) → recall 0.80 may be too low for medicine.

**Imbalanced-data example.** 990 healthy, 10 sick. A model predicting "healthy" for everyone → Accuracy = 990/1000 = **99%**, but Recall = 0/10 = **0%**. Shows why accuracy alone is dangerous.

**One-Hot Encoding example.** Column "City" = {Berlin, Paris, Rome} → three columns City_Berlin, City_Paris, City_Rome, each 0/1. A row for Paris = (0,1,0).

**Normalization example.** Ages {20, 30, 40}, min=20, max=40 → normalized: (20→0), (30→0.5), (40→1.0).

**Standardization example.** Values with μ=30, σ=10: value 40 → z=(40−30)/10 = **1.0**; value 20 → z=**−1.0**.

**PCA example.** 784-pixel MNIST images reduced to a few principal components that keep most variance → faster training, less overfitting.

**t-SNE example.** MNIST digits (0–9) in 784-D visualized in 2D → t-SNE forms 10 clean, separated clusters (one per digit); PCA shows overlapping, messy clusters but meaningful global distances.

**Perplexity example (quiz).** 2 clusters × 50 points: Perplexity=2 → fragmented/misleading; Perplexity=30 → correct 2 clean clusters 😎; Perplexity=100 → clusters merge into one blob.

**Bayes example.** Rare disease (1% prevalence), test 99% accurate → a positive result still gives a relatively low posterior probability of actually being sick (base-rate effect).

**Combine PCA + t-SNE.** Reduce 784-D → 50-D with PCA (fast, denoise), then t-SNE 50-D → 2-D for a clean visualization.

---

# 10. Quiz / Practice Repository

**Q1.** Put in order: Deploy, Explore data, Problem understanding, Modelling, Communicate results, Test & Monitoring.
→ **Problem understanding → Explore data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test & Monitoring.**

**Q2.** Which set tunes hyperparameters? → **Validation set.**

**Q3.** True/False: You may scale using statistics from the whole dataset before splitting. → **False** (leakage; fit scaler on train only).

**Q4.** A "Weather" feature has Sunny/Rainy/Cloudy. One-hot columns count? → **3.**

**Q5.** 995 negatives, 5 positives, model predicts all negative. Accuracy? Recall? → **Accuracy 99.5%, Recall 0%.**

**Q6.** Convert {min=0, max=200} value 50 to Min-Max normalized. → 50/200 = **0.25.**

**Q7.** Z-score of x=85 with μ=75, σ=5? → (85−75)/5 = **2.0.**

**Q8.** PCA: deterministic or stochastic? Global or local? → **Deterministic, global.**

**Q9.** t-SNE: deterministic or stochastic? What does it minimize? → **Stochastic; KL divergence.**

**Q10.** t-SNE high-D vs low-D distributions? → **Gaussian (high-D), t-Student (low-D).**

**Q11.** Best perplexity among {2, 30, 100} for 2 clusters of 50? → **30.**

**Q12.** Two plots: one has 10 clean separated clusters, the other overlapping/messy. Which is t-SNE? → **The clean, separated one.**

**Q13.** Given TP=8, FP=2, FN=4, TN=6: Precision? Recall? Accuracy? Specificity?
→ Precision=8/10=**0.8**; Recall=8/12≈**0.67**; Accuracy=14/20=**0.70**; Specificity=6/8=**0.75**.

**Q14.** F1 when Precision=0.8, Recall=0.67? → 2·(0.8·0.67)/(0.8+0.67) ≈ **0.73.**

**Q15.** In cancer screening, which error do we most want to avoid, and which metric to maximize? → **False Negatives; Recall/Sensitivity.**

**Q16.** AUC = 0.5 means? → **Random guessing.**

**Q17.** ROC axes? → **TPR (y) vs FPR (x).**

**Q18.** Which metric best for imbalanced data — accuracy or F1? → **F1.**

**Q19.** Name a technique to synthetically balance classes. → **SMOTE.**

**Q20.** Why combine PCA before t-SNE? → **Speed up + reduce noise before local visualization.**

**Q21.** What is the curse of dimensionality in one sentence? → **As features grow, data becomes sparse and distances lose meaning.**

**Q22.** Recall vs Specificity — which covers actual negatives? → **Specificity.**

**Q23.** True/False: t-SNE cluster-to-cluster distances are globally meaningful. → **False.**

**Q24.** UMAP full form? → **Uniform Manifold Approximation and Projection.**

**Q25.** "It is not always about ____." → **accuracy.**

---