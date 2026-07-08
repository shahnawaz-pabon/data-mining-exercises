# Sprint 2 | Data Mining Pipeline

---

## Table of Contents
1. Data Mining Project Life Cycle & Domain Knowledge
2. Data Representation
3. Determining Sample Size
4. Train / Validation / Test Split
5. Feature Engineering (Creation, Encoding, Binning)
6. Handling Missing Data
7. Handling Imbalanced Data
8. Feature Scaling & Normalization
9. Dimensionality Reduction: Concepts & Curse of Dimensionality
10. PCA (Principal Component Analysis)
11. t-SNE
12. PCA vs t-SNE vs UMAP
13. Validation Metrics (Confusion Matrix, Precision, Recall, F1, ROC-AUC)
14. Bayes' Theorem in Diagnostic Testing
15. Model Training: Cost Function, Overfitting, Underfitting
16. Hyperparameters vs Model Parameters
17. Choosing the Most Suitable Model
18. **Final Revision Pack** (cheat sheet, 30 Qs, 30 MCQs, hot topics, last-10-minutes summary)

---

## 1. Data Mining Project Life Cycle & Domain Knowledge

### Core Idea
A Data Mining project is not just "run an algorithm" — it's a **cycle** of stages: understanding the problem, exploring/processing data, modelling & analysis, communicating results, deploying the model, and then testing/monitoring it forever after. You loop back to earlier stages as needed — it's not strictly linear.

### Key Points
- The cycle: **Problem understanding → Explore & process data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring → (loop)**
- **Domain knowledge** (subject-matter expertise about the data's real-world context) touches almost every stage:
  - Guides preprocessing decisions
  - Influences which models/features make sense
  - Provides context to explain results to non-technical stakeholders
- Without domain knowledge, you risk building a technically "correct" model that is practically meaningless.

### Definitions
- **Data Mining project life cycle**: The recurring set of stages (problem understanding → exploration → modelling → communication → deployment → monitoring) that a data project moves through.
- **Domain knowledge**: Expert understanding of the subject area the data comes from (e.g., medicine, finance) used to guide analysis decisions.

### Why It Matters
Skipping stages (e.g., jumping straight to modelling without understanding the problem) leads to models that solve the wrong problem, or that can't be explained/deployed. Domain knowledge prevents "garbage in, garbage out" and helps you engineer meaningful features instead of blindly throwing raw columns at a model.

### Easy Example
Imagine you're predicting hospital readmission. Without domain knowledge, you might treat "days since last visit = 0" as a normal number. A clinician tells you "0 means the patient never left" — completely changing how you handle that feature. That's domain knowledge shaping preprocessing.

### Common Mistakes
- Thinking Data Mining = "just modelling." Modelling is only one of six+ stages.
- Assuming the cycle is strictly one-directional — in practice you loop back (e.g., new problems found during modelling send you back to preprocessing).
- Ignoring domain knowledge because "the model will figure it out."

### MCQ Preparation
1. **Which stage comes immediately after "Problem understanding" in the Data Mining life cycle?**
   A) Deploy & operationalize B) Explore and process data C) Communicate results D) Test and Monitoring
   **Answer: B.** The cycle flows understanding → exploration/processing next.

2. **Domain knowledge in a Data Mining project mainly helps with:**
   A) Only writing the final report B) Guiding preprocessing, modelling, and explaining results C) Choosing the programming language D) Buying more compute
   **Answer: B.** Per the slides, domain knowledge guides preprocessing, influences modelling, and provides context for results.

3. **True or False (as MCQ): The Data Mining life cycle must always be followed in strict, one-way order.**
   A) True, no exceptions B) False — stages can be revisited C) Only Test & Monitoring can repeat D) Only true for supervised learning
   **Answer: B.**

4. **The final stage of the Data Mining life cycle (before looping again) is:**
   A) Modelling & Analysis B) Communicate results C) Test and Monitoring D) Explore data
   **Answer: C.** Monitoring an already-deployed model closes the loop.

5. **Why is domain knowledge described as "the key" in the slides?**
   A) It replaces the need for data B) It helps engineer meaningful features, focus analysis, and tell a story the audience understands C) It's only useful for medical data D) It's optional for experienced data scientists
   **Answer: B.**

### Short Questions
1. **Name the six stages of the Data Mining project life cycle.**
   Problem understanding → Explore/process data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring.
2. **What three things does domain knowledge provide, according to the slides?**
   It guides preprocessing, influences modelling, and provides context to explain results.
3. **Is the Data Mining life cycle strictly linear? Explain.**
   No — while stages have a typical order, projects often loop back (e.g., discovering a data issue during modelling sends you back to preprocessing).
4. **Give an example of how missing domain knowledge could hurt a project.**
   Any example where a "weird" data value actually has real-world meaning that a non-expert would misinterpret (e.g., a placeholder value being treated as a real measurement).
5. **Why can't domain knowledge be skipped even with a powerful model?**
   Because the model can only learn patterns present in the features you give it; without domain understanding, you may engineer the wrong features or misinterpret results.

### Memory Tricks
**"PEMCDT"** — **P**roblem, **E**xplore, **M**odel, **C**ommunicate, **D**eploy, **T**est-monitor. (Say it like a name: "Pem-C-D-T".)
Think of domain knowledge as **seasoning** — it doesn't replace the main ingredients (data + model) but without it the dish (results) is bland or wrong.

### Exam Summary (Revision Sheet)
- Life cycle stages (memorize order): Problem understanding → Explore/process → Model & Analyze → Communicate → Deploy → Test/Monitor (loop).
- It's cyclical/iterative, not strictly linear.
- Domain knowledge = guides preprocessing + influences modelling + gives context for results.
- Key exam trap: don't say the cycle is "one-way only."

---

## 2. Data Representation

### Core Idea
Before a model can learn from data, that data must be converted into a **numerical** form a computer can process. The same real-world data (audio, text, images, molecules) can be represented in more than one valid way — the "best" representation depends on context and what you want to analyze.

### Key Points
- To train a model, data must be transformed into an **informative, numerical representation**.
- Different data types → different representations:
  - **Text** → Word Embeddings (real-valued vectors) or TF-IDF
  - **Images** → pixel values
  - **Molecules** → mathematical graphs (atoms = nodes, bonds = edges)
  - **Audio** → spectrograms, raw time series, or wavelets
  - **Graphs** → Node Embeddings (e.g., Node2Vec)
- **There is no single correct representation** — the same signal can be represented multiple valid ways (e.g., audio as spectrogram OR as a 1D time series).

### Definitions
- **Word Embedding**: A real-valued vector representation of a word, capturing semantic meaning (e.g., Word2Vec, GloVe).
- **TF-IDF (Term Frequency–Inverse Document Frequency)**: Converts text into numeric vectors, weighing a word's importance by how often it appears in a document (TF) vs. across all documents (IDF).
- **Node Embedding**: A learned numeric vector representing a node in a graph, capturing structural relationships (e.g., Node2Vec).
- **Spectrogram**: A visual/numeric representation showing how an audio signal's frequency content changes over time.

### Why It Matters
Machines can't process raw text, sound waves, or images directly as "meaning" — everything must become numbers. Choosing the right representation directly affects how well a model can find patterns; the wrong representation can hide the very signal you're looking for.

### Easy Example
Think of a recipe: the same dish (data) can be "represented" as a photo, a written recipe, or a list of nutritional values. Each representation is useful for a different purpose — the dish itself hasn't changed, but how you capture it did. Similarly, a heartbeat signal (ECG) can be represented as a raw waveform or as extracted peak/interval features — same underlying data, different numeric packaging.

### Common Mistakes
- Believing there's only "one correct" representation for a dataset (exam trap!).
- Confusing Word Embeddings (semantic vectors) with TF-IDF (frequency-based weighting) — they solve similar problems differently.
- Thinking changing representation changes the underlying meaning of the data — it doesn't; it just reframes it for processing.

### MCQ Preparation
1. **Which statement best reflects how data is represented in data analysis?**
   A) Each dataset has only one possible representation B) There could be different representation ways depending on context and use C) Changing representation always changes the underlying meaning D) Data must be stored in only one format
   **Answer: B.**
2. **TF-IDF is primarily used to represent:**
   A) Images B) Textual data C) Molecular graphs D) Audio signals
   **Answer: B.**
3. **In a molecular graph representation, an atom corresponds to a:**
   A) Node B) Edge C) Weight D) Vector
   **Answer: A** (bonds = edges).
4. **Node2Vec is an example of:**
   A) Image feature extraction B) Node embedding for graph data C) Audio spectrogram D) TF-IDF variant
   **Answer: B.**
5. **An audio signal being represented as either a spectrogram or a raw time series shows that:**
   A) One of them must be wrong B) Data can have multiple valid representations C) Spectrograms are always superior D) Time series can't be used for audio
   **Answer: B.**

### Short Questions
1. **Why must data be converted into a numerical representation before modelling?** Because electronic devices/models can only manipulate numbers, not raw real-world signals directly.
2. **Give two ways textual data can be represented numerically.** TF-IDF vectors and Word Embeddings (e.g., Word2Vec, GloVe).
3. **What does a spectrogram represent?** How an audio signal's frequency content evolves over time.
4. **How are molecular structures typically represented for modelling?** As mathematical graphs — atoms as nodes, bonds as edges.
5. **Is there always one "correct" data representation? Justify.** No — the same data (e.g., audio) can validly be represented in multiple ways (spectrogram, time series, wavelets) depending on the analysis goal.

### Memory Tricks
**"WIT-GN"**: **W**ord embeddings, **I**mage pixels, **T**F-IDF (text), **G**raphs (molecules), **N**ode2Vec (graphs) — five data types, five representations.

---

## 3. Determining Sample Size

### Core Idea
Sample size (how much data you collect) affects almost everything downstream: accuracy, generalizability, overfitting risk, and even which algorithms you can use. More data is usually better, but "bigger" is not automatically "correct" — quality and representativeness matter too.

### Key Points
- **Statistical power & accuracy**: Larger samples reduce Type II errors (failing to detect a real effect) and improve estimation accuracy.
- **Generalizability**: Small samples may not represent the population well (biased results); larger, diverse samples generalize better.
- **Variability & overfitting**: Small samples → high variance → models memorize noise instead of learning patterns → overfitting risk.
- **Choice of algorithm**: Deep learning needs large datasets; decision trees/k-NN can work with smaller samples.
- **Class imbalance**: Small samples in classification can make rare classes disappear/underrepresented.
- **Computational cost trade-off**: Bigger data = more accurate but more expensive to process; sampling techniques (stratified sampling, bootstrapping) balance this.
- **Big data isn't automatically better**: it reduces noise but does NOT fix bias or poor data quality, and can make tiny/irrelevant effects look statistically significant.

### Definitions
- **Type II Error**: Failing to detect a real effect that actually exists (false negative in a statistical test).
- **Overfitting**: When a model learns noise/specifics of the training data instead of generalizable patterns.
- **Stratified sampling**: Sampling that preserves the proportion of subgroups (e.g., classes) found in the full population.

### Why It Matters
Choosing sample size wrong can produce a model that looks great on paper (high accuracy on a tiny biased sample) but fails completely on new, real-world data — this is one of the most common causes of failed data mining projects.

### Easy Example
Imagine polling 10 people outside one coffee shop to predict a national election vs. polling 10,000 people across the country. The first sample is small AND unrepresentative (biased location) — both problems, not just size.

### Common Mistakes
- Believing "larger dataset = always more accurate conclusions" (exam trap — large data can make trivial effects look important, i.e., statistically significant but practically meaningless).
- Believing more data automatically removes bias — it does not.
- Confusing "small sample = always unreliable" — some algorithms (k-NN, decision trees) can work reasonably with smaller data.

### MCQ Preparation
1. **Which statement is TRUE about sample size?**
   A) Larger datasets always produce more accurate conclusions B) Increasing sample size eliminates bias C) Large datasets can make insignificant effects appear important D) Small datasets are always unreliable
   **Answer: C.**
2. **A small sample size primarily increases the risk of:**
   A) Type I error only B) Overfitting due to high variance C) Guaranteed underfitting D) Perfect generalization
   **Answer: B.**
3. **Which algorithm generally requires the LARGEST amount of data to perform well?**
   A) k-NN B) Decision Trees C) Deep Learning D) None require much data
   **Answer: C.**
4. **What technique helps balance efficiency and accuracy when data is very large?**
   A) Ignoring most of the data randomly B) Stratified sampling / bootstrapping C) Always using the entire dataset regardless of cost D) Removing all minority classes
   **Answer: B.**
5. **What is a Type II error?**
   A) Detecting an effect that isn't real B) Failing to detect a real effect C) A data entry mistake D) A scaling error
   **Answer: B.**

### Short Questions
1. **Why does a larger sample size reduce Type II errors?** It gives more statistical power, making real patterns/effects easier to reliably detect.
2. **Does more data fix poor data quality? Explain.** No — large samples reduce random noise but do not correct bias or poor quality; garbage data stays garbage regardless of volume.
3. **Why might a small sample cause overfitting?** With few data points, models can end up memorizing noise/specific quirks rather than learning generalizable patterns.
4. **Name two things to consider when you have "too much" data.** Sampling techniques for a representative subset, and big data frameworks/learning curves to find a sufficient sample size.
5. **How does sample size interact with class imbalance?** A small sample may barely include any examples of a rare class, biasing the model toward the majority class.

### Memory Tricks
**"SAVE-C"**: **S**tatistical power, **A**ccuracy, **V**ariability/overfitting, **E**xtent of generalizability, **C**hoice of algorithm & **C**lass imbalance & **C**ompute cost — all things sample size affects.
Big data ≠ Good data — remember: **"More can hide, not fix, the mess."**

---

## 4. Train / Validation / Test Split

### Core Idea
We split our dataset into three parts — **training** (to learn parameters), **validation** (to tune hyperparameters/choose the model), and **testing** (to get an honest final performance estimate) — so we can trust that our model will work on new, unseen data, not just memorize what it has already seen.

### Key Points
- **Training data** → fed into the learning algorithm to fit the model's internal parameters.
- **Validation data** → used to check the model as it develops and to decide when to change hyperparameters (not final parameters).
- **Testing data** → used only at the end, to report final, unbiased performance.
- The goal is to **maximize generalization**: avoid **underfitting** (model too simple, doesn't even fit training data well) and avoid **overfitting** (model fits training data too well, including its noise, and fails on new data).
- At **inference** (real-world use), you only have the **final model** + new data point — no labels, no more training.

### Definitions
- **Training set**: Data used to fit/learn a model's parameters.
- **Validation set**: Data used during development to tune hyperparameters and select between candidate models.
- **Test set**: Held-out data used once, at the end, to estimate real-world performance.
- **Inference**: Using a finished, trained model to make predictions on brand-new data.
- **Generalization**: A model's ability to perform well on data it has never seen before.

### Why It Matters
If you tune your model based on test-set performance, the test set is no longer "unseen" — your performance estimate becomes optimistic and misleading. Keeping the three sets separate protects the honesty of your evaluation.

### Easy Example
Think of studying for an exam: **training** = studying the textbook, **validation** = practice quizzes (you adjust your study approach based on quiz mistakes), **test** = the real final exam (you don't get to "adjust" after seeing it — it's the true measure).

### Common Mistakes
- Using the test set to tune hyperparameters (this "leaks" information and inflates apparent performance).
- Thinking validation and test do the same job — validation shapes the model during development; test only evaluates the finished model.
- Forgetting that during inference, there are no labels — you're only predicting.

### MCQ Preparation
1. **Which dataset is used to decide when to change a model's hyperparameters?**
   A) Training set B) Validation set C) Test set D) Inference set
   **Answer: B.**
2. **What is the main risk of NOT splitting data into train/validation/test?**
   A) Faster training only B) You can't tell if the model generalizes or just memorized the data C) The model becomes more interpretable D) There's no risk
   **Answer: B.**
3. **At inference time, what does the model receive?**
   A) Training labels B) A new, unlabeled data point C) The validation error D) The full original dataset
   **Answer: B.**
4. **Overfitting is best described as:**
   A) High bias, low variance B) The model fits training data well but fails on unseen data C) The model fits nothing well D) A synonym for underfitting
   **Answer: B.**
5. **The ultimate goal of the train/validation/test split is to:**
   A) Reduce dataset size B) Maximize generalization to unseen data C) Make training faster D) Avoid using validation data
   **Answer: B.**

### Short Questions
1. **What is each of the three data splits used for?** Training → learn parameters; Validation → tune hyperparameters/select model; Test → final unbiased performance estimate.
2. **Why shouldn't you tune hyperparameters using the test set?** Because it would make the test set no longer "unseen," giving an overly optimistic, misleading performance estimate.
3. **What's the difference between overfitting and underfitting?** Overfitting = model too complex, fits training noise, fails on new data; underfitting = model too simple, fails to capture patterns even in training data.
4. **What happens differently at inference vs. training?** At inference, the model only receives a new data point (no labels, no more parameter learning) — it just outputs a prediction using the already-finalized model.
5. **Why is maximizing generalization the key goal, not just training accuracy?** Because the model's real value comes from performing well on new, real-world data, not just memorizing the training set.

### Memory Tricks
**"Study, Quiz, Exam"** = Train, Validate, Test. You adjust after quizzes (validation), never after the real exam (test).
Underfitting = "didn't study enough" (misses patterns). Overfitting = "memorized the textbook word-for-word" (fails when questions are phrased differently).

---

## 5. Feature Engineering: Creation, Encoding, Binning

### Core Idea
Feature engineering means transforming raw data into new, more useful inputs ("features") for your model. This includes creating new features from existing ones, converting categories into numbers (encoding), and grouping numeric ranges into categories (binning).

### Key Points
**A. Feature Creation / Transformation**
- **Feature Creation**: build new features from existing ones (e.g., speed = distance / time).
- **Combining (interaction terms)**: e.g., income per hour = income / working hours.
- **Aggregations**: mean, sum, count over groups.
- **Datetime extraction**: pull out hour, day, month, season from timestamps.
- **Text feature extraction**: word counts, sentiment, embeddings.
- The right technique depends on: data type (numeric/categorical/text/image), the problem type (classification/regression/clustering), and the model being used.

**B. Feature Encoding**
- Most ML algorithms can't use raw categorical text (e.g., "Toyota", "Ford") — must convert to numbers.
- **Label Encoding** (replacing categories with plain numbers like 0,1,2) is risky: it creates a **false ordinal relationship** (model thinks Ford(1) > Toyota(0), or that categories have "distance" between them), which can mislead distance/math-based models (linear regression, logistic regression, k-NN).
- **One-Hot Encoding**: creates a separate binary (0/1) column per category (e.g., "Toyota" column, "Ford" column, "Tesla" column).
- **Dummy variable trap**: after one-hot encoding, you should **drop one column** — keeping all of them causes **multicollinearity** (redundant information that can distort the model, since the last category is always fully implied by the others being 0).

**C. Feature Binning (Bucketing)**
- Groups numeric subranges into bins/buckets, often turning numeric data into categorical data.
- Example: "Age" (0–130) → bins like Infant (0–1), Toddler (1–3), Teen (12–18), etc.
- Effect: values that are close but fall in different bins (e.g., 1 and 1.5) get treated as very different; values far apart but in the same bin (e.g., 6 and 10) get treated as identical.

### Definitions
- **Label Encoding**: Assigning each category a distinct integer (e.g., Red=0, Green=1, Blue=2).
- **One-Hot Encoding**: Representing each category as its own binary column (1 if present, 0 otherwise).
- **Multicollinearity**: When one feature can be predicted from others (redundant information), distorting a model's parameter estimates.
- **Binning/Bucketing**: Converting a continuous numeric range into discrete categorical intervals.

### Why It Matters
Bad encoding introduces false mathematical relationships that mislead the model (e.g., "Blue is twice Green"). Good feature engineering, meanwhile, injects domain knowledge directly into the numbers the model sees, often boosting performance more than switching algorithms would.

### Easy Example
Encoding "Small/Medium/Large" as 0/1/2 is fine because there IS a real order. But encoding "Cat/Dog/Bird" as 0/1/2 is wrong — there's no natural order, so one-hot encoding (3 separate 0/1 columns, then drop one) is correct.

### Common Mistakes
- Using label encoding on **non-ordinal** categories (exam favorite trap).
- Forgetting to drop one one-hot column → multicollinearity.
- Assuming binning never loses information — it does (values in the same bin become indistinguishable).
- Confusing "feature creation" (building new features) with "feature selection" (choosing among existing ones) — different concepts.

### MCQ Preparation
1. **Why is label encoding risky for non-ordinal categories like car brands?**
   A) It's slower to compute B) It introduces a false ordinal relationship the model may misuse C) It always causes underfitting D) It cannot be reversed
   **Answer: B.**
2. **Why do we typically drop one column after one-hot encoding?**
   A) To save memory only B) To avoid multicollinearity/redundant information C) Because models can't handle more than 2 columns D) It's not necessary, just a style choice
   **Answer: B.**
3. **Binning converts:**
   A) Categorical data into text B) Numerical data into categorical buckets C) Text into images D) Images into categories
   **Answer: B.**
4. **A model treats a value of 1 and 1.5 very differently after binning even though they're numerically close. Why?**
   A) A bug in the model B) They fall into different bins C) Binning always increases precision D) The model ignores numeric values entirely
   **Answer: B.**
5. **Which encoding is more appropriate for a truly ordinal feature like "Low/Medium/High"?**
   A) One-hot encoding always B) Label encoding (since order is meaningful) C) Binning D) TF-IDF
   **Answer: B.**

### Short Questions
1. **What problem does label encoding create for non-ordinal categories?** It implies a false order/distance between categories, which can mislead distance- or math-based models.
2. **What is the "dummy variable trap" and how do we avoid it?** Keeping all one-hot columns creates multicollinearity (redundancy); we avoid it by dropping one column, since it's implied by the rest.
3. **Give an example of feature creation from two existing raw features.** Speed = distance ÷ time (from raw distance and time columns).
4. **What does binning do to numeric data, and what's a downside?** It groups numeric ranges into categories; downside — values close together in different bins are treated as very different, and values far apart in the same bin are treated as identical.
5. **List two datetime-based features you could extract from a raw timestamp.** Hour of day, day of week, month, or season.

### Memory Tricks
**"LOB"**: **L**abel encoding → risk of false **O**rder; one-hot → drop one to avoid **B**ias (multicollinearity).
Binning = "putting people into age groups at a party" — a 1-year-old and 2.9-year-old are both "toddlers," even though they're quite different in reality.

---

## 6. Handling Missing Data

### Core Idea
Real-world data almost always has gaps. Before fixing missing data, you must understand **why** it's missing (the pattern), because the wrong fix can introduce serious bias — the classic example being **survivorship bias**.

### Key Points
- Steps: (1) Define the pattern of missingness → (2) Process it (drop, replace, impute) → (3) Consider algorithms robust to missing data.
- Missing data can appear as **missing values** (a blank cell) or **missing categories/records entirely** (data that never made it into your sample at all — survivorship bias).
- **Strategies to handle missing values:**
  1. Ignore/drop records with missing data
  2. Add an "NA indicator" variable (flag that a value was missing) as a model input
  3. Use a model/method that natively tolerates missing data
  4. **Impute** (fill in) missing values — essentially an educated guess
- **Imputation strategies:**
  - **Manual imputation**: domain expert fills values
  - **Global constant**: replace with "unknown" or a sentinel like -999
  - **Statistical imputation**: mean/median of the variable, or class-wise mean/median (if labels available)
  - **Model-based imputation**: predict missing values using regression/decision trees/other models

### Definitions
- **Survivorship bias**: A bias where you only analyze the data/cases that "survived" some selection process, ignoring the ones that didn't — leading to wrong conclusions.
- **Imputation**: The process of filling in missing values with estimated ones.
- **NA indicator variable**: A binary flag column showing whether a value was originally missing.

### Why It Matters
The famous WWII example: engineers wanted to reinforce planes based on where returning planes had bullet holes. But planes hit in OTHER areas never made it back — so those "missing" (non-returning) planes are the real signal. Reinforcing where survivors were hit would have been exactly wrong; you should reinforce the areas with **no** damage on survivors, because that's where hits were fatal.

### Easy Example
A customer satisfaction survey only reaches people who are still customers (churned customers are "missing" from the data). If you only analyze current customers' satisfaction, you'll systematically miss why people leave — a modern survivorship bias example.

### Common Mistakes
- Treating all missingness the same (missing-at-random vs. missing-not-at-random require different fixes).
- Forgetting survivorship bias — assuming the data you have is a full unbiased picture.
- Blindly imputing with the mean without checking whether missingness itself is informative (in which case an NA indicator variable is better).

### MCQ Preparation
1. **In the WWII returning-planes example, where should reinforcement be added?**
   A) Where returning planes show the most damage B) Where returning planes show NO damage (since hits there likely meant the plane didn't return) C) Randomly across the plane D) Only the cockpit
   **Answer: B.**
2. **Survivorship bias occurs when:**
   A) You analyze only the cases that "survived" a selection process, ignoring the rest B) You impute missing values with the mean C) You drop all missing records D) You use model-based imputation
   **Answer: A.**
3. **Which imputation method uses other variables to predict the missing one?**
   A) Global constant B) Manual imputation C) Model-based imputation D) Mean imputation
   **Answer: C.**
4. **An "NA indicator variable" is used to:**
   A) Delete missing rows B) Flag that a value was originally missing, as its own model input C) Replace missing values with -999 D) Encode categorical data
   **Answer: B.**
5. **The FIRST step in handling missing data should be to:**
   A) Immediately impute with the mean B) Define the pattern of missingness C) Delete the entire dataset D) Ignore it
   **Answer: B.**

### Short Questions
1. **What is survivorship bias? Give an example.** Analyzing only surviving/selected cases while ignoring those that didn't make it into the sample, leading to wrong conclusions — e.g., reinforcing WWII planes based only on returning aircraft's bullet holes.
2. **List the four general strategies for handling missing values.** Ignore/drop records, add an NA indicator variable, use a missing-data-robust method, or impute.
3. **Name the four types of imputation.** Manual, global constant, statistical (mean/median or class-wise), and model-based.
4. **Why might mean imputation be a bad choice sometimes?** It can hide the fact that missingness itself carries information, and it can distort variance/distribution of the feature.
5. **What should you do BEFORE choosing a missing-data strategy?** Understand and define the pattern of missingness (why/how the data is missing).

### Memory Tricks
**"DPC"**: **D**efine the pattern → **P**rocess it → **C**onsider robust models.
Survivorship bias mnemonic: **"Dead men tell no tales — but they hold the real data."** (The planes that didn't return are the missing data that matters most.)

---

## 7. Handling Imbalanced Data

### Core Idea
**Imbalanced data** is when one class (category) appears far more often than another (e.g., 98% "no fraud", 2% "fraud"). Standard models tend to just predict the majority class and still look "accurate," so special strategies are needed.

### Key Points
Four main strategies:
1. **Use a model/objective function suited to imbalance** (e.g., cost-sensitive learning)
2. **Data resampling techniques**: **downsample** the majority class (remove some majority examples) or **upsample** the minority class (duplicate/synthesize more minority examples)
3. **Data augmentation**: create new synthetic examples of the rare class
4. **Use balanced/suited evaluation metrics** (accuracy is misleading here — use precision, recall, F1, etc., covered later)

### Definitions
- **Imbalanced data**: A dataset where some classes/categories occur much less frequently than others.
- **Downsampling**: Reducing the number of majority-class examples to balance class ratios.
- **Upsampling**: Increasing the number of minority-class examples (duplication or synthesis) to balance class ratios.

### Why It Matters
If 98% of your data is "not fraud," a model that always predicts "not fraud" gets 98% accuracy — while being completely useless. Correcting for imbalance ensures the model actually learns to detect the rare, often more important, class (fraud, disease, defect).

### Easy Example
A spam filter trained on 990 real emails and only 10 spam emails will likely just learn to always predict "not spam." Upsampling those 10 spam emails (or downsampling the 990 real ones) helps the model actually learn spam patterns.

### Common Mistakes
- Using plain **accuracy** to judge a model on imbalanced data (huge exam trap — see Section 13).
- Thinking resampling is the only fix — balanced metrics and cost-sensitive models are equally valid approaches.
- Upsampling by just duplicating rows blindly, which can encourage overfitting to those duplicated examples (augmentation/synthesis is often better).

### MCQ Preparation
1. **Which of these is NOT one of the four strategies for handling imbalanced data mentioned in the slides?**
   A) Data resampling B) Using balanced evaluation metrics C) Always deleting the minority class D) Data augmentation
   **Answer: C.**
2. **Downsampling refers to:**
   A) Increasing minority-class samples B) Reducing majority-class samples C) Increasing all classes D) Normalizing features
   **Answer: B.**
3. **Why is accuracy a poor metric for imbalanced datasets?**
   A) It's too slow to compute B) A model predicting only the majority class can still score high accuracy C) It requires labeled data D) It cannot be computed for classification
   **Answer: B.**
4. **Data augmentation for imbalanced data is used to:**
   A) Remove majority class samples B) Create synthetic examples of the rare class C) Normalize numerical features D) Encode categorical variables
   **Answer: B.**
5. **A cost-sensitive learning approach addresses imbalance by:**
   A) Ignoring the minority class entirely B) Penalizing misclassification of the minority class more heavily C) Removing all majority samples D) Standardizing the data
   **Answer: B.**

### Short Questions
1. **Define imbalanced data.** A dataset where some class/category occurs much less frequently than others.
2. **What's the difference between downsampling and upsampling?** Downsampling reduces majority-class examples; upsampling increases minority-class examples (via duplication or synthesis).
3. **Why can't we just rely on accuracy for imbalanced classification?** A trivial "always predict majority class" model achieves high accuracy while being practically useless at detecting the rare class.
4. **Name the four general strategies for imbalance.** Suitable model/objective function, data resampling, data augmentation, and balanced evaluation metrics.
5. **Give a real-world example where imbalanced data is a natural challenge.** Fraud detection, rare disease diagnosis, or manufacturing defect detection.

### Memory Tricks
**"MRAB"**: **M**odel choice, **R**esampling, **A**ugmentation, **B**alanced metrics — the 4 fixes for imbalance.
Remember: **"98% accuracy can still mean 0% useful"** when data is imbalanced.

---

## 8. Feature Scaling & Normalization

### Core Idea
Features often live on very different numeric scales (e.g., house size in m² vs. number of rooms). Many algorithms compare/combine features mathematically, so without scaling, large-scale features unfairly dominate the model.

### Key Points
- **Why scaling matters:**
  - Ensures features contribute equally to the model
  - Prevents large-scale features from dominating smaller-scale ones
  - **Required** for distance-based algorithms (KNN, SVM, clustering)
  - Speeds up convergence in gradient-based algorithms (neural networks)
  - Helps avoid exploding/vanishing gradients in deep learning
- **Common scaling methods:**

| Method | Description | Typical Use Case |
|---|---|---|
| **Min-Max Scaling** | Rescales values to [0, 1] | Neural networks, image data |
| **Z-score (Standardization)** | Centers data to mean 0, std 1 | Most ML algorithms, general purpose |
| **Robust Scaling** | Uses median and IQR, resistant to outliers | Datasets with heavy-tailed/outlier-prone features |

- **Golden rule**: Always **fit** the scaler on training data ONLY. The learned parameters (mean/std, min/max) become **part of the model** and must be saved and reapplied identically to test/production data — never re-fit on test data.

### Definitions
- **Feature Scaling**: Transforming features onto comparable numeric ranges/distributions.
- **Standardization (Z-score)**: Transforming a feature to have mean = 0 and standard deviation = 1.
- **Normalization (Min-Max)**: Rescaling a feature to a fixed range, typically [0, 1].
- **Formula (Z-score)**:
  - μ = (1/M) Σ x⁽ⁱ⁾ (mean)
  - σ² = (1/M) Σ (x⁽ⁱ⁾ − μ)² (variance)
  - x_normalized = (x − μ) / σ² *(as shown in slides; conventionally divided by σ, the standard deviation)*
  - Where **μ** = mean, **σ²** = variance, **σ** = standard deviation, **M** = number of samples.

### Why It Matters
Without scaling, a feature like "house size" (values in the hundreds) will dominate distance calculations over "number of rooms" (values 1–5) in KNN/SVM/clustering — even if rooms matter just as much for the real-world problem. Scaling levels the playing field.

### Easy Example
Comparing House A (50m², 2 rooms) and House B (150m², 3 rooms): the raw size difference (100) numerically swamps the room difference (1). A distance-based model will "see" size and barely notice rooms — scaling fixes this imbalance.

### Common Mistakes
- Fitting the scaler on the **entire dataset** (including test data) — this leaks test information into training (data leakage).
- Forgetting to save and reapply the exact same scaling parameters at deployment/production time.
- Using Min-Max scaling on data with extreme outliers (it compresses everything else into a tiny range) — Robust Scaling is better there.

### MCQ Preparation
1. **Which scaling method uses median and IQR, making it resistant to outliers?**
   A) Min-Max Scaling B) Z-score Standardization C) Robust Scaling D) One-hot encoding
   **Answer: C.**
2. **Feature scaling parameters (mean, std, min, max) should be fit on:**
   A) The entire dataset including test data B) Training data only, then reapplied to test/production data C) Test data only D) Randomly selected data each time
   **Answer: B.**
3. **Which algorithms REQUIRE feature scaling to work properly (distance-based)?**
   A) Decision Trees B) KNN, SVM, clustering C) Naive Bayes only D) None require it
   **Answer: B.**
4. **Min-Max scaling rescales values to:**
   A) Mean 0, std 1 B) [0, 1] range C) [-1, 1] always D) The original range unchanged
   **Answer: B.**
5. **Why must scaling parameters be saved and reused in production?**
   A) They aren't actually needed in production B) Because they are part of the trained model and must transform new data identically to training data C) To make the model file smaller D) Scaling isn't needed after training
   **Answer: B.**

### Short Questions
1. **Why is feature scaling especially important for distance-based algorithms?** Because they compute distances directly from feature values; unscaled features with larger numeric ranges dominate the distance calculation.
2. **What's the difference between Min-Max scaling and Z-score standardization?** Min-Max rescales to a fixed [0,1] range; Z-score centers data to mean 0 and standard deviation 1.
3. **Why should you fit a scaler only on training data?** To avoid data leakage — using test data statistics would let information from the test set improperly influence training.
4. **When would you prefer Robust Scaling over Min-Max?** When the dataset has outliers/heavy tails that would otherwise distort a Min-Max range.
5. **Give an example showing why unscaled features can mislead a model.** House size (0–300) vs. number of rooms (1–5): unscaled, size differences dominate distance-based comparisons even though rooms matter equally.

### Memory Tricks
**"Fit on Train, Apply to All"** — the golden scaling rule.
Min-Max = "squeeze into a box [0,1]." Z-score = "center on zero, unit spread." Robust = "ignore the outlier drama, use the median."

---

## 9. Dimensionality Reduction: Concepts & Curse of Dimensionality

### Core Idea
When a dataset has too many features (dimensions), models can become slow, unreliable, and prone to overfitting. **Dimensionality reduction** transforms data into fewer, more informative dimensions, while **feature selection** simply drops some original features without transforming them.

### Key Points
- **Feature Selection**: reduce dimensionality by **excluding** features, without changing the remaining ones.
- **Dimensionality Reduction**: **transforms** features into a lower-dimensional space (e.g., 1000 features → 10 new features) using linear or nonlinear techniques.
- **Curse of Dimensionality**: As dimensionality increases, classification first gets easier — up to an optimal number of features — but beyond that point, adding more dimensions (without more data) makes classification HARDER and hurts performance, because data becomes increasingly **sparse** in high-dimensional space.
- Benefits of dimensionality reduction:
  - Reduces computational complexity
  - Helps separate classes/groups more clearly (e.g., 3D → 2D can make classes more distinguishable)
  - Essential for **visualizing** high-dimensional data (e.g., scRNA-seq data with 30,000 genes per cell)
- **Two main types of dimensionality reduction:**
  - **Linear**: PCA, LDA, ICA
  - **Non-linear**: t-SNE, UMAP, Autoencoders

### Definitions
- **Feature Selection**: Choosing a subset of the original features to keep, discarding the rest, without transforming any of them.
- **Dimensionality Reduction**: Mathematically transforming high-dimensional data into a lower-dimensional representation while preserving important structure/information.
- **Curse of Dimensionality**: The phenomenon where increasing the number of features (without proportionally more data) increases data sparsity and eventually degrades model performance.
- **Sparsity**: The situation where data points become spread thin/far apart from each other as dimensions increase.

### Why It Matters
Too many features relative to your sample size causes models to struggle to find real patterns among the "noise" of extra dimensions — this hurts both speed and accuracy. Dimensionality reduction is also often the ONLY way to visualize datasets with thousands of features (like gene expression data).

### Easy Example
Imagine trying to find a friend in a single hallway (1D) — easy. Now imagine finding them in an empty football stadium (many dimensions) — even though they're still "findable," the space is so vast and sparse that it's much harder unless you have tons of searchers (data points).

### Common Mistakes
- Confusing feature selection (keeps some original features as-is) with dimensionality reduction (creates brand-new transformed features).
- Thinking "more features = always better" — the curse of dimensionality proves otherwise.
- Assuming sparsity is good — it actually makes finding patterns harder.

### MCQ Preparation
1. **What's the key difference between feature selection and dimensionality reduction?**
   A) They are the same thing B) Feature selection excludes features as-is; dimensionality reduction transforms features into new ones C) Feature selection is always nonlinear D) Dimensionality reduction can only be applied to images
   **Answer: B.**
2. **The "curse of dimensionality" refers to:**
   A) Classification always improving with more dimensions B) Performance degrading past an optimal number of features (without more data) due to sparsity C) A bug specific to PCA D) A rule that only applies to text data
   **Answer: B.**
3. **Which of the following is a LINEAR dimensionality reduction technique?**
   A) t-SNE B) UMAP C) PCA D) Autoencoders
   **Answer: C.**
4. **Which of the following is a NON-linear dimensionality reduction technique?**
   A) PCA B) LDA C) t-SNE D) ICA
   **Answer: C.**
5. **Dimensionality reduction is especially essential for:**
   A) Increasing the number of features B) Visualizing high-dimensional data (e.g., gene expression) C) Removing all categorical variables D) Always improving accuracy regardless of data**
   **Answer: B.**

### Short Questions
1. **Distinguish feature selection from dimensionality reduction.** Feature selection drops some original features unchanged; dimensionality reduction mathematically transforms all features into fewer new ones.
2. **Explain the curse of dimensionality in your own words.** Adding more features without more data makes data sparser in high-dimensional space, which eventually hurts model performance instead of helping it.
3. **Name three linear and two non-linear dimensionality reduction techniques.** Linear: PCA, LDA, ICA. Non-linear: t-SNE, UMAP (also Autoencoders).
4. **Why is dimensionality reduction useful for scRNA-seq data?** Because such data can have up to 30,000 gene-features per cell; reducing dimensions makes it possible to visualize and analyze the gene-cell matrix.
5. **Why does reducing dimensions sometimes make classes easier to separate?** Removing noisy/redundant dimensions can reveal the underlying structure more clearly, similar to how 3D data projected into 2D can show cleaner class separation.

### Memory Tricks
**"Select vs. Transform"**: Feature **Selection** = pick and choose (keep originals). Dimensionality **Reduction** = blend and reshape (new features).
Curse of dimensionality = **"Too many cooks spoil the broth"** — too many features (without enough data) confuse rather than help the model.

---

## 10. PCA (Principal Component Analysis)

### Core Idea
PCA is a **linear** technique that reduces the number of features while keeping as much of the original **variance (information)** as possible. It does this by rotating the coordinate system to line up with the directions where the data varies the most, called **Principal Components**.

### Key Points
- PCA transforms original correlated features into new, uncorrelated variables called **Principal Components (PCs)**.
- PCA finds the main **orthogonal** (perpendicular/independent) directions of variation in the data, then rotates the coordinate system to align with them.
- You can drop the later PCs (which explain little variance) without losing much information — this IS the dimensionality reduction.
- **The Mechanics of PCA (3 steps):**
  1. **Standardize the Data (Center)**: Subtract the mean from each variable so data is centered around zero. This ensures PCA captures the true shape/spread of data, not its location.
  2. **Find Principal Components**:
     a. Compute the **covariance matrix** (shows how features vary together)
     b. Calculate **eigenvalues and eigenvectors** of that covariance matrix — **eigenvectors** = direction of each principal component; **eigenvalues** = how much variance lies along that direction
     c. Sort eigenvectors by eigenvalue, descending — the top ones capture the most variance
  3. **Project Data onto top Components**: Transform data into the new coordinate system (keeping only the top components).
- **Covariance formula**: Cov(x,y) = (1/(n−1)) Σ (xᵢ − x̄)(yᵢ − ȳ)
- **Loadings / Projection**: PCᵢ = X · vᵢ (dot product of centered data X with eigenvector vᵢ). Expanded: PC₁ = a₁x₁ + a₂x₂ + ... + a_d x_d, where a₁...a_d are **loadings** (weights) showing how much each original feature contributes to that PC.
- **Scree plot**: shows how much variance each PC explains — first PC captures the most, second captures the most of what's left (orthogonal to the first), etc.
- **Limitation**: PCA captures **linear** correlations well but **ignores non-linear/higher-order interactions**. If data lies on a curved manifold, PCA flattens it, destroying the true geometric structure.

### Definitions
- **Principal Component (PC)**: A new, uncorrelated variable formed as a linear combination of the original features, capturing a direction of maximum variance.
- **Eigenvector**: A direction (axis) found in the covariance matrix; defines a principal component's orientation.
- **Eigenvalue**: A number showing how much variance is captured along its corresponding eigenvector's direction.
- **Covariance Matrix**: A matrix showing how every pair of features varies together.
- **Loadings**: Weights showing how strongly each original feature contributes to a given principal component.
- **Scree Plot**: A chart of variance explained by each successive principal component, used to decide how many PCs to keep.

### Why It Matters
PCA lets you simplify complex, high-dimensional data (e.g., 30,000 genes per cell) down to a handful of components that still capture most of the meaningful variation — making visualization, storage, and further modelling far more tractable, while staying interpretable through loadings.

### Easy Example
Imagine a cloud of points shaped like a cigar in 3D space. Most of the variation is along the cigar's long axis (PC1), a little across its width (PC2), and almost none through its thin depth (PC3). Dropping PC3 barely changes anything — you've reduced 3D to 2D with almost no information loss.

### Common Mistakes
- Forgetting to standardize/center data before PCA — skipping this step means PCA captures the data's location instead of its true spread.
- Thinking PCA works well on strongly non-linear/curved data — it doesn't (that's what t-SNE/UMAP are for).
- Confusing eigenvalues (amount of variance) with eigenvectors (direction of variance) — a very common exam trap.
- Assuming the top 2 PCs always explain "enough" variance — always check the scree plot.

### MCQ Preparation
1. **What do eigenvectors of the covariance matrix represent in PCA?**
   A) The amount of variance along a direction B) The direction/axis of a principal component C) The original feature values D) The loss function
   **Answer: B.**
2. **What do eigenvalues represent in PCA?**
   A) The direction of principal components B) How much variance is captured along a given eigenvector C) The number of features D) The correlation between two random features
   **Answer: B.**
3. **Why must data be centered (mean-subtracted) before PCA?**
   A) To increase dimensionality B) So PCA captures the true shape/spread of data, not its location C) To make computations impossible D) Centering is not necessary for PCA
   **Answer: B.**
4. **A major limitation of PCA is that it:**
   A) Cannot be used for visualization B) Ignores non-linear/higher-order interactions in the data C) Cannot reduce dimensionality D) Requires labeled data
   **Answer: B.**
5. **The scree plot is used to:**
   A) Show class labels B) Show how much variance each principal component explains, to choose how many to keep C) Show missing data patterns D) Encode categorical variables
   **Answer: B.**

### Short Questions
1. **List the three main steps ("mechanics") of PCA.** (1) Standardize/center the data, (2) find principal components via covariance matrix + eigenvalues/eigenvectors, (3) project data onto the top components.
2. **What's the difference between an eigenvector and an eigenvalue in PCA?** An eigenvector gives the direction of a principal component; the eigenvalue tells you how much variance lies along that direction.
3. **What are "loadings" in PCA, and why are they useful?** Loadings are the weights showing how much each original feature contributes to a principal component; they help interpret what drives the variance captured by that PC.
4. **Why can PCA "fail" on highly non-linear data?** Because PCA only finds linear directions of variance; if data lies on a curved manifold, PCA projects it onto a flat plane, destroying the true structure.
5. **How does PCA achieve dimensionality reduction in practice?** By keeping only the top PCs (those with the highest eigenvalues/variance) and dropping the rest, since the dropped PCs contribute little information.

### Memory Tricks
**"Vector = Direction, Value = Volume (of variance)"** — eigenVECTOR gives direction, eigenVALUE gives the amount.
Remember the 3-step mechanic as **"Center, Compute, Cast (project)."**
PCA is like **rotating a camera** around a 3D object cloud until you find the angle that shows the most spread — then you can drop the "boring" angles.

---

## 11. t-SNE (t-Distributed Stochastic Neighbour Embedding)

### Core Idea
t-SNE is a **non-linear** dimensionality reduction technique mainly used for **visualizing** high-dimensional data in 2D or 3D. Unlike PCA, it focuses on preserving **local structure** — keeping similar points close together — rather than global variance.

### Key Points
- t-SNE separates data that can't be separated by a straight line (non-linear).
- It preserves **local structure**: points that were neighbors in high-dimensional space stay close together after reduction (clusters are preserved).
- **How it works, step by step:**
  1. In the **original high-dimensional space**, for each point xᵢ, compute the probability that another point xⱼ would be picked as its neighbor, based on a **Gaussian distribution** centered at xᵢ:
     P(j|i) = exp(−‖xᵢ−xⱼ‖²/2σᵢ²) / Σₖ≠ᵢ exp(−‖xᵢ−xₖ‖²/2σᵢ²)
     - ‖xᵢ−xⱼ‖² = squared Euclidean distance between points
     - σᵢ = a scaling factor controlled by **perplexity** (how many neighbors to consider around each point)
  2. Since this matrix is **asymmetric**, t-SNE symmetrizes it into a joint probability: Pᵢⱼ = (P(j|i) + P(i|j)) / 2N
  3. Points are randomly scattered into a **lower-dimensional** space, and similarities there are modelled with a **t-Student distribution** (Q) instead of Gaussian (this heavier-tailed distribution helps avoid "crowding" points together).
  4. **Gradient Descent** is used to minimize the **KL (Kullback-Leibler) Divergence** between the original distribution P and the low-dimensional distribution Q:
     KL(P‖Q) = Σᵢ≠ⱼ Pᵢⱼ log(Pᵢⱼ/Qᵢⱼ)
  5. Points are iteratively "attracted" or "repelled" until distances converge to a stable low-dimensional map.
- **Perplexity** (key hyperparameter): controls how many neighbors each point considers, balancing local vs. global structure. Too low = fragmented tiny clusters; too high = loses local detail; good values are typically around 30 for many datasets (context-dependent).

### Definitions
- **t-SNE**: A stochastic, non-linear dimensionality reduction algorithm that preserves local neighborhood structure for visualization.
- **Perplexity**: A t-SNE hyperparameter controlling the effective number of neighbors considered around each point.
- **KL Divergence**: A measure of how different one probability distribution is from another; t-SNE minimizes this between high-D and low-D similarity distributions.
- **Stochastic**: Involving randomness — re-running t-SNE can give slightly different layouts.

### Why It Matters
t-SNE is the go-to tool for visually exploring complex, high-dimensional datasets (e.g., single-cell gene expression, image embeddings) because it reveals clusters and groupings that linear methods like PCA might blur together.

### Easy Example
Imagine sorting a huge pile of mixed photos by content (cats together, dogs together, cars together) just by "who looks similar to whom nearby" — without worrying about how far the cat pile is from the car pile overall. That's t-SNE: great at local grouping, not meant for judging distances between distant clusters.

### Common Mistakes
- Interpreting distances **between** clusters in a t-SNE plot as meaningful — they are NOT reliable (only local/within-cluster distances matter).
- Assuming one run of t-SNE is definitive — it's stochastic, so results can vary between runs.
- Ignoring perplexity — using a wildly wrong perplexity distorts cluster shapes/sizes.
- Confusing t-SNE (visualization-focused, local, non-linear, slow) with PCA (general dimensionality reduction, global, linear, fast).

### MCQ Preparation
1. **t-SNE primarily preserves:**
   A) Global variance B) Local neighborhood structure C) Feature correlations only D) Class labels
   **Answer: B.**
2. **What distribution is used to model similarities in the ORIGINAL high-dimensional space in t-SNE?**
   A) t-Student distribution B) Gaussian distribution C) Uniform distribution D) Poisson distribution
   **Answer: B.**
3. **What distribution is used to model similarities in the LOW-dimensional space?**
   A) Gaussian B) t-Student distribution C) Binomial D) Exponential
   **Answer: B.**
4. **What does t-SNE minimize using gradient descent?**
   A) Mean Squared Error B) KL (Kullback-Leibler) Divergence C) Accuracy D) Covariance
   **Answer: B.**
5. **Perplexity in t-SNE controls:**
   A) The number of PCs kept B) How many neighbors each point effectively considers C) The learning rate of PCA D) The number of classes
   **Answer: B.**

### Short Questions
1. **What kind of dimensionality reduction technique is t-SNE (linear or non-linear), and what is it mainly used for?** Non-linear; mainly used for visualizing high-dimensional data in 2D/3D.
2. **Why does t-SNE use a Gaussian distribution in high-D space but a t-Student distribution in low-D space?** The t-Student distribution has heavier tails, which helps prevent points from "crowding" together in the lower-dimensional map, giving clearer separation.
3. **What does t-SNE optimize, and how?** It minimizes the KL divergence between the high-dimensional (P) and low-dimensional (Q) similarity distributions, using gradient descent.
4. **What is perplexity, and why does it matter?** A hyperparameter controlling the effective number of neighbors per point; it balances local vs. global structure and strongly affects the resulting visualization.
5. **Why shouldn't you trust distances between separate clusters in a t-SNE plot?** Because t-SNE only preserves local neighborhood relationships — global/inter-cluster distances are not guaranteed to be meaningful.

### Memory Tricks
**"Gaussian up top, t-Student down low"** — remembers which distribution is used in which space.
**"t-SNE = neighborhood gossip"** — it only cares about who's near whom, not the big picture of the whole town (cluster-to-cluster distances).
Perplexity = "how many friends does each point hang out with?"

---

## 12. PCA vs t-SNE vs UMAP (Comparison)

### Core Idea
PCA and t-SNE solve related but different problems: PCA is a fast, deterministic, linear technique good for general dimensionality reduction; t-SNE is a slower, stochastic, non-linear technique focused purely on visualization. UMAP is a newer alternative to t-SNE with similar goals.

### Comparison Table

| Aspect | PCA | t-SNE |
|---|---|---|
| **Purpose** | Dimensionality reduction & feature transformation | Data visualization in low dimensions (typically 2D) |
| **Type of algorithm** | Deterministic | Stochastic |
| **Approach** | Linear transformation (distances make sense) | Probabilistic, local similarity embedding |
| **Speed** | Fast | Slow |
| **Noisy data** | Handles noisy data well | Struggles with noisy data |
| **Structure preserved** | Global | Local only |

- **Combining both**: It's common to apply PCA first (to reduce noise/dimensions quickly), then t-SNE on the PCA output (to visualize) — getting the best of both worlds.
- **UMAP (Uniform Manifold Approximation and Projection)**: Another powerful non-linear visualization tool, often considered after mastering PCA and t-SNE; used similarly to t-SNE but often faster and better at preserving some global structure.

### Definitions
- **Deterministic algorithm**: Always produces the same output given the same input (like PCA).
- **Stochastic algorithm**: Involves randomness, so outputs can vary between runs (like t-SNE).
- **UMAP**: A non-linear dimensionality reduction/visualization technique, an alternative/complement to t-SNE.

### Why It Matters
Choosing the wrong tool wastes time and can mislead conclusions: using PCA on strongly non-linear, cluster-heavy data (like cell types) will blur groups together; using t-SNE just to "reduce dimensions for modelling" is inappropriate since its distances aren't globally meaningful.

### Easy Example
PCA is like flattening a map of a mountain range by keeping its overall shape (heights become an average slope) — good for a bird's-eye general view. t-SNE is like drawing separate zoomed-in maps of each village, prioritizing correctly grouping houses within a village, even if the maps of different villages aren't placed at realistic distances from each other.

### Common Mistakes
- Assuming t-SNE plot distances between clusters have real-world meaning (they generally don't).
- Using t-SNE for general-purpose dimensionality reduction before modelling — it is designed for visualization, not feature engineering.
- Forgetting PCA is deterministic and t-SNE is stochastic (an easy MCQ fact to miss).

### MCQ Preparation
1. **Which algorithm is deterministic — always giving the same output for the same input?**
   A) t-SNE B) PCA C) UMAP D) None of them
   **Answer: B.**
2. **Which technique preserves GLOBAL structure of the data?**
   A) t-SNE B) PCA C) Neither D) Both equally
   **Answer: B.**
3. **Which technique is generally SLOWER to compute?**
   A) PCA B) t-SNE C) Both are equally fast D) Neither uses computation
   **Answer: B.**
4. **A common workflow that gets "the best of both worlds" is:**
   A) Running t-SNE twice B) Applying PCA first, then t-SNE on the PCA output C) Only ever using UMAP D) Avoiding PCA entirely
   **Answer: B.**
5. **Which technique struggles more with noisy data?**
   A) PCA B) t-SNE C) Neither is affected by noise D) Both are equally robust
   **Answer: B.**

### Short Questions
1. **List two differences between PCA and t-SNE.** (Any two) PCA is linear & deterministic & fast & preserves global structure; t-SNE is non-linear & stochastic & slow & preserves local structure only.
2. **Why might you combine PCA and t-SNE in one workflow?** PCA quickly reduces noise/dimensionality first, then t-SNE is applied on the reduced data for a cleaner, faster visualization — combining PCA's speed/noise-robustness with t-SNE's local structure clarity.
3. **What does it mean that PCA is "deterministic" and t-SNE is "stochastic"?** PCA always gives the same result for the same input; t-SNE involves randomness, so results can differ slightly between runs.
4. **When would t-SNE be a poor choice compared to PCA?** When you need fast, reliable, general-purpose dimensionality reduction for modelling (not just 2D visualization), or when data is noisy.
5. **What is UMAP generally used for?** As an alternative (often faster) non-linear technique for visualizing/understanding high-dimensional data, similar in spirit to t-SNE.

### Memory Tricks
**"P" for PCA = "Predictable" (deterministic) and "t-SNE" = "stochastic, stumbles with noise, slow."**
Global vs Local: **PCA sees the whole forest; t-SNE sees the trees closest to you.**

---

## 13. Validation Metrics: Confusion Matrix, Precision, Recall, Accuracy, Specificity, F1, ROC-AUC

### Core Idea
A single number like "accuracy" often lies to you, especially with imbalanced classes. The **confusion matrix** breaks predictions into 4 buckets, and from it we derive several metrics (precision, recall, accuracy, specificity, F1, ROC-AUC) — each answering a different business question.

### Key Points — The Confusion Matrix (binary classification)

|  | **Predicted Positive** | **Predicted Negative** |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

*(Example used in slides: classifying pathological images as malignant (positive) vs. benign (negative).)*

- **Precision** = TP / (TP + FP) → "Of everything I predicted positive, how much was actually positive?"
- **Recall** (Sensitivity, True Positive Rate) = TP / (TP + FN) → "Of all actual positives, how many did I correctly find?"
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) → "Of all predictions, how many were correct?"
- **Specificity** (True Negative Rate) = TN / (TN + FP) → "Of all actual negatives, how many did I correctly find?"
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) → harmonic mean of precision & recall; preferred when the positive class is rare AND correctly identifying it matters.
  - F1 = 1 → perfect precision and recall
  - F1 = 0 → worst possible (no true positives found)
  - 0 < F1 < 1 → balance between precision and recall (closer to 1 is better)
- **ROC Curve** (Receiver Operating Characteristic): plots True Positive Rate (Recall/Sensitivity, y-axis) vs. False Positive Rate (x-axis) across all possible classification thresholds. A perfect classifier hugs the top-left corner; the diagonal line represents random guessing. We plot ROC for each model/threshold and pick the best-performing one and most suitable threshold for the application.
- **Danger of a single metric**: Always predicting positive → 100% recall, but poor precision. Always predicting negative → 100% specificity, but 0% recall. If 98% of data is negative, always predicting "negative" → 98% accuracy despite being useless.

### Definitions
- **True Positive (TP)**: Correctly predicted positive case.
- **False Positive (FP)**: Incorrectly predicted as positive (actually negative) — a "false alarm."
- **False Negative (FN)**: Incorrectly predicted as negative (actually positive) — a "missed case."
- **True Negative (TN)**: Correctly predicted negative case.
- **Precision**: Fraction of predicted positives that are actually positive.
- **Recall/Sensitivity**: Fraction of actual positives correctly identified.
- **Specificity**: Fraction of actual negatives correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC Curve**: A graph of True Positive Rate vs. False Positive Rate across classification thresholds, used to compare classifiers and pick a threshold.
- **AUC (Area Under Curve)**: The area under the ROC curve; higher = better overall discrimination ability, regardless of threshold.

### Why It Matters
Different real-world problems care about different errors. In cancer screening, missing a true case (FN) is far worse than a false alarm (FP) — so recall matters more. In spam filtering, wrongly flagging a real email (FP) as spam might be worse than missing some spam (FN) — so precision may matter more. Metrics let you match your evaluation to the actual cost of different mistakes.

### Easy Example
Imagine airport security screening (positive = "has a weapon"):
- **Recall** = did we catch ALL the actual threats? (missing one is very costly → recall matters a lot)
- **Precision** = of everyone we flagged, how many actually had a weapon? (too many false alarms annoy travelers)
- **F1** balances both; **ROC/AUC** lets you tune your metal-detector sensitivity (threshold) based on which mistake you can tolerate more.

### Common Mistakes
- Confusing precision and recall (exam favorite: memorize which denominator uses FP vs. FN).
- Believing accuracy is always a good metric — it's misleading with imbalanced classes.
- Forgetting: predicting all positive → Recall = 100% (not precision); predicting all negative → Specificity = 100% (not accuracy, unless negatives dominate).
- Thinking F1 is just an average of precision and recall — it's the **harmonic** mean, which penalizes large gaps between the two more than a simple average would.

### MCQ Preparation
1. **A medical test dataset has 5% positive, 95% negative cases. A classifier predicts every patient as negative. Which is TRUE?**
   A) Recall = 100% B) Specificity = 0% C) Specificity = 100% D) Precision = 100%
   **Answer: C.** (All actual negatives are correctly identified as negative since it predicts negative for everyone; TN/(TN+FP) = 100%. Recall would be 0% since no actual positives are found.)
2. **A spam filter predicts every email as spam (positive). Which metric is guaranteed to be 100%?**
   A) Accuracy B) Recall C) Specificity D) Precision
   **Answer: B.** (All actual positives — spam — get caught since everything is predicted spam, so TP/(TP+FN) = 100%. Precision and accuracy depend on how much spam vs. real mail exists.)
3. **Precision is calculated as:**
   A) TP/(TP+FN) B) TP/(TP+FP) C) TN/(TN+FP) D) (TP+TN)/(all)
   **Answer: B.**
4. **F1 Score is preferred over accuracy when:**
   A) Classes are perfectly balanced B) The positive class is rare and correctly predicting it matters C) You only care about true negatives D) You have no false positives
   **Answer: B.**
5. **The ROC curve plots:**
   A) Precision vs. Recall B) True Positive Rate vs. False Positive Rate across thresholds C) Accuracy vs. F1 D) TP vs. TN only
   **Answer: B.**

### Short Questions
1. **Write the formulas for Precision and Recall, and explain the difference in plain English.** Precision = TP/(TP+FP) — "of what I called positive, how much was right?" Recall = TP/(TP+FN) — "of all the actual positives, how many did I catch?"
2. **Why is accuracy a poor metric on imbalanced data? Give a numeric illustration.** If 98% of data is negative, a model that always predicts "negative" scores 98% accuracy while catching 0% of the rare positive class — accuracy hides this failure.
3. **What does the F1 score balance, and when should you use it?** It balances precision and recall via their harmonic mean; use it when the positive class is rare and you need a single number reflecting both false alarms and missed cases.
4. **What does the ROC curve let you do that a single metric can't?** It lets you see performance across ALL classification thresholds simultaneously, so you can choose the best threshold/model trade-off between true positive rate and false positive rate for your specific application.
5. **In cancer screening, why might recall matter more than precision?** Because missing an actual cancer case (false negative) can be life-threatening, while a false alarm (false positive) just leads to extra, safer follow-up testing.

### Memory Tricks
**"Precision = Purity of Positives predicted; Recall = Retrieval of Real positives."**
**PRESS**: **P**recision → predicted-positive denominator; **R**ecall → actual-positive (real) denominator — remember Recall's "R" pairs with "Real."
For confusion matrix cells: **TP/FN are in the "actual positive" row; FP/TN are in the "actual negative" row** — draw the 2×2 grid from memory before answering these questions.

---

## 14. Bayes' Theorem in Diagnostic Testing (Worked Example)

### Core Idea
Even a test that's 99% accurate can give surprisingly LOW real-world confidence when the disease itself is rare. **Bayes' Theorem** lets you correctly compute "given a positive test, what's the real probability of having the disease?" — this is very different from just "the test's accuracy."

### Key Points — The Slide's Worked Example
- **Setup**: Disease prevalence P(D) = 0.001 (1 in 1000). P(Healthy) = P(H) = 0.999.
- Test sensitivity: P(Positive | Disease) = P(P|D) = 0.99 (99% of sick people test positive)
- False positive rate: P(Positive | Healthy) = P(P|H) = 0.02 (2% of healthy people test positive anyway)
- **Question**: If someone tests positive, what's P(Disease | Positive)?
- **Bayes' Theorem**:
  P(D|P) = [P(P|D) × P(D)] / P(P) = [P(P|D) × P(D)] / [P(P|D)×P(D) + P(P|H)×P(H)]
- **Plugging in numbers**:
  P(D|P) = (0.99 × 0.001) / (0.99×0.001 + 0.02×0.999) = 0.00099 / (0.00099 + 0.01998) ≈ **4.72%**
- **Key takeaway**: Even with a "99% accurate" test, a positive result only means ~4.72% real chance of having the disease — because the disease is so rare that false positives from the huge healthy population outnumber true positives from the tiny sick population.

### Definitions
- **Bayes' Theorem**: A rule for updating the probability of a hypothesis (e.g., having a disease) given new evidence (e.g., a positive test), combining prior probability with the test's reliability.
- **Prior probability, P(D)**: The probability of having the disease before seeing any test result (based on prevalence).
- **Posterior probability, P(D|P)**: The updated probability of having the disease AFTER seeing a positive test result.
- **Sensitivity** (same as Recall): P(Positive|Disease) — probability the test correctly flags a sick person.
- **False Positive Rate**: P(Positive|Healthy) — probability the test incorrectly flags a healthy person.

### Why It Matters
This example is a classic "the numbers lie unless you do the full math" trap. It explains why doctors don't panic (or celebrate) based on a single positive/negative test alone for rare conditions, and why follow-up/confirmatory testing exists.

### Easy Example
Imagine a haystack with 1 needle per 1000 pieces of straw. Even a great metal detector (99% likely to find the needle when swept over it) will also occasionally "beep" on ~2% of the straw. Since there's SO much more straw than needles, most beeps are actually straw, not needle.

### Common Mistakes
- Confusing the test's accuracy (99%) with the actual post-test probability of disease (~4.72%) — these are NOT the same thing (huge exam trap).
- Forgetting to include BOTH the true positive contribution AND the false positive contribution in the denominator of Bayes' formula.
- Not accounting for how rare the condition is (the prior) — a rare disease dramatically lowers the post-test probability even with a "good" test.

### MCQ Preparation
1. **In the slide's example, what is the chance someone who tested positive actually has the disease?**
   A) 99% B) 2% C) ~4.72% D) 100%
   **Answer: C.**
2. **Why is the post-test probability so much lower than the test's stated 99% sensitivity?**
   A) The math is wrong B) The disease is rare, so false positives from the large healthy population outnumber true positives C) The test has no false positives D) Because sensitivity and post-test probability are always equal
   **Answer: B.**
3. **What does P(D) represent in this Bayes' example?**
   A) The test's sensitivity B) The prior probability (prevalence) of the disease C) The false positive rate D) The posterior probability
   **Answer: B.**
4. **The denominator of Bayes' theorem in this example represents:**
   A) Only true positives B) The total probability of testing positive (from both diseased and healthy people) C) Only false negatives D) The prior probability alone
   **Answer: B.**
5. **This example illustrates that:**
   A) Test accuracy alone tells you everything you need to know B) Even accurate tests can have low real-world positive predictive value for rare conditions C) Rare diseases don't need any testing D) False positive rates don't matter
   **Answer: B.**

### Short Questions
1. **Write Bayes' Theorem as used to compute P(Disease | Positive test).** P(D|P) = [P(P|D)×P(D)] / [P(P|D)×P(D) + P(P|H)×P(H)].
2. **Using the slide's numbers, calculate (or state) the post-test probability of disease.** ≈4.72% — (0.99×0.001)/(0.99×0.001+0.02×0.999).
3. **Why does a rare disease prevalence dramatically lower the post-test probability even with a 99%-sensitive test?** Because the healthy population is so much larger, even a small false-positive rate (2%) produces more false positives in absolute numbers than the true positives from the tiny diseased population.
4. **What two pieces of information does Bayes' theorem combine?** The prior probability (disease prevalence) and the test's reliability (sensitivity and false positive rate).
5. **What real-world lesson does this teach about interpreting positive medical test results for rare diseases?** A single positive result doesn't mean high certainty of disease — confirmatory/follow-up testing is often needed, especially for rare conditions.

### Memory Tricks
**"Rare disease + imperfect test = surprisingly unreliable single positive result."**
Remember Bayes' denominator as **"True Positives PLUS False Positives"** (both ways you could have tested positive).

---

## 15. Model Training: Cost Function, Overfitting, Underfitting

### Core Idea
Training a model means iteratively adjusting its internal parameters to minimize a **cost function** (the "penalty" for wrong predictions). Depending on how this process goes, the model can end up **overfitting** (too complex, memorizes noise) or **underfitting** (too simple, misses real patterns).

### Key Points
- **Training loop**: Features ("covariates") + Training labels → fed into the model (prediction function) → compute the **cost function** (comparing prediction to ground truth) → use the resulting error to update model parameters → repeat.
- **Cost function** (also called objective function in supervised learning): generally a penalty for misclassification/wrong prediction, used to assess a model's predictive performance during training.
- **Overfitting** = **high variance, little bias**: the model fits training data extremely well (even its noise/quirks) but generalizes poorly to new data.
- **Underfitting** = **high bias, low variance**: the model is too simple, and performs poorly even on the training data (fails to capture real patterns).
- The goal of training is a model that generalizes well — neither underfitting nor overfitting.

### Definitions
- **Cost/Objective Function**: A measure of how wrong a model's predictions are, which the learning algorithm tries to minimize during training.
- **Overfitting**: A modelling error where the model captures noise/random details of the training data, causing poor performance on new data (high variance).
- **Underfitting**: A modelling error where the model is too simple to capture the underlying pattern, performing poorly even on training data (high bias).
- **Bias**: Error from overly simplistic assumptions in the model (associated with underfitting).
- **Variance**: Error from the model's sensitivity to small fluctuations in training data (associated with overfitting).

### Why It Matters
Understanding this training loop lets you diagnose WHY a model is failing: if training performance itself is poor → underfitting (try a more complex model or better features); if training performance is great but validation/test performance is poor → overfitting (try regularization, more data, or a simpler model).

### Easy Example
Underfitting = trying to fit a straight line through data that's clearly curved (misses the pattern everywhere). Overfitting = drawing a wiggly line that passes through every single training point exactly, including random noise — perfect on training data but nonsensical on new points.

### Common Mistakes
- Confusing overfitting (high variance) with underfitting (high bias) — they are opposites.
- Believing more model complexity always helps — beyond a point it causes overfitting.
- Forgetting that the cost function is only a "proxy" for real-world objectives, not identical to them (see Section 13's point on validation metrics).

### MCQ Preparation
1. **Overfitting is characterized by:**
   A) High bias, low variance B) High variance, little bias C) Low bias, low variance D) Neither bias nor variance
   **Answer: B.**
2. **Underfitting is characterized by:**
   A) High bias, low variance B) High variance, little bias C) Perfect generalization D) Zero training error always
   **Answer: A.**
3. **The cost function in supervised learning is generally used as:**
   A) A measure of computational speed B) A penalty for misclassification, used to guide parameter updates C) A method for encoding categorical features D) A synonym for the confusion matrix
   **Answer: B.**
4. **In the training loop, what is used to update model parameters?**
   A) The raw features alone B) The error/cost computed by comparing predictions to ground truth C) The validation set only D) Random noise
   **Answer: B.**
5. **A model that performs poorly even on training data is likely:**
   A) Overfitting B) Underfitting C) Perfectly generalized D) Using too much data
   **Answer: B.**

### Short Questions
1. **Describe the basic training loop in your own words.** Features and labels are fed to the model, which makes predictions; the cost function computes the error; that error is used to update the model's parameters; repeat until performance is acceptable.
2. **What's the difference between bias and variance, in relation to underfitting/overfitting?** Bias reflects error from oversimplified assumptions (underfitting); variance reflects sensitivity to training data specifics/noise (overfitting).
3. **How would you recognize overfitting vs. underfitting from training/validation performance?** Underfitting: poor performance even on training data. Overfitting: great training performance but much worse validation/test performance.
4. **Why is the cost function called only a "proxy" for real-world objectives?** Because minimizing prediction error mathematically doesn't always align perfectly with the true business/real-world goal (e.g., cost of different types of mistakes may not be captured).
5. **Give an example of an underfit and an overfit model on the same dataset.** Underfit: a straight line forced through clearly curved data. Overfit: a highly wiggly curve that passes through every single training point, including noise.

### Memory Tricks
**"Under = simple & wrong everywhere. Over = complex & wrong on new things."**
**Bias = "Blind to patterns" (too simple). Variance = "Very sensitive to noise" (too complex).**

---

## 16. Hyperparameters vs Model Parameters

### Core Idea
**Model parameters** are what the model LEARNS automatically during training (e.g., weights). **Hyperparameters** are settings YOU choose before training that control how the learning happens (e.g., learning rate).

### Key Points — Comparison Table

| Feature | Model Parameters | Hyperparameters |
|---|---|---|
| **Set by** | Learned during training | Manually set before training |
| **Purpose** | Define the model's behavior | Control the learning process |
| **Examples** | Weights, biases | Learning rate, epochs, tree depth |
| **Tuned using** | Optimization algorithms | Search or validation methods |

- Hyperparameter tuning happens using the **validation set** (from Section 4), often via grid search, random search, or more advanced methods.

### Definitions
- **Model Parameter**: A value internal to the model, learned from training data (e.g., a neural network's weights).
- **Hyperparameter**: A configuration value set before training that governs the training process itself, not learned from data directly (e.g., number of trees in a random forest, learning rate).

### Why It Matters
Confusing the two leads to modelling mistakes — e.g., trying to "train" a hyperparameter like you would a weight, or forgetting that hyperparameters must be tuned using validation data, not the training set (which would tune the model too closely to seen data).

### Easy Example
Baking a cake: the **recipe settings** you choose beforehand — oven temperature, baking time — are hyperparameters. How the cake batter actually rises and browns in response is like the model parameters "learned" during the bake.

### Common Mistakes
- Thinking hyperparameters are learned automatically during training — they are NOT; they're set manually or searched for beforehand/alongside training using validation performance.
- Confusing "tree depth" (a hyperparameter you set) with the actual learned split rules (which are more like parameters learned by the algorithm).
- Tuning hyperparameters directly on the test set (data leakage — should use validation).

### MCQ Preparation
1. **Which of these is a model PARAMETER (not a hyperparameter)?**
   A) Learning rate B) Number of epochs C) Neural network weights D) Tree depth
   **Answer: C.**
2. **Which of these is a HYPERPARAMETER?**
   A) Learned weights B) Learning rate C) Biases D) None of the above
   **Answer: B.**
3. **Hyperparameters are typically tuned using:**
   A) Training set only B) Search or validation methods C) Test set only D) They cannot be tuned
   **Answer: B.**
4. **Model parameters are:**
   A) Manually set before training B) Learned during training C) Never adjusted D) The same as hyperparameters
   **Answer: B.**
5. **An example of a hyperparameter in decision trees is:**
   A) The specific split values learned B) Maximum tree depth (set before training) C) Leaf node predictions D) None of the above
   **Answer: B.**

### Short Questions
1. **Define model parameter and hyperparameter, with one example each.** Model parameter: learned during training (e.g., neural network weights). Hyperparameter: manually set beforehand (e.g., learning rate).
2. **Why should hyperparameters be tuned on the validation set, not the test set?** To keep the test set truly unseen, so it can give an unbiased final performance estimate.
3. **Give three examples of hyperparameters.** Learning rate, number of training epochs, tree depth (or number of trees, batch size, etc.).
4. **How are model parameters "tuned" compared to hyperparameters?** Model parameters are optimized automatically via optimization algorithms (e.g., gradient descent) during training; hyperparameters are tuned via search or validation-based methods, usually before/alongside training.
5. **Why is it problematic to think of hyperparameters as something the model "learns" like weights?** Because hyperparameters control HOW learning happens (the process itself) — they must be chosen/searched before or outside the normal training loop, not learned through gradient updates like weights.

### Memory Tricks
**"Hyper = Human-set. Parameter = Pattern-learned."**
Recipe analogy: **Hyperparameters = oven settings you dial in; Parameters = how the cake actually turns out (learned through baking).**

---

## 17. Choosing the Most Suitable Model

### Core Idea
The model with the highest accuracy is NOT automatically the "best" or most suitable choice — real-world deployment requires balancing performance with interpretability, simplicity, speed, and stability.

### Key Points
- **What to consider when choosing the most suitable model:**
  - **Generalizability**: avoids overfitting, performs well on unseen data
  - **Simplicity & Maintainability**: easier to debug, update, deploy
  - **Speed & Efficiency**: meets latency and resource constraints
  - **Interpretability**: critical in domains like healthcare, finance, law (regulators/stakeholders may require explainable decisions)
  - **Stability & Reproducibility**: performs consistently across data variations
- **Summary**: The most suitable model balances performance with interpretability, scalability, and operational constraints.
- **How to choose between Statistical Modelling, Machine Learning, or Deep Learning**: depends on characteristics of BOTH the data and the desired model properties:
  - **Data factors**: number of features, size of data, input/output format, specifics of the data
  - **Model factors**: fitness (accuracy), simplicity, interpretability, speed of modelling
- Remember: Data Mining is broader than just Machine Learning — it also includes visualization and other analytics approaches.
- Tools like the **scikit-learn algorithm cheat-sheet** exist to help practitioners choose an appropriate estimator/algorithm based on data type, size, and task.
- **Common misconception (from Prototyping slide)**: people think you always pick "the best model in terms of accuracy" — in practice, you usually start with the SIMPLEST model and build complexity only as needed.

### Definitions
- **Interpretability**: How easily humans can understand why a model made a particular prediction.
- **Prototyping**: A rapid, iterative process of testing and identifying robust modelling solutions (starting simple, adding complexity gradually).

### Why It Matters
In regulated or high-stakes fields (medicine, finance, law), a slightly less accurate but interpretable model may be preferred over a highly accurate "black box," because stakeholders need to trust and explain decisions. Also, unnecessarily complex models cost more to maintain and run slower in production.

### Easy Example
A hospital might prefer a simple, interpretable logistic regression (that doctors can explain to patients: "your risk went up because of X, Y, Z") over a marginally more accurate deep neural network that can't explain its reasoning — because trust and explainability matter enormously in healthcare decisions.

### Common Mistakes
- Believing "highest accuracy = best model" — ignoring interpretability, speed, and maintainability (a classic exam trap).
- Starting a project with the most complex model (e.g., deep learning) instead of the simplest reasonable baseline.
- Forgetting Data Mining includes visualization/analytics beyond pure ML modelling.

### MCQ Preparation
1. **The most suitable model for a project is defined as the one that:**
   A) Always has the highest accuracy B) Balances performance with interpretability, scalability, and operational constraints C) Is always the most complex available D) Requires the least data
   **Answer: B.**
2. **In regulated domains like healthcare or finance, which factor becomes especially critical when choosing a model?**
   A) Training speed only B) Interpretability C) Number of hyperparameters D) Dataset file format
   **Answer: B.**
3. **A common misconception about prototyping/modelling, per the slides, is that:**
   A) You should start with the simplest model B) You should always pick the model with best accuracy first, rather than starting simple C) Prototyping is unnecessary D) All models are equally interpretable
   **Answer: B.** (i.e., the misconception is thinking accuracy is everything — reality is you should start simple.)
4. **Which of these is NOT one of the "what to consider" factors for model suitability listed in the slides?**
   A) Generalizability B) Speed & Efficiency C) The programming language used D) Interpretability
   **Answer: C.**
5. **Data Mining, according to the slides, is:**
   A) Only about Machine Learning B) Also about visualization and other types of data analytics, not only ML C) Strictly limited to deep learning D) The same thing as data entry
   **Answer: B.**

### Short Questions
1. **List three factors (besides accuracy) to consider when choosing the most suitable model.** Any three of: generalizability, simplicity/maintainability, speed/efficiency, interpretability, stability/reproducibility.
2. **Why might a less accurate but more interpretable model be preferred in healthcare?** Because stakeholders (doctors, patients, regulators) need to understand and trust the reasoning behind decisions, not just get a number.
3. **What is the common misconception about model selection addressed in the "Prototyping" slide?** That you should always pick the model with the best accuracy; in reality, you should typically start with the simplest model and build complexity only as needed.
4. **What data-related factors influence whether to use statistical modelling, ML, or deep learning?** Number of features, size of the data, input/output format, and other data specifics.
5. **Why is Data Mining considered broader than just Machine Learning?** Because it also encompasses visualization and other types of data analytics, not solely predictive modelling.

### Memory Tricks
**"GSSIS"**: **G**eneralizability, **S**implicity, **S**peed, **I**nterpretability, **S**tability — the 5 pillars of model suitability.
**"Start simple, add complexity only if needed"** — the golden prototyping rule.

---

# 18. FINAL REVISION PACK

## 18.1 One-Page Cheat Sheet

**Data Mining Life Cycle**: Problem understanding → Explore/process data → Model & Analyze → Communicate results → Deploy → Test/Monitor (loop). Domain knowledge touches every stage.

**Data Representation**: No single "correct" format — text (TF-IDF/Word Embeddings), images (pixels), molecules (graphs), audio (spectrogram/time series/wavelets), graphs (Node2Vec).

**Sample Size**: Bigger ↑ statistical power, ↓ Type II error, ↓ overfitting risk, better generalization — BUT big data ≠ good data (doesn't fix bias/quality; can make trivial effects look significant).

**Train/Val/Test**: Train = learn parameters. Validation = tune hyperparameters/select model. Test = final unbiased score (use once!). Goal = generalization, avoiding under/overfitting.

**Feature Engineering**: Creation (e.g., speed=distance/time), Combining, Aggregation, Datetime extraction, Text features. **Encoding**: Label encoding risks false order; One-hot encoding needs one column dropped (avoid multicollinearity). **Binning**: groups numeric ranges into categories (some info loss).

**Missing Data**: Define pattern → process (drop/NA-flag/robust model/impute) → consider robust algorithms. Imputation: manual, global constant, statistical (mean/median), model-based. Watch for **survivorship bias**.

**Imbalanced Data**: Fixes = suited model/objective function, resampling (up/down-sample), augmentation, balanced metrics. Accuracy is misleading here.

**Feature Scaling**: Min-Max→[0,1] (NN, images). Z-score→mean 0/std 1 (general ML). Robust→median/IQR (outliers). **Always fit scaler on train only**, reapply saved params to test/production.

**Dimensionality Reduction vs Feature Selection**: Selection = drop features as-is. Reduction = transform into new, fewer features. Curse of dimensionality: too many features + not enough data → sparsity → worse performance. Linear: PCA, LDA, ICA. Non-linear: t-SNE, UMAP, Autoencoders.

**PCA**: Linear, deterministic, fast, preserves GLOBAL variance. Steps: Center data → Covariance matrix → Eigenvectors (direction)/Eigenvalues (variance amount) → sort descending → project onto top PCs. Loadings = feature contribution weights. Weak on non-linear data.

**t-SNE**: Non-linear, stochastic, slow, preserves LOCAL structure only (don't trust inter-cluster distances). Gaussian similarity in high-D, t-Student in low-D, minimizes KL Divergence via gradient descent. Perplexity = neighborhood size hyperparameter.

**Confusion Matrix & Metrics**:
- Precision = TP/(TP+FP) — "purity of positive predictions"
- Recall/Sensitivity = TP/(TP+FN) — "coverage of real positives"
- Accuracy = (TP+TN)/All — misleading if imbalanced
- Specificity = TN/(TN+FP) — "coverage of real negatives"
- F1 = 2·(Precision·Recall)/(Precision+Recall) — harmonic mean, good for rare positive class
- ROC curve = TPR vs FPR across thresholds; pick best model/threshold via AUC

**Bayes' Theorem (diagnostic testing)**: P(D|P) = [P(P|D)P(D)] / [P(P|D)P(D)+P(P|H)P(H)]. Rare disease + imperfect test → surprisingly low post-test probability (e.g., 4.72% in the slide's example) even with 99% sensitivity.

**Training**: Cost/objective function = penalty for wrong predictions, guides parameter updates. Overfitting = high variance/low bias (memorizes noise). Underfitting = high bias/low variance (too simple).

**Hyperparameters vs Parameters**: Parameters learned during training (weights). Hyperparameters set beforehand, tuned via validation (learning rate, tree depth, epochs).

**Choosing a Model**: Balance generalizability, simplicity, speed, interpretability, stability — NOT just accuracy. Start simple, add complexity as needed. Data Mining ⊃ visualization + analytics, not just ML.

---

## 18.2 Top 30 Exam Questions & Model Answers

1. **List the stages of the Data Mining project life cycle.** Problem understanding → Explore/process data → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring (cyclical).
2. **What three roles does domain knowledge play in a project?** Guides preprocessing, influences modelling, provides context for explaining results.
3. **Is there only one correct way to represent a given type of data? Explain with an example.** No — e.g., audio can be a spectrogram, a raw time series, or wavelets; the best choice depends on the analysis goal.
4. **Why can very large datasets still lead to wrong conclusions?** They reduce noise but don't fix bias/poor quality, and can make tiny, irrelevant effects appear statistically significant.
5. **Explain the purpose of splitting data into train, validation, and test sets.** Training fits parameters, validation tunes hyperparameters/selects models, test gives a final unbiased performance estimate — protecting against overfitting and biased evaluation.
6. **Why is label encoding risky for non-ordinal categorical variables?** It introduces a false sense of order/distance between categories that can mislead models using mathematical distance (e.g., k-NN, linear regression).
7. **What is the dummy variable trap, and how do we avoid it?** Keeping all one-hot encoded columns creates multicollinearity (redundancy); avoid it by dropping one column.
8. **What is binning, and what's a downside of it?** Grouping numeric ranges into categorical bins; downside is losing precision — near values in different bins are treated as very different, and far values in the same bin are treated as identical.
9. **Explain survivorship bias with the WWII airplane example.** Analysts saw bullet holes only on RETURNING planes; planes hit elsewhere didn't survive to be observed. Reinforcement should target the areas showing NO damage on survivors — those are likely fatal hit locations.
10. **List four strategies for handling missing values.** Drop records, add an NA indicator, use missing-data-robust models, or impute.
11. **List four strategies for handling imbalanced data.** Use a suited model/objective function, resample (up/downsample), data augmentation, use balanced evaluation metrics.
12. **Why must scalers be fit only on training data?** To prevent data leakage — using test data statistics during fitting would let test information improperly influence the model.
13. **Compare Min-Max scaling, Z-score standardization, and Robust scaling.** Min-Max → [0,1] range (NNs, images). Z-score → mean 0, std 1 (general ML). Robust → median/IQR based, resistant to outliers.
14. **Distinguish feature selection from dimensionality reduction.** Feature selection drops original features unchanged; dimensionality reduction mathematically transforms features into new, fewer ones.
15. **Explain the curse of dimensionality.** Beyond an optimal number of features, adding more dimensions without more data increases sparsity and degrades model performance.
16. **List the 3 main mechanical steps of PCA.** Standardize/center the data → compute covariance matrix and find eigenvectors/eigenvalues → project data onto the top principal components.
17. **What do eigenvalues and eigenvectors represent in PCA?** Eigenvectors = direction of a principal component; eigenvalues = amount of variance captured along that direction.
18. **Why does PCA struggle with non-linear data?** It only finds linear directions of variance; on curved/manifold data it flattens the structure, losing true geometric relationships.
19. **What does t-SNE preserve, and what does it NOT reliably preserve?** Preserves local neighborhood structure (similar points stay close); does NOT reliably preserve global/inter-cluster distances.
20. **What does t-SNE optimize, and using what method?** It minimizes the KL divergence between high-D (Gaussian-based) and low-D (t-Student-based) similarity distributions, using gradient descent.
21. **What is perplexity in t-SNE?** A hyperparameter controlling the effective number of neighbors considered per point, balancing local vs global structure.
22. **Give one key difference each for: purpose, algorithm type, speed, and structure preserved between PCA and t-SNE.** Purpose: PCA=dimensionality reduction/transformation, t-SNE=visualization. Type: PCA=deterministic, t-SNE=stochastic. Speed: PCA=fast, t-SNE=slow. Structure: PCA=global, t-SNE=local.
23. **Write and explain the Precision formula.** Precision = TP/(TP+FP) — of all predicted positives, the fraction that are truly positive.
24. **Write and explain the Recall formula.** Recall = TP/(TP+FN) — of all actual positives, the fraction correctly identified.
25. **Why is accuracy misleading for imbalanced datasets? Give a concrete number.** If 98% of cases are negative, always predicting "negative" yields 98% accuracy while catching 0% of the rare positive class.
26. **When is the F1 score preferred, and why?** When the positive class is rare and correctly identifying it matters; F1 (harmonic mean of precision & recall) penalizes large imbalances between the two more than a plain average would.
27. **Explain, using Bayes' theorem, why a "99% accurate" test for a rare disease can still yield a low post-test probability.** Because the disease is rare, the huge healthy population contributes many false positives (2% of a large group), which can outnumber the true positives from the tiny diseased group, dropping the post-test probability to just ~4.72% in the slide's example.
28. **Differentiate overfitting from underfitting in terms of bias and variance.** Overfitting = high variance, low bias (memorizes training noise, fails on new data). Underfitting = high bias, low variance (too simple, fails even on training data).
29. **Differentiate model parameters from hyperparameters, with examples.** Parameters are learned during training (e.g., weights); hyperparameters are set beforehand and tuned via validation (e.g., learning rate, tree depth).
30. **List the factors to weigh when choosing "the most suitable model" beyond accuracy.** Generalizability, simplicity/maintainability, speed/efficiency, interpretability, stability/reproducibility.

---

## 18.3 Top 30 MCQs & Answers

1. Which stage follows "Problem understanding" in the DM life cycle? → **Explore and process data**
2. Domain knowledge mainly helps with: → **Guiding preprocessing, modelling, and explaining results**
3. TF-IDF is used to represent: → **Textual data**
4. Node2Vec is an example of: → **Node embedding for graph data**
5. Which statement about sample size is TRUE? → **Large datasets can make insignificant effects appear important**
6. Small sample size mainly increases risk of: → **Overfitting (high variance)**
7. Which dataset is used to tune hyperparameters? → **Validation set**
8. Overfitting is best described as: → **Model fits training well but fails on unseen data**
9. Why is label encoding risky for non-ordinal categories? → **It introduces a false ordinal relationship**
10. Why drop one column after one-hot encoding? → **To avoid multicollinearity**
11. Binning converts: → **Numerical data into categorical buckets**
12. In the WWII example, where should reinforcement be added? → **Where returning planes show NO damage**
13. Survivorship bias occurs when: → **You analyze only surviving/selected cases, ignoring the rest**
14. Which strategy is NOT for handling imbalanced data? → **Always deleting the minority class**
15. Why is accuracy a poor metric for imbalanced data? → **A majority-only prediction can still score high accuracy**
16. Feature scaling parameters should be fit on: → **Training data only, then reapplied to test/production**
17. Which scaling method is resistant to outliers? → **Robust Scaling**
18. Key difference: feature selection vs dimensionality reduction → **Selection excludes features as-is; reduction transforms them**
19. Curse of dimensionality refers to: → **Performance degrading past an optimal feature count due to sparsity**
20. Which is a LINEAR dimensionality reduction technique? → **PCA**
21. Which is a NON-linear dimensionality reduction technique? → **t-SNE**
22. Eigenvectors in PCA represent: → **The direction of a principal component**
23. Eigenvalues in PCA represent: → **How much variance is captured along a direction**
24. Why must data be centered before PCA? → **So PCA captures the true shape/spread, not the data's location**
25. A major limitation of PCA: → **Ignores non-linear/higher-order interactions**
26. t-SNE primarily preserves: → **Local neighborhood structure**
27. What does t-SNE minimize via gradient descent? → **KL (Kullback-Leibler) Divergence**
28. Which technique is deterministic? → **PCA**
29. Precision is calculated as: → **TP/(TP+FP)**
30. F1 score is preferred when: → **The positive class is rare and correctly identifying it matters**

---

## 18.4 Concepts Professors Frequently Ask About (High-Yield Watchlist)

- The exact **order and names of the DM life cycle stages**
- **Survivorship bias** (WWII planes example) — a favorite conceptual trap
- **Label encoding vs one-hot encoding** and WHY (false order / multicollinearity)
- **Why accuracy is misleading** on imbalanced data (always tie this to a numeric scenario)
- **Precision vs Recall vs Specificity** formulas — professors love flipping "predict all positive/negative" scenarios
- **PCA mechanics**: eigenvalue vs eigenvector distinction (extremely common trick question)
- **PCA vs t-SNE comparison table** (linear/non-linear, deterministic/stochastic, global/local, fast/slow)
- **Bayes' theorem** applied to rare-disease testing — expect a full numeric calculation question
- **Overfitting vs underfitting** in terms of bias/variance
- **Hyperparameters vs parameters** — give clean examples every time
- **"Best model ≠ Most accurate model"** — interpretability/simplicity/speed tradeoffs
- **Feature scaling**: must fit on train only, reused at production (data leakage trap)
- **Curse of dimensionality**: increasing features helps only up to a point

---

## 18.5 "Last 10 Minutes Before Exam" Revision Summary

- **Life cycle**: Problem → Explore → Model → Communicate → Deploy → Monitor (loop). Domain knowledge everywhere.
- **Big data ≠ good data.** More data fights noise, not bias.
- **Train = learn. Validate = tune. Test = judge once.**
- **Label encoding → false order. One-hot → drop 1 column (multicollinearity).**
- **Survivorship bias = the missing data (non-survivors) is the real story.**
- **Imbalance fixes: model choice / resample / augment / balanced metrics. Never trust plain accuracy.**
- **Scale: fit on train, reuse on test/production. Min-Max=[0,1]; Z-score=mean0/std1; Robust=outlier-proof.**
- **Selection = drop features. Reduction = transform features. Curse of dimensionality = too many features, too little data → sparsity → worse performance.**
- **PCA: linear, deterministic, global, fast. Eigenvector=direction, Eigenvalue=amount of variance.**
- **t-SNE: non-linear, stochastic, local only, slow. Gaussian(high-D)→t-Student(low-D), minimizes KL divergence. Don't trust cluster-to-cluster distance.**
- **Precision = TP/(TP+FP). Recall = TP/(TP+FN). Specificity = TN/(TN+FP). F1 = harmonic mean, use when positive class is rare.**
- **Bayes: rare disease + imperfect test → surprisingly low post-test probability. Always include BOTH TP and FP terms in the denominator.**
- **Overfit = high variance (memorizes). Underfit = high bias (misses pattern).**
- **Hyperparameters = set before training (learning rate, depth). Parameters = learned (weights).**
- **Best model = balances accuracy + interpretability + speed + simplicity + stability — start simple, add complexity only if needed.**
