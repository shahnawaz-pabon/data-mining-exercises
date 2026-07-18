# 📖 Lecture: 3rd_sprint_Classification

---

# 1. Lecture Overview

## Main Topics
- **Supervised Learning** and where classification fits in data mining
- **Classification vs. Regression** (categorical vs. continuous targets)
- **Decision Trees**: structure, splitting, **Entropy**, **Information Gain**, Gini
- **Overfitting** in trees and how **pruning** / depth limits help
- **K-Nearest Neighbors (KNN)** as an instance-based classifier
- **Model Evaluation**: Confusion Matrix, **Accuracy, Precision, Recall, F1-Score**, ROC/AUC
- **Ensemble Learning**: **Bagging**, **Random Forest (RDF)**, **Boosting**, **AdaBoost**
- **Random Forest vs. AdaBoost** — overfitting, noise/outliers, imbalanced data

## Learning Objectives
After this lecture you should be able to:
1. Explain what supervised classification is and give real-world examples.
2. Build the intuition of a decision tree: choose the **best split** using **Entropy** and **Information Gain**.
3. Recognize and prevent **overfitting** (pruning, max depth, min samples).
4. Explain how **KNN** classifies a new point and the role of **k** and distance.
5. Read a **confusion matrix** and compute Accuracy, Precision, Recall, F1.
6. Explain **bagging vs. boosting** and how **Random Forest** and **AdaBoost** work.
7. Choose the right algorithm for a given scenario (robustness vs. accuracy trade-off).

## Important Takeaways
- Classification predicts **discrete class labels** from **labeled** training data.
- Trees split on the attribute with the **highest Information Gain** (largest entropy reduction).
- **Accuracy alone is misleading on imbalanced data** — use Precision/Recall/F1.
- **Bagging reduces variance** (Random Forest); **Boosting reduces bias** (AdaBoost) but is **sensitive to outliers**.
- Random Forest = parallel, independent trees + voting; AdaBoost = sequential weak learners that focus on previous mistakes.

---

# 2. Topic-wise Notes

## Topic 1: Supervised Learning

### Definition
Learning from data where each example already has a known **label** (the answer). The model learns the mapping from input features to labels so it can predict labels for new, unseen data.

### Simple Example
Spam filtering: thousands of emails already marked "spam"/"not spam" train a model that labels new incoming emails.

### Why is it Important?
It is the foundation of classification and regression — the most widely used family of data mining tasks in industry.

### Key Points
- Needs **labeled training data** (features **X** + target **y**).
- Two phases: **Training** (learn from labeled data) → **Prediction/Testing** (apply to unseen data).
- Two main task types: **Classification** (categorical output) and **Regression** (numerical output).
- Model quality is checked on a **separate test set**, never on training data.

### Common Mistakes
- Confusing supervised with unsupervised learning (clustering has **no labels**).
- Evaluating on the training set → overly optimistic results (overfitting goes unnoticed).

### Comparison

| Aspect | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Labels | Required | Not available |
| Goal | Predict a known target | Find hidden structure |
| Examples | Classification, Regression | Clustering, PCA |
| Evaluation | Compare prediction vs. true label | Internal measures (e.g., cohesion) |

| Aspect | Classification | Regression |
|---|---|---|
| Output | Categorical / discrete class | Continuous number |
| Question | "Which class?" | "How much?" |
| Example | Spam vs. not spam | House price |
| Algorithms | Decision Tree, KNN, RF, AdaBoost | Linear regression |

### Important Keywords
- **Label / Target (y):** the known answer for each training example.
- **Feature (X):** input attribute used for prediction.
- **Training set:** data used to fit the model.
- **Test set:** unseen data used to evaluate the model.

### Memory Tips
**Supervised = a Supervisor gives the answers** (labels). Classification → **Class** (category); Regression → **Real number**.

---

## Topic 2: Classification — Concept & Workflow

### Definition
Classification is the task of assigning a data point to one of a set of **predefined categories** using a model trained on labeled examples.

### Simple Example
A bank predicts whether a loan applicant is "credit-worthy" or "not credit-worthy" from income, age, and payment history.

### Why is it Important?
Most real business decisions are categorical (approve/reject, sick/healthy, churn/stay), making classification the core supervised task of this course.

### Key Points
- Pipeline: **Data → Split (train/test) → Train classifier → Predict → Evaluate**.
- **Binary classification**: 2 classes; **multi-class**: more than 2.
- The learned model defines a **decision boundary** separating classes in feature space.
- Different algorithms create different boundary shapes (trees → axis-parallel rectangles; KNN → irregular local boundaries).

### Common Mistakes
- Confusing classification with **clustering** (clustering discovers groups without labels).
- Ignoring class imbalance when interpreting results.

### Comparison

| Aspect | Classification | Clustering |
|---|---|---|
| Learning type | Supervised | Unsupervised |
| Labels | Known in advance | Unknown |
| Goal | Assign to predefined class | Discover natural groups |
| Evaluation | Accuracy, F1, etc. | Silhouette, cohesion |

### Diagram / Image Explanation
**Classification workflow diagram (slides):** raw labeled data flows into training; the trained model is then applied to new records and evaluated.
- **Why important:** exam questions often ask you to **order the pipeline steps**.
- **Possible exam question:** "Put in order: evaluate, train, split, collect labeled data." → collect → split → train → predict → evaluate.
- **Common mistake:** evaluating before/without splitting data.

### Important Keywords
- **Decision boundary:** the surface separating predicted classes.
- **Binary / multi-class:** number of possible output classes.
- **Classifier:** the trained model that assigns labels.

### Memory Tips
Think of a **sorting machine**: it was shown labeled examples first, then sorts new items into labeled bins.

---

## Topic 3: Decision Trees

### Definition
A decision tree is a flowchart-like model that classifies a sample by asking a sequence of **yes/no (or attribute-value) questions**, moving from the **root** through **internal nodes** to a **leaf** that holds the predicted class.

### Simple Example
"Should I play tennis?" — Root: *Outlook?* If Sunny → check *Humidity*; if Overcast → **Yes**; if Rain → check *Wind*.

### Why is it Important?
It is the lecture's central classifier: highly **interpretable**, and it is the **building block of Random Forest and AdaBoost**.

### Key Points
- **Root node** = best (most informative) attribute; **internal nodes** = tests; **branches** = attribute values; **leaves** = class labels.
- Built **top-down, greedily**: at each node pick the attribute with the **highest Information Gain**.
- Splitting stops when nodes are **pure** (one class), attributes run out, or stopping criteria (max depth, min samples) are met.
- Works with categorical and numerical features; **no feature scaling needed**.
- Produces **axis-parallel (rectangular) decision boundaries**.

### Common Mistakes
- Thinking the root attribute is arbitrary — it is the attribute with **maximum Information Gain**.
- Believing deeper trees are always better — deep trees **overfit**.
- Confusing leaf (decision/class) with internal node (test).

### Comparison

| Aspect | Decision Tree | Random Forest |
|---|---|---|
| Models | Single tree | Many trees (ensemble) |
| Overfitting | High risk | Robust (bagging averages it out) |
| Interpretability | High (visualizable) | Low (black-box-ish) |
| Stability | Small data change → very different tree | Stable |
| Accuracy | Lower | Usually higher |

### Formula Explanation — Entropy
**H(S) = − Σᵢ pᵢ · log₂(pᵢ)**
- **Variables:** *pᵢ* = proportion of class *i* in set *S*.
- **Meaning:** measures **impurity/uncertainty/disorder** of a dataset.
- **Intuition:** all one class → H = 0 (pure, no surprise); 50/50 two classes → H = 1 bit (maximum uncertainty).
- **When used:** at every candidate split during tree building.
- **Interpretation:** lower entropy after a split = better split.
- **Common mistakes:** forgetting log base 2; thinking entropy can be negative (it cannot); mixing up "high entropy = pure" (it is the opposite: **high entropy = impure**).

### Formula Explanation — Information Gain
**IG(S, A) = H(S) − Σᵥ (|Sᵥ|/|S|) · H(Sᵥ)**
- **Variables:** H(S) = entropy of parent node; Sᵥ = subset for attribute value v; |Sᵥ|/|S| = weight of subset.
- **Meaning:** the **reduction in entropy** achieved by splitting on attribute A.
- **Intuition:** "How much uncertainty does this question remove?"
- **When used:** to select the **best splitting attribute** at each node (highest IG wins → root).
- **Interpretation:** IG = 0 → useless split; IG = H(S) → perfect split into pure children.
- **Common mistakes:** forgetting to **weight** child entropies by subset size; choosing the attribute with **lowest** IG (it must be the **highest**).

### Diagram / Image Explanation
- **Tree diagram (root → branches → leaves):** shows how one path from root to leaf = one classification rule. *Exam question:* "Classify this new record by following the tree." *Mistake:* reading branches in the wrong order.
- **Best-attribute example (slides 7–9):** entropy computed before and after candidate splits; the attribute with the largest drop (highest IG) becomes the root. *Exam question:* "Given these counts, which attribute is chosen?"
- **Decision-boundary quiz figure:** a tree carves the feature space into **axis-parallel rectangles** — if the plotted boundary is diagonal/curved, it is **not** a plain decision tree.

### Important Keywords
- **Root node:** first/best splitting attribute.
- **Leaf node:** terminal node holding a class label.
- **Pure node:** contains samples of only one class (entropy 0).
- **Greedy algorithm:** locally best split at each step, no backtracking.
- **Gini impurity:** alternative impurity measure to entropy.

### Memory Tips
- **Entropy = mess**: a messy (mixed) room has high entropy; a tidy (pure) one has entropy 0.
- **IG = Information Gained = mess removed** — pick the question that cleans up the most.
- Coin flip 50/50 → entropy exactly **1**.

---

## Topic 4: Overfitting & Pruning

### Definition
Overfitting means the model memorizes training data (including noise) instead of learning general patterns, so it performs well on training data but poorly on new data. Pruning cuts back an overly complex tree to improve generalization.

### Simple Example
A tree with a leaf for every single training customer classifies training data 100% correctly but fails on any new customer.

### Why is it Important?
It is the **key limitation of decision trees** highlighted in the lecture and the main motivation for **Random Forest**.

### Key Points
- Symptom: **high training accuracy, low test accuracy**.
- Causes: tree too **deep**, too few samples per leaf, noisy data.
- Remedies: **pruning**, limiting **max depth**, **min samples per split/leaf**, or using **ensembles (bagging)**.
- Decision trees are also **unstable**: small data changes can produce a completely different tree.

### Common Mistakes
- Confusing overfitting (too complex) with **underfitting** (too simple → bad on both train and test).
- Assuming more depth always improves the model.

### Comparison

| Aspect | Overfitting | Underfitting |
|---|---|---|
| Model complexity | Too high | Too low |
| Training error | Very low | High |
| Test error | High | High |
| Fix | Prune, regularize, ensemble | More complex model / features |

### Important Keywords
- **Pruning:** removing branches that add little predictive power.
- **Generalization:** performance on unseen data.
- **Variance:** sensitivity of the model to the specific training sample.

### Memory Tips
Overfitting = **memorizing the textbook answers**; you fail when the exam asks anything new.

---

## Topic 5: K-Nearest Neighbors (KNN)

### Definition
KNN classifies a new point by looking at the **k closest training points** (by distance) and assigning the **majority class** among them. It builds no explicit model — it just stores the data.

### Simple Example
To guess a new house's neighborhood price class, look at the 5 nearest houses; if 3 of 5 are "expensive," predict "expensive."

### Why is it Important?
Simplest instance-based ("lazy") classifier; contrasts with the model-based ("eager") decision tree and shows why **distance and scaling** matter.

### Key Points
- **Lazy learner:** no training phase; all work happens at prediction time.
- Distance usually **Euclidean** — so **feature scaling/normalization is required**.
- **k small** → noisy, overfits; **k large** → too smooth, underfits; choose **odd k** (binary) to avoid ties.
- Prediction is slow on large datasets (must compare to all points).

### Common Mistakes
- Forgetting to normalize features (a large-scale feature dominates distance).
- Using even k in binary problems → ties.
- Thinking KNN "trains" a model — it does not.

### Comparison

| Aspect | KNN | Decision Tree |
|---|---|---|
| Learning style | Lazy (no training) | Eager (builds model) |
| Prediction speed | Slow (search neighbors) | Fast (follow path) |
| Feature scaling | Required | Not required |
| Interpretability | Low | High |
| Boundary shape | Irregular, local | Axis-parallel rectangles |

### Important Keywords
- **k:** number of neighbors consulted.
- **Euclidean distance:** straight-line distance between points.
- **Majority vote:** class chosen most often among neighbors.
- **Lazy learning:** deferring computation until prediction.

### Memory Tips
**"Tell me who your neighbors are and I'll tell you who you are."** Small k = listening to one loud neighbor; large k = averaging the whole town.

---

## Topic 6: Model Evaluation — Confusion Matrix & Metrics

### Definition
A confusion matrix is a table comparing **predicted vs. actual** classes, counting **TP, TN, FP, FN**. From it we derive Accuracy, Precision, Recall, and F1-Score.

### Simple Example
COVID test: TP = sick & test positive; FP = healthy but test positive (false alarm); FN = sick but test negative (missed case, dangerous!); TN = healthy & negative.

### Why is it Important?
Exams love metric calculations; and **accuracy alone is misleading** for imbalanced classes — the lecture stresses using precision/recall/F1.

### Key Points
- **TP/TN** = correct predictions; **FP** = Type I error (false alarm); **FN** = Type II error (miss).
- **Accuracy** works only when classes are balanced.
- **Precision**: of all predicted positives, how many are truly positive.
- **Recall (Sensitivity)**: of all actual positives, how many were found.
- **F1**: harmonic mean of precision and recall — best single number under imbalance.
- **Precision–recall trade-off**: raising one usually lowers the other.

### Common Mistakes
- Swapping precision and recall definitions.
- Reporting 99% accuracy on a dataset with 99% negatives (a useless "always negative" model achieves this).
- Using arithmetic instead of **harmonic** mean for F1.

### Comparison

| Metric | Formula | Answers the question | Prioritize when… |
|---|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | How often is the model right? | Classes balanced |
| Precision | TP/(TP+FP) | Of predicted positives, how many are real? | FP is costly (spam filter) |
| Recall | TP/(TP+FN) | Of real positives, how many found? | FN is costly (cancer, fraud) |
| F1-Score | 2·(P·R)/(P+R) | Balance of P and R | Imbalanced data |

### Formula Explanation
- **Accuracy = (TP+TN)/Total** — overall correctness; fails on imbalance.
- **Precision = TP/(TP+FP)** — "quality" of positive predictions; mistake: putting FN in the denominator.
- **Recall = TP/(TP+FN)** — "coverage" of actual positives; mistake: confusing with precision.
- **F1 = 2PR/(P+R)** — harmonic mean punishes an extreme imbalance between P and R; F1 is high only when **both** are high.

### Diagram / Image Explanation
**Confusion matrix 2×2 grid:** rows = actual, columns = predicted (or vice versa — check labels!).
- **Exam question:** "Given this matrix, compute precision and recall."
- **Common mistake:** mixing up the row/column orientation → swapping FP and FN.

### Important Keywords
- **TP / TN / FP / FN:** the four confusion-matrix cells.
- **Sensitivity:** another name for recall.
- **False alarm:** FP. **Missed detection:** FN.
- **Imbalanced dataset:** one class heavily outnumbers the other.

### Memory Tips
- **P**recision = **P**redicted positives as denominator base.
- **R**ecall = **R**eal (actual) positives as base.
- Doctor story: **Recall = don't miss any sick patient**; **Precision = don't scare healthy ones**.

---

## Topic 7: Ensemble Learning

### Definition
Ensemble learning combines **multiple models ("weak learners")** into one stronger model, because a committee of diverse models makes fewer mistakes than any single one.

### Simple Example
"Ask the audience" in a quiz show: the aggregated vote of many people beats most individual guesses.

### Why is it Important?
It solves the decision tree's overfitting/instability problems and leads to the two star algorithms of this lecture: **Random Forest** and **AdaBoost**.

### Key Points
- Two main strategies: **Bagging** (parallel) and **Boosting** (sequential).
- **Bagging (Bootstrap Aggregating):** train each model on a random **bootstrap sample** (sampling **with replacement**); combine by **majority vote** → **reduces variance**.
- **Boosting:** train models **one after another**; each new model focuses on the **misclassified samples** of the previous ones → **reduces bias**.
- Diversity among the base models is essential.

### Common Mistakes
- Thinking bagging models are trained sequentially (they are **parallel & independent**).
- Confusing which technique reduces variance (bagging) vs. bias (boosting).
- Forgetting bootstrap sampling is **with replacement**.

### Comparison

| Aspect | Bagging | Boosting |
|---|---|---|
| Training | Parallel, independent | Sequential, dependent |
| Data per model | Bootstrap samples (with replacement) | Reweighted full data (focus on errors) |
| Combines by | Equal-weight majority vote | Weighted vote (better learners count more) |
| Reduces | **Variance** (overfitting) | **Bias** |
| Outlier sensitivity | Low (averaging) | **High** (chases hard points) |
| Example | Random Forest | AdaBoost |

### Important Keywords
- **Weak learner:** simple model only slightly better than random guessing.
- **Bootstrap sample:** random sample drawn **with replacement**.
- **Aggregation:** combining predictions (voting/averaging).

### Memory Tips
- **B**agging = **B**ootstrap + parallel **B**allot (vote).
- Boosting = a relay team where each runner fixes the previous runner's mistakes.

---

## Topic 8: Random Forest (RDF)

### Definition
Random Forest builds **many decision trees**, each on a **bootstrap sample** and each considering only a **random subset of features** at every split; the final class is decided by **majority vote**.

### Simple Example
Netflix predicting what a new user will like: hundreds of trees each look at different user aspects (age, location, viewing time); the forest votes on the recommendation (lecture's streaming-service example).

### Why is it Important?
The lecture's flagship bagging method: **robust to overfitting, noise, and outliers** thanks to averaging.

### Key Points
- Two sources of randomness: **random data** (bootstrap) + **random features** per split → decorrelated trees.
- **Majority voting** across trees gives the final prediction.
- More robust and stable than a single tree; handles many features well.
- Downside: **loses interpretability** ("hard to explain why it recommends a specific movie") and costs more computation.

### Common Mistakes
- Forgetting the **random feature subset** — that is what makes it a *random* forest, not just bagged trees.
- Claiming RF is easy to interpret like a single tree (it is not).

### Comparison

| Aspect | Random Forest | AdaBoost |
|---|---|---|
| Ensemble type | Bagging (parallel) | Boosting (sequential) |
| **Overfitting** | **More robust** — bagging reduces variance | More prone |
| **Noise/outliers** | **Less sensitive** (averaging) | **Sensitive** — fits misclassified points, which may be outliers |
| **Imbalanced data** | Handles reasonably | Focuses on misclassified rather than minority class; may create **redundant weak classifiers** → overhead & performance drop |
| Base learners | Full/deep trees | Stumps (weak learners) |
| Vote | Equal weight | Weighted by learner accuracy |

*(This table reproduces the lecture's "Adaptive Boosting vs. RDF" slide — very exam-relevant.)*

### Diagram / Image Explanation
**Forest diagram: one dataset → many bootstrap samples → many trees → majority vote → final class.**
- **Exam question:** "Explain the two levels of randomness in RF." / "How is the final prediction made?"
- **Common mistake:** saying trees are trained sequentially.

### Important Keywords
- **Bootstrap aggregating (bagging):** RF's core mechanism.
- **Random feature subset:** features considered at each split.
- **Majority vote:** final decision rule.
- **Variance reduction:** why RF resists overfitting.

### Memory Tips
**Forest = many trees + democracy.** Random twice: random **rows** (bootstrap) and random **columns** (features).

---

## Topic 9: Boosting & AdaBoost (Adaptive Boosting)

### Definition
AdaBoost trains a sequence of **weak learners** (often decision stumps); after each round it **increases the weights of misclassified samples** so the next learner focuses on them, and combines all learners with a **weighted vote**.

### Simple Example
Music recommendation (lecture homework example): start with a simple rule — "people under 25 like pop"; the next rule corrects mistakes — "but if they follow many indie artists, they prefer indie pop"; each rule patches the previous ones.

### Why is it Important?
The lecture's flagship boosting method; exam contrasts with RF center on its **strengths (high accuracy, bias reduction)** and **weaknesses (outlier sensitivity, imbalance issues)**.

### Key Points
- **Sequential**: each learner depends on the previous learners' errors.
- **Adaptive sample weights:** misclassified points get **higher weight**; correctly classified get lower.
- **Weighted final vote:** more accurate learners get a bigger say.
- **Sensitive to noise/outliers**, because it keeps trying to fit hard (possibly noisy) points.
- On **imbalanced data**: focuses on misclassified samples rather than the minority class; can generate **many redundant/useless weak classifiers**, increasing processing overhead and reducing performance (lecture slide).
- Can achieve **very high accuracy** but "can be thrown off by a few unusual users."

### Common Mistakes
- Saying AdaBoost reweights the *models* only — it reweights **samples** during training *and* weights **models** in the final vote.
- Claiming AdaBoost is robust to outliers (it is the opposite).
- Confusing bagging's equal vote with boosting's **weighted** vote.

### Comparison
See the **Random Forest vs. AdaBoost** table in Topic 8 (identical to the lecture's comparison slide).

### Diagram / Image Explanation
**Boosting sequence diagram: Learner 1 → errors highlighted/enlarged → Learner 2 trained on reweighted data → … → weighted combination.**
- **Exam question:** "What happens to the weights of misclassified samples after each round?" → They **increase**.
- **Common mistake:** thinking every learner sees the same, unweighted data.

### Important Keywords
- **Weak learner / decision stump:** one-split tree used as base model.
- **Sample weight:** importance of a training point, adapted each round.
- **Weighted vote:** final combination favoring accurate learners.
- **Sequential training:** each model builds on previous errors.

### Memory Tips
**Ada = ADAptive**: it **adapts weights** to its mistakes. Boosting = a tutor who spends extra time on the questions you got wrong.

---

## Topic 10: Choosing Between RF and AdaBoost (Decision Guidance)

### Definition
Algorithm choice depends on priorities: a **robust** model handling a wide range of scenarios (Random Forest) vs. a **highly accurate** model that may be sensitive to unusual data (AdaBoost).

### Simple Example
Streaming service (lecture): RF gives a good, robust overall prediction; AdaBoost gives sharper rules but can be misled by a few unusual users.

### Why is it Important?
This is exactly the lecture's closing homework question — a near-certain exam scenario question.

### Key Points
- Noisy data / many outliers → **Random Forest**.
- Clean data, want maximum accuracy → **AdaBoost**.
- Need robustness to overfitting → **Random Forest** (bagging reduces variance).
- Both lose the interpretability of a single decision tree.

### Memory Tips
**Robust Forest, Accurate (but touchy) Boost.**

---

# 3. Topic-wise Exam Practice

## Supervised Learning & Classification
1. **MCQ:** Which task requires labeled data? a) Clustering b) Classification c) PCA d) Association rules → **b**
2. **True/False:** Regression predicts categorical labels. → **False** (continuous values)
3. **Fill in:** In supervised learning, the known answer for each sample is called the ______. → **label/target**
4. **One-word:** Learning without labels is called? → **Unsupervised**
5. **Scenario:** Predicting whether a customer will churn (yes/no). Which analysis type? → **Binary classification**
6. **Pipeline ordering:** Order: evaluate / train / split data / collect labeled data / predict → **collect → split → train → predict → evaluate**

## Decision Trees, Entropy & Information Gain
7. **MCQ:** The attribute chosen as root has… a) lowest entropy b) **highest information gain** c) most values d) fewest values → **b**
8. **Formula interpretation:** A node has 50% class A, 50% class B. Entropy = ? → **1** (maximum for two classes)
9. **True/False:** A pure node has entropy 0. → **True**
10. **Short answer:** Why is decision tree building called "greedy"? → It picks the locally best split at each node without revisiting earlier choices.
11. **Fill in:** Information Gain = Entropy(parent) − ______. → **weighted average entropy of children**
12. **Two-word:** The terminal node of a tree holding a class label. → **Leaf node**
13. **Scenario:** Splitting on "CustomerID" gives perfect training purity. Good idea? → No — it does not generalize; classic **overfitting** on a useless attribute.
14. **Diagram interpretation:** A decision boundary plot shows only axis-parallel rectangles. Which classifier? → **Decision tree**

## Overfitting
15. **MCQ:** High train accuracy + low test accuracy indicates: a) underfitting b) **overfitting** c) good fit d) data leak only → **b**
16. **Short answer:** Name two ways to reduce tree overfitting. → Pruning; limiting max depth / min samples; use Random Forest.
17. **True/False:** Underfit models perform well on training data. → **False**

## KNN
18. **MCQ:** KNN is called a ______ learner. a) eager b) **lazy** c) deep d) boosted → **b**
19. **Short answer:** Why must features be scaled for KNN? → Distance-based; large-scale features would dominate the Euclidean distance.
20. **Scenario:** k=1 gives 100% training accuracy but poor test accuracy. Why? → k=1 memorizes/overfits; every point is its own nearest neighbor.
21. **One-word:** Rule used by KNN to decide the class among neighbors. → **Majority (vote)**

## Evaluation Metrics
22. **Calculation:** TP=40, FP=10, FN=20, TN=30. Precision? Recall? Accuracy? → P = 40/50 = **0.80**; R = 40/60 ≈ **0.67**; Acc = 70/100 = **0.70**
23. **MCQ:** For cancer detection, the most critical error to minimize is: a) FP b) **FN** c) TN d) TP → **b** (missed sick patient)
24. **True/False:** Accuracy is a reliable metric for imbalanced datasets. → **False**
25. **Fill in:** F1 is the ______ mean of precision and recall. → **harmonic**
26. **Matching:** Match — Precision / Recall / Accuracy / F1 with: "found all real positives" / "predicted positives that are correct" / "overall correctness" / "balance of P & R" → Recall / Precision / Accuracy / F1
27. **Scenario:** 99% of emails are non-spam; a model predicting "never spam" gets 99% accuracy. What metric exposes it? → **Recall (=0)** and hence F1.

## Ensemble Learning / RF / AdaBoost
28. **MCQ:** Bagging primarily reduces: a) bias b) **variance** c) both equally d) neither → **b**
29. **True/False:** Bootstrap sampling is done without replacement. → **False** (with replacement)
30. **MCQ:** In AdaBoost, after a round, misclassified samples' weights: a) decrease b) **increase** c) stay same d) reset → **b**
31. **Two-word:** RF's final decision mechanism. → **Majority vote**
32. **Algorithm selection:** Dataset is noisy with many outliers. RF or AdaBoost? → **Random Forest** (averaging → less sensitive to outliers)
33. **Algorithm selection:** Clean data, top accuracy needed, interpretability unimportant. → **AdaBoost** (or RF; AdaBoost per lecture = "highly accurate")
34. **Short answer:** Why is AdaBoost problematic on imbalanced data? → It focuses on misclassified samples rather than the minority class and may create many redundant weak classifiers, adding overhead and hurting performance.
35. **True/False:** Random Forest trees are trained sequentially. → **False** (parallel/independent)
36. **Fill in:** AdaBoost combines learners with a ______ vote. → **weighted**
37. **Scenario (lecture homework):** A streaming service must choose between RF and AdaBoost. State the trade-off. → RF: robust, handles wide range of scenarios, harder to explain a specific recommendation; AdaBoost: iteratively learns from mistakes, highly accurate, but thrown off by a few unusual users.

---

# 4. Flashcards

- **What is supervised learning?** → Learning from labeled data to predict labels for new data.
- **Classification vs. regression output?** → Categorical class vs. continuous number.
- **What does entropy measure?** → Impurity/uncertainty of a dataset (0 = pure, 1 = max for 2 balanced classes).
- **Entropy formula?** → H = −Σ pᵢ log₂ pᵢ
- **Information Gain?** → Entropy(parent) − weighted entropy(children); pick the attribute with the **highest** IG.
- **Which attribute becomes the root?** → The one with maximum information gain.
- **What is a leaf node?** → Terminal node containing the class prediction.
- **What is overfitting?** → Memorizing training data (incl. noise); great train performance, poor test performance.
- **Two fixes for tree overfitting?** → Pruning; depth/min-sample limits (or ensembles).
- **Why is KNN "lazy"?** → No training phase; computes distances only at prediction time.
- **Why scale features for KNN?** → Distance-based; unscaled features distort neighbor selection.
- **Small k vs. large k?** → Small = overfits/noisy; large = underfits/oversmooth.
- **TP? FP? FN? TN?** → Correct positive; false alarm; missed positive; correct negative.
- **Precision formula?** → TP/(TP+FP) — quality of positive predictions.
- **Recall formula?** → TP/(TP+FN) — coverage of actual positives.
- **F1 formula?** → 2PR/(P+R), harmonic mean.
- **When is accuracy misleading?** → Imbalanced classes.
- **When prioritize recall?** → When missing a positive is costly (disease, fraud).
- **When prioritize precision?** → When false alarms are costly (spam filter).
- **What is ensemble learning?** → Combining many weak models into one stronger model.
- **Bagging?** → Bootstrap samples + parallel models + majority vote → reduces variance.
- **Boosting?** → Sequential models focusing on previous errors → reduces bias.
- **Bootstrap sample?** → Random sample with replacement.
- **Random Forest's two randomness sources?** → Bootstrap rows + random feature subsets per split.
- **RF final prediction?** → Majority vote of all trees.
- **AdaBoost weight update?** → Increase weights of misclassified samples each round.
- **AdaBoost final prediction?** → Weighted vote of weak learners.
- **Which is robust to outliers, RF or AdaBoost?** → Random Forest (averaging); AdaBoost is sensitive.
- **Which resists overfitting more?** → Random Forest (bagging reduces variance).
- **AdaBoost on imbalanced data?** → Focuses on misclassified not minority class; redundant weak classifiers → overhead + performance loss.
- **Weak learner?** → A model slightly better than random guessing (e.g., decision stump).

---

# 5. Rapid Revision

- **Supervised** = labeled data; **classification** = categorical target; **regression** = numeric target.
- **Decision tree:** root (best attribute) → tests → leaves (classes); greedy top-down building.
- **Entropy** −Σp log₂p: 0 = pure, 1 = 50/50. **IG** = parent entropy − weighted child entropy; **highest IG splits**.
- **Trees overfit** → prune / limit depth / use ensembles; boundaries are axis-parallel rectangles.
- **KNN:** lazy, distance-based, needs scaling, odd k, majority vote; small k overfits.
- **Confusion matrix:** TP, FP (false alarm), FN (dangerous miss), TN.
- **Accuracy** (TP+TN)/all — fails on imbalance; **Precision** TP/(TP+FP); **Recall** TP/(TP+FN); **F1** harmonic mean.
- **Bagging** = parallel + bootstrap + equal vote → less **variance**; **Boosting** = sequential + reweighting + weighted vote → less **bias**.
- **RF:** many decorrelated trees (random rows + random features), majority vote; robust to overfitting, noise, outliers; less interpretable.
- **AdaBoost:** stumps in sequence; misclassified samples get heavier; weighted final vote; very accurate but **sensitive to outliers** and weak on **imbalanced data**.
- **Choice:** robustness → RF; peak accuracy on clean data → AdaBoost.

---

# 6. High Probability Exam Questions

### ★★★★★ Very High
1. Compare **Random Forest vs. AdaBoost** on overfitting, noise/outliers, and imbalanced data (dedicated lecture slide!).
2. Compute **entropy and information gain** and pick the best splitting attribute.
3. From a **confusion matrix**, compute accuracy, precision, recall, F1.
4. Explain why **accuracy fails on imbalanced data** and what to use instead.
5. Explain how **AdaBoost adapts sample weights** and combines learners.
6. Explain **bagging vs. boosting** (parallel vs. sequential; variance vs. bias).

### ★★★★ High
7. Define **overfitting**, its symptoms, and remedies for decision trees.
8. Describe the **Random Forest algorithm** (bootstrap + random features + majority vote).
9. Explain how **KNN** classifies and the effect of k; why scaling matters.
10. Structure of a decision tree (root/internal/leaf) and how a record is classified.
11. Scenario question: choose RF or AdaBoost for a streaming-service recommender and justify (lecture homework).
12. Precision vs. Recall — which to prioritize in medical diagnosis vs. spam filtering.

### ★★★ Medium
13. Supervised vs. unsupervised; classification vs. clustering; classification vs. regression.
14. Why decision trees are unstable / greedy.
15. What is a weak learner / decision stump.
16. Decision-boundary shape identification (tree → rectangles).
17. What is bootstrap sampling (with replacement).

---

# 7. Lecture Cheat Sheet

**Definitions**
- Classification: assign predefined class labels using labeled training data.
- Entropy: impurity measure, H = −Σp log₂p (0 pure, 1 max for 2 classes).
- Information Gain: entropy reduction from a split; max IG → root.
- Overfitting: memorize train data, fail on test.
- KNN: classify by majority vote of k nearest points.
- Bagging: bootstrap + parallel models + vote (↓variance).
- Boosting: sequential learners focusing on errors (↓bias).
- Random Forest: bagged trees + random feature subsets + majority vote.
- AdaBoost: reweight misclassified samples; weighted vote of weak learners.

**Key Formulas**
| Formula | Expression |
|---|---|
| Entropy | H = −Σ pᵢ log₂ pᵢ |
| Information Gain | IG = H(parent) − Σ(|Sᵥ|/|S|)·H(Sᵥ) |
| Accuracy | (TP+TN)/(TP+TN+FP+FN) |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN) |
| F1 | 2PR/(P+R) |

**Frequently Confused**
- Precision (predicted-positive base) vs. Recall (actual-positive base)
- Overfitting (too complex) vs. underfitting (too simple)
- Bagging (parallel, variance↓) vs. boosting (sequential, bias↓)
- Classification (labels) vs. clustering (no labels)
- FP (false alarm, Type I) vs. FN (miss, Type II)

**Algorithm Pick Guide**
- Interpretability → Decision Tree
- Noisy/outlier data, robustness → Random Forest
- Max accuracy, clean data → AdaBoost
- Small dataset, simple baseline → KNN (scale features!)

**Pipeline:** labeled data → train/test split → train → predict → evaluate (confusion matrix → metrics).

---

# 8. Lecture Summary

## Top Probable Exam Questions (condensed 50)
1–6. The six ★★★★★ questions above.
7. Define supervised learning; 8. Classification vs. regression; 9. Classification vs. clustering; 10. Draw/label tree parts; 11. Entropy of a pure node; 12. Entropy of a 50/50 node; 13. Why highest IG?; 14. Greedy meaning; 15. Tree decision-boundary shape; 16. Two overfitting fixes; 17. Underfitting vs. overfitting; 18. Why is KNN lazy; 19. Effect of k; 20. Why scale for KNN; 21. Odd k reason; 22. Define TP/FP/FN/TN; 23. Compute all four metrics from a matrix; 24. F1 = harmonic mean — why; 25. Recall-priority scenario; 26. Precision-priority scenario; 27. Imbalance and accuracy paradox; 28. Define ensemble; 29. Weak learner; 30. Bootstrap sample definition; 31. Bagging steps; 32. Boosting steps; 33. Which reduces variance/bias; 34. RF's two randomness levels; 35. RF final vote; 36. Why RF resists overfitting; 37. RF interpretability drawback; 38. AdaBoost weight update rule; 39. AdaBoost final weighted vote; 40. AdaBoost outlier sensitivity — why; 41. AdaBoost on imbalanced data (redundant weak classifiers); 42. RF vs. AdaBoost: overfitting; 43. RF vs. AdaBoost: noise; 44. RF vs. AdaBoost: imbalance; 45. Streaming-service scenario choice; 46. Sequential vs. parallel training; 47. Equal vs. weighted vote; 48. Decision stump definition; 49. Order the classification pipeline; 50. Identify classifier from a decision-boundary plot.

## Top Definitions (30)
Supervised learning · Label · Feature · Training/Test set · Classification · Regression · Clustering · Decision boundary · Decision tree · Root node · Internal node · Leaf node · Pure node · Entropy · Information Gain · Gini impurity · Greedy algorithm · Overfitting · Underfitting · Pruning · KNN · Lazy learner · Euclidean distance · Confusion matrix · Precision · Recall · F1-Score · Ensemble · Bagging/Bootstrap · Boosting/AdaBoost/Random Forest/weak learner.

## Top Keywords (30)
TP, FP, FN, TN, accuracy, sensitivity, majority vote, weighted vote, bootstrap (with replacement), variance, bias, decision stump, sample weights, sequential, parallel, axis-parallel boundary, max depth, min samples, generalization, imbalanced data, minority class, redundant weak classifiers, noise, outliers, robustness, interpretability, k (neighbors), feature scaling, harmonic mean, train/test split.

## Top Comparisons (20 → key 10)
Supervised vs. Unsupervised · Classification vs. Regression · Classification vs. Clustering · Overfitting vs. Underfitting · Precision vs. Recall · Accuracy vs. F1 · KNN vs. Decision Tree · Decision Tree vs. Random Forest · Bagging vs. Boosting · **Random Forest vs. AdaBoost** (overfitting / noise / imbalance — the lecture's own slide).

## Top Formulas
Entropy · Information Gain · Accuracy · Precision · Recall · F1 (see cheat sheet).

## Top Diagrams
Classification pipeline · Decision tree structure · Entropy/IG split example · Decision-boundary plots (quiz) · Confusion matrix · Bagging diagram (bootstrap → trees → vote) · Boosting sequence (reweighting) · RF architecture · AdaBoost stump sequence · RF-vs-AdaBoost comparison table.

## 10-Minute Revision Sheet
1. Supervised = labels; classification = category output.
2. Tree: split by **max Information Gain**; entropy 0 = pure, 1 = 50/50.
3. Trees overfit → prune/limit depth → or use **ensembles**.
4. KNN: lazy, distance, scale features, majority of k neighbors.
5. Metrics: **P = TP/(TP+FP)**, **R = TP/(TP+FN)**, F1 = harmonic mean; accuracy lies under imbalance.
6. **Bagging** parallel/variance↓ (**Random Forest**: bootstrap + random features + majority vote).
7. **Boosting** sequential/bias↓ (**AdaBoost**: reweight errors, weighted vote).
8. RF robust to overfitting/noise/outliers; AdaBoost accurate but outlier-sensitive and weak on imbalance (redundant weak classifiers).
9. Choose by priority: **robustness → RF; accuracy → AdaBoost**.

---

# Examples Repository

### 1. Streaming Service (Netflix/Spotify) — RF vs. AdaBoost *(Homework slide 50)*
- **Topic:** Random Forest vs. Adaptive Boosting.
- **Lecture context:** "Imagine we're working for a major streaming service… predict what movies or music a new user will like based on their profile and behavior."
- **Simple explanation:** RF uses many factors (age, location, listening time) → robust overall prediction but hard to explain a specific recommendation. AdaBoost starts with a simple rule ("people under 25 usually like pop") and iteratively adds correcting rules ("but if they follow many indie artists → indie pop") → highly accurate but can be thrown off by a few unusual users.
- **Why important:** it is the lecture's official homework and encapsulates the whole RF-vs-AdaBoost trade-off.
- **Possible exam questions:** "Which algorithm would you choose for the recommender and why?" / "Give the AdaBoost-style rule sequence from the example." / "What is the stated trade-off at the end?" → *robust & wide-ranging vs. highly accurate but sensitive to unusual data.*

### 2. Play Tennis / Weather-style Split Example *(Decision tree slides)*
- **Topic:** Entropy & Information Gain.
- **Lecture context:** worked example computing entropy before/after candidate splits to select the best (root) attribute.
- **Simple explanation:** compute parent entropy, then weighted child entropies per attribute; the attribute with the biggest entropy drop wins.
- **Why important:** the canonical calculation exam question.
- **Possible exam question:** "Given these class counts per attribute value, compute IG and choose the split."

### 3. Decision-Boundary Quiz Figures
- **Topic:** Decision tree geometry / classifier identification.
- **Lecture context:** quiz slides showing 2-D scatter plots with class regions.
- **Simple explanation:** trees always cut parallel to the axes → rectangular regions; other classifiers produce curved/irregular boundaries.
- **Why important:** trains visual identification of classifiers.
- **Possible exam question:** "Which of these boundary plots was produced by a decision tree?"

### 4. Medical / Diagnostic Metric Example
- **Topic:** Precision vs. Recall.
- **Lecture context:** interpreting FP vs. FN costs.
- **Simple explanation:** FN (missed sick patient) is far more dangerous than FP (extra test) → prioritize **recall** in medicine.
- **Possible exam question:** "In disease screening, which metric matters most and why?"

### 5. Ask-the-Audience / Committee Intuition
- **Topic:** Ensemble learning.
- **Simple explanation:** many diverse weak opinions aggregated outperform a single expert — the motivation for bagging & boosting.
- **Possible exam question:** "Why do ensembles outperform single models?" → error averaging / diversity.

---

# Quiz Repository

### Q1. Decision-boundary quiz *(slide ~16–18)*
- **Question:** Which decision boundary corresponds to a decision tree classifier?
- **Correct answer:** The plot with **axis-parallel rectangular regions**.
- **Explanation:** Each tree node tests one feature against a threshold, producing straight, axis-aligned cuts; nested tests create rectangles.
- **Related topic:** Decision Trees.
- **Difficulty:** Medium.

### Q2. Best-attribute selection exercise *(slides 7–9)*
- **Question:** Given the class distributions for each candidate attribute, which attribute should be chosen for the split?
- **Correct answer:** The attribute with the **highest Information Gain** (largest entropy reduction).
- **Explanation:** IG = H(parent) − weighted H(children); the greedy algorithm always maximizes IG at each node.
- **Related topic:** Entropy & Information Gain.
- **Difficulty:** Medium.

### Q3. Entropy checkpoint
- **Question:** What is the entropy of a node containing only one class? Of a 50/50 node?
- **Inferred Answer:** **0** and **1** respectively.
- **Explanation:** −1·log₂1 = 0; −(0.5 log₂0.5 + 0.5 log₂0.5) = 1.
- **Related topic:** Entropy.
- **Difficulty:** Easy.

### Q4. Metrics checkpoint
- **Question:** Given a confusion matrix, compute accuracy, precision, recall.
- **Inferred Answer:** Apply (TP+TN)/total, TP/(TP+FP), TP/(TP+FN).
- **Explanation:** Precision uses the *predicted-positive* column; recall uses the *actual-positive* row.
- **Related topic:** Model evaluation.
- **Difficulty:** Medium.

### Q5. Ensemble discussion question
- **Question:** Why does combining many weak models often beat one strong model?
- **Inferred Answer:** Diverse models make different errors; aggregation (voting/averaging) cancels individual mistakes, reducing variance (bagging) or bias (boosting).
- **Related topic:** Ensemble learning.
- **Difficulty:** Medium.

### Q6. Homework — RF vs. AdaBoost for a streaming service *(slide 50)*
- **Question:** For predicting what a new Netflix/Spotify user will like, should we use Random Forest or Adaptive Boosting?
- **Correct answer (from the slide):** It **depends on priorities** — RF if we want a model that is **robust and handles a wide range of scenarios**; AdaBoost if we want one that is **highly accurate but might be sensitive to unusual data**.
- **Explanation:** RF averages many diverse trees (robust to noise/outliers, harder to explain); AdaBoost iteratively corrects mistakes with rules (accurate, but a few unusual users can throw it off).
- **Related topic:** Random Forest vs. AdaBoost.
- **Difficulty:** Hard.

### Q7. Comparison-slide check *(slide 49)*
- **Question:** State how RF and AdaBoost differ on (a) overfitting, (b) noise/outliers, (c) imbalanced data.
- **Correct answer:** (a) RF is **more robust to overfitting** because bagging reduces variance. (b) AdaBoost is **sensitive to outliers** (fits misclassified instances, which may be outliers); RF is **less sensitive** due to averaging. (c) AdaBoost can process imbalanced data but focuses on misclassified samples rather than the minority class, and may generate **many redundant/useless weak classifiers**, increasing overhead and reducing performance.
- **Related topic:** Ensemble comparison.
- **Difficulty:** Hard.

### Q8. Final Questions & Answers session *(slide 51)*
- **Question:** Open Q&A (puzzle slide — "Questions / Answers").
- **Answer:** Use this guide's Rapid Revision and High-Probability lists as your self-Q&A.
- **Related topic:** Whole lecture.
- **Difficulty:** —

---