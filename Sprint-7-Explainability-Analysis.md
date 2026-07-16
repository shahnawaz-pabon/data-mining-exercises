# 📖 Lecture: Explainability_analysis.pdf

---

# 1. Lecture Overview

## Main Topics Covered
1. **Introduction to Explainability** (what, why, who needs it)
2. **Types of Explainability** (Global vs Local, Inherent vs Post-hoc)
3. **Feature Importance of inherently explainable models** (Decision Trees, Logistic Regression)
4. **Permutation Feature Importance** (algorithm, formula, pros/cons)
5. **Shapley Values / SHAP** (fair attribution, weighting schema, plots)
6. **Introduction to Causal Analysis** (correlation vs causation, confounders)

## Learning Objectives
- Understand **why** model decisions must be understandable to humans (not just *what*, but *why* and *how*)
- Distinguish **global vs local** and **inherent vs post-hoc** explainability
- Explain and apply the **permutation feature importance algorithm**
- Understand **Shapley values** as fair credit distribution among features
- Recognize that **explainability ≠ causality**, and know the role of **confounders**

## Important Takeaways
- **Explainability = making model decisions understandable to humans.**
- **Trade-off**: interpretable models (decision trees) vs accurate models (neural networks).
- **Permutation importance** = performance drop when a feature is shuffled.
- **SHAP** = fair, averaged marginal contribution of each feature (from game theory).
- ⚠️ **SHAP values do NOT reflect causality** — the most repeated warning in the lecture.
- **Causal analysis** answers "what happens if we intervene?", explainability answers "what influenced the model?"

---

# 2. Topic-wise Notes

---

## Topic 1: Explainability (Explainable Modelling / XAI)

### Definition
Explainability means making a model's decisions **understandable to humans**. It is not only about *what* the model decided, but *why* and *how* it decided it.

### Simple Example
A hospital AI recommends a risky, expensive treatment. Without explainability, the doctor cannot justify it to the patient. With explainability, the model highlights the **symptoms, risk factors, and patterns** it used, so the doctor can verify it against medical judgment.

### Why is it Important?
It builds **trust**, enables **error correction**, satisfies **regulations**, and lets stakeholders **validate** and **act** on model decisions — critical when stakes are high.

### Key Points
- Explainability = **understandable to humans** — the *why* and *how*, not just the *what*.
- **Five purposes of explaining** (cycle diagram): Explain to **justify** → **improve** → **build trust** → **control** → **discover**.
- **Different stakeholders** need explainability for different reasons (loan approval example):
  - **Developer**: "Why does the model work like this? How do I correct an error?"
  - **Domain user**: "Why is this client risky? Can I trust this model?"
  - **End user**: "Why was my loan denied?"
  - **Auditor**: "Are minority groups disproportionately denied? Does it use appropriate factors (income) and exclude prohibited ones (race)?"
- **Matters more when**: decisions affect **people's lives** (healthcare, finance, justice), **regulations** require transparency, **trust and accountability** are critical.
- **Matters less when**: decisions are **low risk** (product recommendations), or **speed/accuracy** matters more than interpretability.
- **Bottomline**: Explainability is essential when **stakes are high** — legally, ethically, or socially.
- Fields where crucial: **Transportation, Finance, Security, Legal, Medicine, Military**.

### Common Mistakes
- ❌ Thinking explainability is only about model accuracy — it's about **human understanding**.
- ❌ Assuming every application needs explainability — low-risk applications (recommendations) do not.
- ❌ Confusing the "what" (prediction) with the "why/how" (explanation).

### Comparison: Trade-off Explainability vs Accuracy

| | **Decision Tree** | **Neural Network** |
|---|---|---|
| Interpretability | 🙂 High | ☹️ Low |
| Accuracy (complex tasks) | ☹️ Lower | 🙂 High |

- Highly accurate models (deep NNs) often lack transparency; interpretable models (decision trees, linear regression) may sacrifice predictive performance.
- Business view: model choice balances **ROI** (profit) and **compliance** (legal/societal standards).

### Formula Explanation
None for this topic.

### Diagram / Image Explanation
1. **Rocket "WHAT / HOW / WHY"**: The rocket that flies has the **WHY** stage — models with the "why" (explanation/causality) go further than those with only "what" and "how". *Exam question: what does the rocket metaphor represent?*
2. **Explain-to cycle**: Justify → Improve → Build trust → Control → Discover — the 5 motivations for explainability.
3. **Loan approval stakeholders diagram**: 4 personas (developer, domain user, end user, auditor) with their questions. *Exam question: match stakeholder to their question.*

### Important Keywords
- **Explainability (XAI)** — making model decisions understandable to humans.
- **Trust** — main outcome of explainability for users/doctors.
- **Compliance** — legal reason to require explainability.
- **ROI** — business factor traded against compliance.
- **Stakeholders** — developer, domain user, end user, auditor.
- **High stakes** — situations where explainability is essential.

### Memory Tips
- **"JIT-CD"**: **J**ustify, **I**mprove, **T**rust, **C**ontrol, **D**iscover — the 5 explain-to's.
- Stakeholders = **DDEA**: **D**eveloper (debug), **D**omain user (trust), **E**nd user (why me?), **A**uditor (fairness).
- High-stakes fields = "**Money, Body, Law, War**" (finance, medicine, legal/justice, military/security).

---

## Topic 2: Feature Importance (Concept)

### Definition
Feature Importance Analysis highlights the **most influential features** in a model. Often the objective is to **understand what drives a target variable, not to predict it**.

### Simple Example
From hundreds of financial indicators (transaction history, credit mix, payment patterns), find which ones **truly drive loan defaults**.

### Why is it Important?
It shifts the goal from prediction to **understanding associations** between inputs and the target — the basis of insight-driven data mining.

### Key Points
- Sometimes we care more about **understanding associations** than predicting.
- Use-cases from the lecture:
  - Which **financial indicators** drive **loan defaults**
  - Which **behavioral signals** (clicks, scroll depth, session time) drive **conversion or churn**
  - Which **network traffic features** indicate a **cyberattack**
- **Two-step workflow** (disease-risk use-case):
  1. **Train a model** on multi-dimensional health data (EMR, claims) → model learns risk factors → predictions (e.g., hospitalization risk, fast CKD progressors, Long-COVID risk).
  2. **Explain the model** → extract the risk factors (X1…X6) the model used.

### Common Mistakes
- ❌ Thinking the model itself is the end goal — here the model is a **tool to reveal drivers**.
- ❌ Interpreting "important feature" as "causal feature" (see causality topic).

### Comparison

| Prediction Goal | Understanding Goal |
|---|---|
| Output: accurate label/value | Output: ranked influential features |
| "Who will churn?" | "**Why** do people churn?" |
| Model performance is the metric | Feature ranking is the insight |

### Diagram / Image Explanation
- **Step 1 / Step 2 pipeline slides**: Input data → ML Model → Predictions; then the model is queried to expose X1…X6, arrow thickness = importance toward "Risk Patients". *Exam question: order the pipeline: train model → explain model → extract risk factors.*

### Important Keywords
- **Feature importance** — measure of a feature's influence in a model.
- **Target variable** — the outcome being driven/predicted.
- **Associations** — relationships between inputs and target (not causation).
- **EMR** — electronic medical records (example health data).

### Memory Tips
- Motto of the slide: "**Understand the drivers, not just predict**."
- Pipeline: **T**rain → **E**xplain ("**TE**ll me why").

---

## Topic 3: Global vs Local Explainability

### Definition
**Local explainability** explains **one single prediction** ("Why did the model predict this for this instance?"). **Global explainability** describes the model's **overall behavior** across all data ("What features are important for the model in general?").

### Simple Example
- Local: "Why was **my** loan denied?" → SHAP force plot for that one customer.
- Global: "Which features does the bank's model rely on overall?" → feature importance ranking.

### Why is it Important?
Choosing the right scope determines the right tool: end-user communication needs **local**; auditing, feature selection, and monitoring need **global**.

### Key Points
- **Local**: explains a single prediction; shows how each feature contributed to a **specific output**.
  - Useful for: case-by-case analysis (loan approval), **debugging unexpected predictions**, communicating results to end users.
  - Examples: **SHAP Force Plot** (one prediction), **LIME** (linear approximation around one point), **Counterfactual explanations** ("What if income was higher?").
- **Global**: understands overall behavior; helps **build trust** and **identify biases**.
  - Useful for: **feature selection**, **model monitoring**, **documentation & auditing**.
  - Examples: **feature importance rankings**, **SHAP Summary Plot**, **Global Surrogate Models**.

### Common Mistakes
- ❌ Using a global ranking to explain one customer's decision (wrong scope).
- ❌ Forgetting SHAP works at **both** levels (force plot = local; summary plot = global).

### Comparison

| | **Local** | **Global** |
|---|---|---|
| Goal | Explain **a single prediction** | Model's **overall behavior** |
| Question | "Why *this* prediction for *this* instance?" | "What features matter for the model?" |
| Useful for | Case analysis, debugging, end-user communication | Feature selection, monitoring, auditing |
| Examples | SHAP force plot, LIME, counterfactuals | Feature importance ranking, SHAP summary plot, global surrogate models |

### Important Keywords
- **LIME** — Local linear approximation around one data point.
- **Counterfactual explanation** — "What if income was higher?" — smallest change that flips the prediction.
- **Global surrogate model** — simple model trained to mimic a black box globally.
- **SHAP force plot** — local; **SHAP summary plot** — global.

### Memory Tips
- **L**ocal = **L**IME = one **L**ine of data.
- **G**lobal = **G**eneral behavior = whole dataset.
- Counterfactual = "**What if…?**"

---

## Topic 4: Inherent vs Post-hoc Explainability

### Definition
**Inherently explainable** ("glass box") models are transparent by design — you can directly read their logic. **Post-hoc explainability** applies **separate techniques** (SHAP, LIME) to interpret complex "black box" models **after** training.

### Simple Example
A logistic regression's weights tell you directly that "tumor radius" increases cancer probability (glass box). An XGBoost model needs SHAP applied afterward to reveal the same insight (black box + post-hoc).

### Why is it Important?
This distinction decides the strategy: use a transparent model and accept possibly lower accuracy, or keep a high-performing black box and add post-hoc tools.

### Key Points
- **Glass box (inherent)**: transparent, self-explanatory; inspect logic/coefficients directly.
  - Examples: **Linear Regression, Logistic Regression, (shallow) Decision Trees, Rule-based models**.
  - May **sacrifice accuracy** on complex tasks.
- **Black box (post-hoc)**: complex/opaque; need interpretation tools.
  - Examples: **Neural Networks, Random Forests, Gradient Boosting (XGBoost), SVMs in high dimensions**.
  - Post-hoc tools: **SHAP, LIME**, etc. → **maintains performance while offering insights**.
- **Decision Tree explainability**: the **order of the splits** reveals feature importance (first split = most important feature). Example tree: Study Hours → Social Media time / bedtime → Sport time → Pass/Fail.
- **Logistic Regression explainability**: the **learnt weights (w)** reveal the model's logic. Pipeline: inputs xᵢ × weights wᵢ → weighted sum z = wᵀx + b → **sigmoid** σ(z) → probability ŷ → **quantizer/threshold** → binary {0,1}. The threshold is adjustable.
- **Why Random Forest is a black box**: each individual tree is explainable, but the **collection** (Bagging + Random Subspace Method + Decision Tree learning) makes the combined decision hard to explain → use post-hoc methods.

### Common Mistakes
- ❌ "Decision trees are always explainable" — only **shallow** trees; and a **forest** of trees is a black box.
- ❌ Thinking post-hoc methods change the model — they only **interpret** it.
- ❌ Confusing sigmoid output (probability) with the final class — the **threshold/quantizer** converts probability → {0,1}.

### Comparison

| | **Inherent (Glass box)** | **Post-hoc (Black box)** |
|---|---|---|
| Transparency | Self-explanatory | Opaque, needs tools |
| How to interpret | Inspect coefficients/splits directly | Apply SHAP, LIME, permutation importance |
| Examples | Linear/Logistic Regression, shallow DT, rule-based | NN, Random Forest, XGBoost, SVM (high-dim) |
| Accuracy | May sacrifice accuracy | Maintains performance |

### Formula Explanation — Logistic Regression
- **z = Σᵢ (wᵢxᵢ) + b = wᵀx + b**
  - **xᵢ** = feature values (e.g., age, radius_mean, texture_mean), **wᵢ** = learnt weights, **b** = bias.
  - Meaning: weighted sum of evidence.
- **ŷ = σ(z) = 1 / (1 + e^(−(wᵀx+b)))**
  - Sigmoid squashes z into a probability (0–1).
  - Intuition: bigger positive weight × feature → higher probability of class 1.
  - When to use: binary classification with interpretable weights (lecture example: **breast cancer prediction from pathological image features**).
  - Common mistake: forgetting the **threshold step** that turns probability into a binary decision; assuming weights of unscaled features are directly comparable.

### Diagram / Image Explanation
1. **Glass jar vs box with eyes** (slide 15): glass box = see inside (inherent); black box = opaque (post-hoc needed).
2. **Student pass/fail decision tree** (slide 16): root = Study Hours (<3 / >3), then Social Media time / bedtime, then Sport time. Shows **split order = importance order**. *Exam question: "Which feature is most important in this tree?" → Study Hours (root).*
3. **Logistic regression network diagram** (slide 17): inputs → weights → Σ → sigmoid → ŷ → quantizer → {0,1}. *Exam question: name the components in order.*
4. **Random Forest diagram** (slide 18): table of fruit data → Bagging + Random Subspace + DT learning → Trees 1…N. Shows why the ensemble is a black box.

### Important Keywords
- **Glass box** — transparent, inherently interpretable model.
- **Black box** — opaque model needing post-hoc tools.
- **Post-hoc** — explanation applied *after* training.
- **Sigmoid** — activation squashing z to (0,1).
- **Quantizer / threshold** — converts probability to binary prediction; adjustable.
- **Bagging + Random Subspace Method** — what makes a Random Forest.
- **Split order** — importance ranking in a decision tree.

### Memory Tips
- **Glass** = you can **see through** it; **Black** = need a **flashlight** (SHAP/LIME).
- Logistic regression flow: "**Sum → Squash → Snap**" (weighted sum → sigmoid → threshold).
- "One tree = glass, a forest = dark" (you can't see through a forest).

---

## Topic 5: Permutation Feature Importance (PFI)

### Definition
Permutation Feature Importance measures how much a model's **performance drops when a single feature's values are randomly shuffled**. Shuffling **breaks the relationship** between the feature and the target — a big drop means the feature was important.

### Simple Example
Titanic survival model: shuffle the "sex" column → F1 score falls sharply → sex is the most important feature. Shuffle "embarked" → almost no change → unimportant.

### Why is it Important?
It is a **model-agnostic, post-hoc** method that works on any trained model — the simplest way to rank features of a black box.

### Key Points
- **Underlying hypothesis**: if a feature significantly contributes to prediction, permuting it should cause a **proportional decrease** in performance, because permutation **breaks the feature–target relationship**.
- **Algorithm (7 steps)**:
  1. **Train** a model on training data.
  2. **Measure baseline performance** on a validation/test (**held-out**) set with a **scoring function** (accuracy, R², F1) → score **s** (e.g., 90%).
  3. **Pick one feature** (e.g., Age).
  4. **Permute (shuffle)** that feature's values across all rows — all other features unchanged.
  5. **Evaluate again** on the modified data (e.g., 78%).
  6. **Importance = Baseline − New** (90% − 78% = 12%).
  7. **Repeat for all features**; rank by drop (higher drop = more important).
- **Steps 4–6 are repeated K times** to account for **random variation**; scores are averaged.
- Permutation is done **one feature at a time**.
- Titanic example (F1 score): importance order → **sex > pclass > age > fare > sibsp > parch > embarked** (embarked/parch slightly negative).
- **Advantages**: **model-agnostic**; easy to explain and implement.
- **Limitations**:
  - **Correlated features can mask each other's importance**
  - **Computationally expensive** (many features)
  - Can be **unstable** on small datasets or high-variance models

### Common Mistakes
- ❌ Evaluating on training data — baseline must be a **held-out** set.
- ❌ **Retraining** the model after shuffling — the same trained model is reused.
- ❌ Interpreting **negative importance** as "useless feature" — see quiz: it suggests **noise, overfitting, redundancy, leakage, or a harmful learned dependency** (model does *better* when the feature's information is destroyed).
- ❌ Shuffling several features at once — it's one feature at a time.
- ❌ Trusting a single shuffle — repeat K times and average.

### Comparison: PFI vs SHAP

| | **Permutation Importance** | **SHAP** |
|---|---|---|
| Basis | Performance drop after shuffling | Game-theoretic marginal contributions |
| Scope | Global ranking | Local + global |
| Cost | Expensive with many features | Very expensive (all subsets; approximated) |
| Output | One score per feature | One value per feature **per prediction** |
| Weakness | Correlated features mask each other | Not causal; computational cost |

### Formula Explanation
**iⱼ = s − (1/K) Σₖ₌₁ᴷ sₖ,ⱼ** *(slide displays iⱼ = s − Σₖ sₖ,ⱼ with the average taken over repetitions)*
- **iⱼ** = importance of feature j
- **s** = baseline score before permutation (held-out set)
- **sₖ,ⱼ** = score after the k-th shuffle of feature j
- **K** = number of repetitions (accounts for random variation)
- Intuition: importance = baseline minus average shuffled performance. Large positive → important. ~0 → irrelevant. Negative → suspicious (noise/leakage/redundancy).
- When to use: any trained model + a held-out set + a scoring function.
- Common mistake: forgetting the average over K, or using training data for s.

### Diagram / Image Explanation
1. **7-step algorithm infographic** (slides 20 & 22 — shown **twice**, so highly exam-relevant): Train → Baseline → Pick feature → Shuffle → Re-evaluate → Compute drop → Repeat for all → Rank. Highlighted note: **"Steps 4–6 are repeated K times to account for random variation."** *Exam question: order the steps / what does shuffling break?*
2. **Titanic PFI bar chart** (slide 24): x-axis = mean decrease in F1; sex ≈ 0.20 top bar. *Exam question: which feature is most important and why can embarked be slightly negative?*

### Important Keywords
- **Permutation / shuffling** — randomly reordering one feature's values.
- **Baseline performance (s)** — score before shuffling on held-out data.
- **Held-out set** — validation/test data used for scoring.
- **Scoring function** — accuracy, R², F1.
- **Model-agnostic** — works for any model.
- **K repetitions** — averaging to handle random variation.
- **Negative importance** — performance improves after shuffling → noise/overfitting/redundancy/leakage.

### Memory Tips
- "**Shuffle → Suffer**": if the model suffers when a feature is shuffled, the feature matters.
- Formula in words: **Importance = Before − After**.
- Negative importance causes = "**NORL**": **N**oise, **O**verfitting, **R**edundancy, **L**eakage.

---

## Topic 6: Shapley Values / SHAP

### Definition
Shapley values come from **cooperative game theory**: a way to **fairly distribute a total payout among players** who collaborated. In ML, **players = features** and **payout = model prediction**; each feature's SHAP value is its **fair share of credit** for a specific prediction.

### Simple Example
Titanic: what is the importance of "fare" for predicting survival? Compute how much adding "fare" changes the prediction for every possible subset of other features (∅, {age}, {sex}, {age,sex,sibsp}…), then take a **weighted average** of these marginal contributions → φ_fare = 0.1667.

### Why is it Important?
SHAP is the standard **post-hoc, fair** attribution method: it gives per-prediction (local) explanations that also aggregate into global importance.

### Key Points
- Invented by **Lloyd Shapley** (Nobel Memorial Prize in Economic Sciences, 2012).
- In ML it is called **SHAP = SHapley Additive exPlanations**.
- **Goal**: quantify **each feature's contribution to a specific prediction**.
- **Method**: for feature i, consider **every possible subset S of features excluding i**; compute the **marginal contribution** **f(S ∪ {i}) − f(S)**; then **average over all possible combinations, weighted** by how often each subset appears in orderings.
- Result: features that **consistently add value** get high Shapley scores; features that only sometimes help get proportionally less credit.
- **Why weighting is needed**: a feature's contribution **depends on what information is already available** (house-price example with Age, Income, Gender). Averaging over **all orderings/permutations** makes credit assignment **fair**.
- Weighting table (3 features → 3! = 6 permutations): Gender's marginal contribution differs depending on whether it comes first ({} before it), second ({Age} or {Income} before), or last ({Age, Income} before).
- **SHAP plots**:
  - **Summary bar plot (global)**: mean(|SHAP value|) per feature = average impact magnitude → Titanic: sex > class > fare > age > embarked.
  - **Beeswarm plot (local view of all points)**: each dot = one passenger; x = SHAP value (impact vs baseline), color = feature value (red high / blue low). Answers "how much does each feature, for each data point, contribute compared to a baseline?"
- ⚠️ **SHAP values do NOT reflect causality.** Predictive models don't deliver causal insights unless explicitly modified and trained to do so. (Recommended reading: *"Be careful when interpreting predictive models in search of causal insights."*)

### Common Mistakes
- ❌ **The #1 exam trap**: interpreting high SHAP value as causal effect ("changing fare would change survival") — SHAP shows **predictive** importance only.
- ❌ Forgetting SHAP is computed **per prediction** (local), then aggregated for global views.
- ❌ Ignoring the weighting — a simple unweighted average over subsets is not the Shapley value.
- ❌ Confusing SHAP summary plot (global) with force plot (local).

### Comparison: Predictive Importance vs Causality (from quiz slide)

| **Predictive importance** | **Causality / Actionability** |
|---|---|
| "The model **uses** this." | "**Changing** this changes the outcome." |
| SHAP, permutation importance | Interventions, causal inference |
| Fare is important for prediction | Equalizing fares wouldn't change survival |

### Formula Explanation
**φᵢ(v) = Σ_{S ⊆ N\{i}} [ |S|! (n − |S| − 1)! / n! ] · (v(S ∪ {i}) − v(S))**
- **φᵢ** = Shapley value of feature i
- **N** = set of all n features; **S** = a subset **not containing i**
- **v(S)** = model prediction using only subset S
- **v(S ∪ {i}) − v(S)** = **marginal contribution** of i to subset S
- **|S|!(n−|S|−1)!/n!** = **combinatorial weighting term** — "gives **fairness** by averaging over **all possible orderings** of features" (quoted from the slide)
- Intuition: play the "add features one at a time" game in every possible order; average what feature i adds each time.
- Lecture example: fare's Shapley value for Titanic = **0.1667** (marginal contributions 0.2, 0.2, 0.2, …, 0.10 across subsets).
- Common mistakes: forgetting S excludes i; forgetting the weights; expecting causal meaning.

### Diagram / Image Explanation
1. **Marginal contribution table** (slide 28): subsets ∅, {age}, {sex}, {age,sex,sibsp} with v(S), v(S∪{fare}), and differences → feeds the formula → 0.1667. *Exam question: compute a marginal contribution from a row.*
2. **Permutation/ordering table** (slide 30): 6 orderings of Gender/Age/Income and the features "before Gender" in each. Shows why weights = fairness over orderings.
3. **SHAP summary bar plot vs beeswarm** (slide 31): left = global mean|SHAP|; right = per-point SHAP values with feature-value coloring. *Exam question: which plot is global vs local? What does red mean? (high feature value).*

### Important Keywords
- **Cooperative game theory** — origin of Shapley values.
- **Players = features, payout = prediction** — the ML mapping.
- **Marginal contribution** — f(S∪{i}) − f(S).
- **SHAP** — SHapley Additive exPlanations.
- **Combinatorial weight** — fairness term averaging all orderings.
- **Summary plot** — global bar plot of mean |SHAP|.
- **Beeswarm plot** — per-instance SHAP values, colored by feature value.
- **Baseline** — reference prediction that SHAP contributions are measured against.

### Memory Tips
- **"Features are players, the prediction is the prize."**
- Marginal contribution = "**With me minus without me**."
- Fairness trick: "**Try every order, then average**."
- SHAP warning: "**SHAP shows what the model thinks, not what the world does.**"

---

## Topic 7: Causality Analysis

### Definition
Causal analysis studies **cause-and-effect relationships** instead of mere correlations. It identifies what actually *causes* outcomes, which is essential for **decision-making, policy setting, and interventions**.

### Simple Example
A model finds umbrella owners have more traffic accidents. The real cause is **rain** (the confounder): rain increases both umbrella use and accident risk. **Giving everyone an umbrella will NOT reduce accidents.**

### Why is it Important?
Decisions and interventions require knowing **what changes the outcome**, not what merely predicts it — acting on non-causal features wastes resources or causes harm.

### Key Points
- **Key concepts** (slide 35):
  - **Causal Inference**: determining if one variable causes another.
  - **Counterfactual Analysis**: "what if" questions under different scenarios.
  - **Interventions**: simulating/applying actions to see causal impact.
- Use cases: **Healthcare** (effect of a treatment on outcomes), **Economics** (impact of policy changes).
- **"Mary" breast-cancer timeline example** (slides 36–37): Diagnosis (X) → Treatment → Death (Y). "A long survival time **may be** because of treatment!" But: what if we start aggressive treatment based on predicted outcome? What if unrelated **comorbidities** caused death? What if lifestyle became unhealthy after treatment? → **Decision-making requires understanding interventions, not just model reasoning.**
- Other causal questions: Does smoking cause breast cancer? Does obesity cause low response to drug A? Pr(lung cancer | smoker) vs Pr(lung cancer | nonsmoker).
- **Why prediction/correlation is not enough** (slide 38):
  - Explainability reveals **what influenced the model**, not **what changes the outcome**.
  - Explainability of predictive models is like **correlation**; it may attribute importance to **confounders** — hidden variables that create misleading correlations.
  - **Causality accounts for confounders** to separate true causes from misleading correlations.
- **Titanic causal examples**:
  - **Sex vs survival**: model says sex is most important; but being female did not biologically cause survival — the cause was the **"women and children first" evacuation policy** (confounder, **not recorded in the dataset**).
  - **Fare vs survival**: paying more didn't directly save you; high fare ↔ **higher passenger class** → cabins **closer to lifeboats**, **priority access**. Confounders: **passenger class, cabin location, crew assistance**.

### Common Mistakes
- ❌ "Important feature ⇒ cause" — the central misconception this lecture attacks.
- ❌ Believing intervening on a predictive feature changes the outcome (same-fare quiz: outcomes unchanged, model gets less accurate).
- ❌ Forgetting confounders can be **unrecorded** in the dataset (evacuation policy).

### Comparison: Explainability vs Causal Analysis (Summary slide 42)

| | **Explainability Analysis** | **Causal Analysis** |
|---|---|---|
| Focus | Making model decisions understandable to humans | Cause-and-effect relationships, beyond correlation |
| Method | Feature importance, attention maps, decision trees | Causal inference: domain knowledge, experiments, observational data |
| Question answered | "**What factors influenced this particular decision?**" | "**What would happen if we intervened or changed a specific factor?**" |
| Analogy | Like correlation | Accounts for confounders |

### Formula Explanation
No formal formula; the notation **Pr(lung cancer | smoker) vs Pr(lung cancer | nonsmoker)** shows a **conditional-probability comparison** — a correlational quantity that motivates but does not prove causation.

### Diagram / Image Explanation
1. **Mary timeline** (slides 36–37): X = Diagnosis, Treatment marks in between, Y = Death on a time axis. Point: survival time may be due to treatment — or comorbidities or lifestyle. *Exam question: why can't the predictive model alone tell us whether treatment works?*
2. **Umbrella example** (slide 39): model explanation vs causal reality with rain as confounder. *Exam question: identify the confounder; will intervening on umbrellas help? (No).*

### Important Keywords
- **Causal inference** — determining whether one variable causes another.
- **Counterfactual analysis** — assessing "what if" scenarios.
- **Intervention** — changing a variable to observe its causal effect.
- **Confounder** — hidden variable creating misleading correlations (rain; evacuation policy; passenger class).
- **Correlation ≠ causation** — predictive importance is correlational.
- **Actionability** — knowing that changing X changes Y.

### Memory Tips
- **"Rain, not umbrellas"** — remember the confounder story.
- Explainability = "**What did the model look at?**"; Causality = "**What actually moves the world?**"
- Titanic double trap: **Sex → policy**, **Fare → class/cabin/crew** (feature → real confounder).
- Causality triad: "**I-C-C**": **I**nference, **C**ounterfactuals, interventions ("If I **C**hange it, does it **C**hange?").

---

# 3. Exam Practice (Topic-wise)

## Topic 1: Explainability Introduction

**MCQs**
1. Explainability is primarily about making model decisions:
   a) Faster b) **Understandable to humans** ✅ c) More accurate d) Cheaper
2. In which case does explainability matter LEAST?
   a) Cancer diagnosis b) Loan approval c) **Product recommendation** ✅ d) Criminal justice
3. Which stakeholder asks "Are minority groups disproportionately denied loans?"
   a) Developer b) End user c) **Auditor** ✅ d) Domain user
4. Two key business factors when selecting an ML model are:
   a) Speed & memory b) **ROI & compliance** ✅ c) Data size & GPUs d) Accuracy & recall

**True/False**
5. Explainability answers only *what* the model decided. → **False** (also *why* and *how*).
6. Deep neural networks are typically both highly accurate and highly interpretable. → **False** (trade-off).
7. Explainability is essential when stakes are high legally, ethically, or socially. → **True**.

**Fill in the blanks**
8. Models that are highly accurate, such as deep neural networks, often lack ______. → **transparency**.
9. Explain to justify, improve, build trust, ______, and discover. → **control**.

**One-word / Two-word answers**
10. Which stakeholder asks "Why was my loan denied?" → **End user**.
11. Two factors: profit maximization goal + legal standards → **ROI, compliance**.

**Short answer**
12. Give two conditions when explainability "matters more." → Decisions affect people's lives; regulations require transparency (also: trust/accountability critical).

**Scenario**
13. A hospital AI recommends a high-risk treatment with no reasoning. Name two problems. → Doctor/patient can't understand the *why*; trust is compromised and the doctor can't justify/validate the treatment.

**Matching**
14. Match: Developer / End user / Auditor / Domain user ↔ (i) "How do I correct an error?" (ii) "Why was my loan denied?" (iii) "Does the model exclude race?" (iv) "Can I trust this model?"
    → Developer–i, End user–ii, Auditor–iii, Domain user–iv.

## Topic 2: Feature Importance Concept

**MCQ**
1. "Often the objective is to understand what ______ a target variable, not to predict it." → **drives**.
2. Feature importance analysis highlights:
   a) Data errors b) **Most influential features** ✅ c) Best hyperparameters d) Fastest model

**Pipeline ordering**
3. Order: (a) explain the model (b) train a model (c) extract risk factors → **b → a → c**.

**Short answer**
4. Give one lecture use-case of feature importance. → Financial indicators driving loan defaults / behavioral signals driving churn / network traffic features indicating a cyberattack.

**True/False**
5. In the disease-risk use-case, the model is trained mainly to deploy it for prediction. → **False** (trained to understand drivers of risk).

## Topic 3: Global vs Local

**MCQs**
1. "Why did the model predict this for this instance?" is answered by:
   a) Global b) **Local** ✅ c) Causal d) Surrogate
2. Which is a LOCAL method? a) SHAP summary plot b) Feature importance ranking c) **LIME** ✅ d) Global surrogate model
3. Feature selection and model monitoring are use-cases of:
   a) Local b) **Global** ✅

**Fill in blank**
4. "What if income was higher?" is a ______ explanation. → **counterfactual**.

**Visualization selection**
5. You must show an end user why *their* loan was denied. Which plot? → **SHAP force plot** (local).
6. You must audit which features the model relies on overall. → **SHAP summary plot / feature importance ranking** (global).

**Two-word answer**
7. Simple model trained to mimic a black box globally → **surrogate model** (global surrogate).

## Topic 4: Inherent vs Post-hoc

**MCQs**
1. Which is NOT inherently explainable? a) Linear regression b) Rule-based model c) **Random Forest** ✅ d) Shallow decision tree
2. Post-hoc explainability tools include: a) **SHAP and LIME** ✅ b) PCA and t-SNE c) Bagging and boosting d) K-means
3. In a decision tree, feature importance can be inferred from:
   a) Leaf colors b) **Order of the splits** ✅ c) Number of leaves d) Tree depth only
4. In logistic regression, model logic is read from:
   a) Activations b) **Learnt weights** ✅ c) Data size d) Threshold only

**True/False**
5. A Random Forest is a black box even though each tree is explainable. → **True**.
6. Post-hoc methods sacrifice model performance. → **False** (they maintain performance while offering insights).
7. The classification threshold in logistic regression is fixed and cannot be changed. → **False** (adjustable).

**Fill in blanks**
8. Glass box models are transparent and ______. → **self-explanatory**.
9. Random Forest = ______ + Random Subspace Method + Decision Tree learning. → **Bagging**.

**Formula interpretation**
10. In ŷ = σ(wᵀx + b), what does σ do? → Squashes the weighted sum into a probability between 0 and 1 (sigmoid).

**Diagram interpretation**
11. In the pass/fail tree, which feature is most important? → **Study Hours** (root split).

**Algorithm selection**
12. You need high accuracy on a complex task AND explanations. Strategy? → Use a black-box model (e.g., XGBoost) + **post-hoc tools (SHAP/LIME)**.

## Topic 5: Permutation Feature Importance

**MCQs**
1. PFI measures importance as: a) Weight size b) **Performance drop after shuffling a feature** ✅ c) Correlation with target d) Number of splits
2. Why is shuffling used? a) Data augmentation b) **It breaks the feature–target relationship** ✅ c) Removes outliers d) Balances classes
3. Steps 4–6 are repeated K times to: a) Improve accuracy b) **Account for random variation** ✅ c) Reduce overfitting d) Tune hyperparameters
4. Baseline performance must be computed on: a) Training set b) **Held-out (validation/test) set** ✅ c) Shuffled data d) Any subset
5. A limitation of PFI: a) Model-specific b) **Correlated features can mask each other's importance** ✅ c) Needs retraining d) Only works for trees

**True/False**
6. PFI requires retraining the model for each feature. → **False** (same trained model, re-evaluate).
7. PFI is model-agnostic. → **True**.
8. Features are shuffled all at once. → **False** (one feature at a time).
9. Negative permutation importance proves the feature is useless. → **False** (suggests noise, overfitting, redundancy, leakage, or harmful learned dependency).

**Fill in blanks**
10. Importance = ______ − New performance. → **Baseline**.
11. PFI can be ______ on small datasets or high-variance models. → **unstable**.

**Formula interpretation**
12. In iⱼ = s − Σₖ sₖ,ⱼ (averaged over K), what is s? → Baseline score on held-out data before permutation; sₖ,ⱼ = score after the k-th shuffle of feature j.

**Scenario**
13. Baseline accuracy 90%; after shuffling "Age", accuracy 78%. Importance of Age? → **12%** (0.12).
14. Shuffling "Region" gives 90.5% (better than baseline). Interpretation? → Negative importance → the feature's information may be noise/redundant/leaky/harmful; not necessarily useless.

**Pipeline ordering**
15. Order the steps: (a) shuffle feature (b) train model (c) rank features (d) baseline score (e) re-evaluate (f) compute drop (g) repeat for all features → **b → d → a → e → f → g → c**.

**Diagram interpretation**
16. Titanic PFI chart: most important feature? → **sex** (~0.20 mean F1 decrease), followed by **pclass**.

## Topic 6: SHAP / Shapley Values

**MCQs**
1. Shapley values originate from: a) Statistics b) **Cooperative game theory** ✅ c) Deep learning d) Bayesian inference
2. In ML, "players" and "payout" map to: a) Models, accuracy b) **Features, model prediction** ✅ c) Data points, loss d) Trees, votes
3. SHAP stands for: → **SHapley Additive exPlanations**.
4. The marginal contribution of feature i to subset S is:
   a) f(S)−f(∅) b) **f(S∪{i}) − f(S)** ✅ c) f({i}) d) f(N)−f(S)
5. The combinatorial term |S|!(n−|S|−1)!/n! provides:
   a) Speed b) **Fairness by averaging over all possible orderings** ✅ c) Regularization d) Normalization to 1
6. SHAP values reflect: a) Causal effects b) **Predictive importance** ✅ c) Data quality d) Feature correlation only

**Quiz-style scenario (from the lecture — very likely on exam)**
7. A SHAP report shows Fare is one of the most important features for Titanic survival. Best interpretation?
   → **(b) Fare is useful for prediction, but it may simply capture information about other factors that actually influenced survival.** ✅
8. If every passenger paid the same fare but everything else stayed the same:
   → **(b) The model would become less accurate, but actual survival outcomes would likely remain unchanged.** ✅

**True/False**
9. Lloyd Shapley won the Nobel Memorial Prize in Economic Sciences. → **True** (2012).
10. Features that consistently add value get higher Shapley scores. → **True**.
11. A feature's contribution is independent of which features are already present. → **False** (depends on what information is already available — hence weighting).
12. SHAP values reflect causality. → **False** (explicit lecture warning).

**Fill in blank**
13. In the beeswarm plot, red dots represent ______ feature values. → **high**.
14. The SHAP summary bar plot shows mean(______) per feature. → **|SHAP value|**.

**Visualization selection**
15. To see global importance AND direction of effect per data point → **beeswarm plot**; global magnitude only → **summary bar plot**.

**Formula/table interpretation**
16. Given v({age}) = 0.4 and v({age, fare}) = 0.6, marginal contribution of fare to {age}? → **0.2**.
17. The lecture's computed Shapley value for fare was → **0.1667**.

**Matching**
18. Match: Force plot / Summary plot / Beeswarm ↔ (i) one prediction (local), (ii) global bar ranking, (iii) all points with value coloring → Force–i, Summary–ii, Beeswarm–iii.

## Topic 7: Causality Analysis

**MCQs**
1. Causal analysis focuses on: a) Correlations b) **Cause-and-effect relationships** ✅ c) Model accuracy d) Feature scaling
2. Asking "what if" questions under different scenarios is: a) Intervention b) **Counterfactual analysis** ✅ c) Inference d) Simulation
3. A hidden variable creating misleading correlations is a: → **Confounder**.
4. In the umbrella example, the confounder is: → **Rain**.
5. In Titanic sex-vs-survival, the true underlying cause was:
   a) Biology b) **"Women and children first" evacuation policy** ✅ c) Fare d) Age
6. "What would happen if we intervened?" is answered by:
   a) Explainability b) **Causal analysis** ✅ c) LIME d) PFI

**True/False**
7. Giving everyone an umbrella will reduce traffic accidents. → **False** (rain is the confounder).
8. Explainability reveals what influenced the model, not what changes the outcome. → **True**.
9. The evacuation-policy confounder was recorded in the Titanic dataset. → **False** (not directly recorded).
10. Causality accounts for confounders to separate true causes from misleading correlations. → **True**.

**Fill in blanks**
11. Explainability of predictive models is like ______. → **correlations**.
12. Fare confounders in Titanic: passenger class, ______, crew assistance. → **cabin location**.
13. Predictive importance = "the model uses this"; actionability/causality = "changing this changes the ______." → **outcome**.

**One/Two-word answers**
14. Determining if one variable causes another → **Causal inference**.
15. Simulating actions to see causal impact → **Interventions**.

**Scenario**
16. Mary survives long after breast cancer treatment. Can we conclude treatment caused survival? → No — comorbidities or lifestyle changes could be responsible; we'd need causal analysis of the intervention, not just prediction.
17. A churn model ranks "number of support calls" highest. Should the company block support calls? → No — likely a symptom/correlate of dissatisfaction (confounder), not the cause of churn.

**Analysis type identification**
18. "Which features did the model rely on for this loan denial?" → **Explainability (local)**.
19. "Would lowering interest rates reduce defaults?" → **Causal analysis (intervention)**.
20. Pr(lung cancer | smoker) vs Pr(lung cancer | nonsmoker) is what kind of quantity? → **Conditional probability / correlational comparison** (motivates causal questions).

---

# 4. Flashcards

| Question | Answer |
|---|---|
| What is explainability? | Making a model's decisions understandable to humans — the *why* and *how*, not just the *what* |
| The 5 "explain to…" purposes? | Justify, Improve, Build trust, Control, Discover |
| When does explainability matter more? | High stakes: lives affected, regulations, trust/accountability |
| When does it matter less? | Low-risk decisions; speed/accuracy over interpretability |
| Trade-off: decision tree vs neural network? | DT: interpretable, lower accuracy; NN: accurate, opaque |
| Two business factors for model selection? | ROI and compliance |
| What does the auditor persona check? | Fairness — no prohibited factors (race), no disproportionate denial |
| Goal of feature importance analysis? | Highlight the most influential features / understand what drives the target |
| Local explainability? | Explains a single prediction for one instance |
| Global explainability? | Model's overall behavior across all data |
| Three local methods? | SHAP force plot, LIME, counterfactual explanations |
| Three global methods? | Feature importance ranking, SHAP summary plot, global surrogate model |
| What is LIME? | Linear approximation around one point (local) |
| What is a counterfactual explanation? | "What if income was higher?" — a what-if change flipping the prediction |
| Glass box models? | Transparent, self-explanatory: linear/logistic regression, shallow DT, rule-based |
| Black box models? | NN, Random Forest, Gradient Boosting (XGBoost), high-dim SVM |
| Post-hoc explainability? | Separate tools (SHAP, LIME) applied after training to interpret black boxes |
| How to read importance from a decision tree? | Order of the splits (root = most important) |
| How to read logic from logistic regression? | Learnt weights |
| Logistic regression pipeline? | Weighted sum z = wᵀx+b → sigmoid σ(z) → threshold/quantizer → {0,1} |
| Why is a Random Forest a black box? | Each tree explainable, but the collection (bagging + random subspace) is not |
| Permutation feature importance? | Performance drop when one feature's values are randomly shuffled |
| PFI underlying hypothesis? | Shuffling breaks the feature–target relationship → important features cause proportional performance drop |
| PFI importance formula? | Importance = Baseline − New (averaged over K repeats): iⱼ = s − Σₖ sₖ,ⱼ |
| Why repeat K times? | To account for random variation |
| Where is baseline measured? | On a held-out (validation/test) set with a scoring function (accuracy, R², F1) |
| PFI advantages? | Model-agnostic; easy to explain and implement |
| PFI limitations? | Correlated features mask each other; computationally expensive; unstable on small data |
| Negative permutation importance means? | Model performs better when feature destroyed → noise, overfitting, redundancy, leakage, harmful dependency |
| Titanic PFI top feature? | Sex (then pclass) |
| Origin of Shapley values? | Cooperative game theory — fair payout distribution (Lloyd Shapley, Nobel 2012) |
| ML mapping of Shapley? | Players = features; Payout = model prediction |
| SHAP acronym? | SHapley Additive exPlanations |
| Marginal contribution? | f(S ∪ {i}) − f(S) |
| Role of \|S\|!(n−\|S\|−1)!/n!? | Fairness — weights averaging over all possible orderings of features |
| Why weighting needed? | A feature's contribution depends on what information is already available |
| SHAP summary plot shows? | Global: mean(\|SHAP value\|) per feature |
| Beeswarm plot shows? | Per-data-point SHAP values; color = feature value (red high, blue low) |
| Do SHAP values reflect causality? | **NO** — predictive importance only |
| Predictive importance vs actionability? | "Model uses this" vs "Changing this changes the outcome" |
| Causal analysis? | Understanding cause-and-effect, beyond correlation, for decisions/interventions |
| Three key causal concepts? | Causal inference, counterfactual analysis, interventions |
| Confounder? | Hidden variable creating misleading correlations |
| Umbrella example confounder? | Rain (umbrellas don't cause accidents) |
| Titanic sex confounder? | "Women and children first" evacuation policy (not in dataset) |
| Titanic fare confounders? | Passenger class, cabin location, crew assistance |
| Question explainability answers? | "What factors influenced this particular decision?" |
| Question causal analysis answers? | "What would happen if we intervened or changed a specific factor?" |

---

# 5. Rapid Revision

**Explainability intro**
- Explainability = human-understandable *why/how* of model decisions
- 5 purposes: justify, improve, trust, control, discover
- Stakeholders: developer (debug), domain user (trust), end user (why me), auditor (fairness)
- Matters more: high stakes/lives/regulation; less: low-risk (recommendations)
- Trade-off: interpretability ↔ accuracy (DT vs NN); business: ROI + compliance

**Feature importance**
- Goal often = understand drivers, not predict
- Workflow: train model → explain model → extract drivers
- Use cases: loan defaults, churn signals, cyberattack traffic

**Global vs Local**
- Local = one prediction (SHAP force, LIME, counterfactuals)
- Global = overall behavior (rankings, SHAP summary, surrogate models)

**Inherent vs Post-hoc**
- Glass box: linear/logistic regression, shallow DT, rules — read coefficients/splits directly
- Black box: NN, RF, XGBoost, high-dim SVM — need SHAP/LIME
- DT importance = split order; LogReg logic = weights; sigmoid → threshold → {0,1}
- RF = bagging + random subspace → ensemble is opaque

**Permutation importance**
- Shuffle one feature → break relationship → measure performance drop
- Importance = baseline − shuffled score; repeat K times, average
- Pros: model-agnostic, simple; Cons: correlated features, expensive, unstable
- Negative importance ⇒ noise/overfitting/redundancy/leakage

**SHAP**
- Game theory: features = players, prediction = payout
- φᵢ = weighted average of marginal contributions f(S∪{i})−f(S) over all subsets
- Weight term = fairness over all orderings
- Summary plot = global; beeswarm = per point (red = high value)
- **SHAP ≠ causality**

**Causality**
- Cause-and-effect vs correlation; concepts: inference, counterfactuals, interventions
- Confounders create misleading correlations (rain/umbrella; evacuation policy; class/cabin)
- Explainability: "what influenced the model"; Causality: "what changes the outcome"

---

# 6. High Probability Exam Questions

★★★★★ **Very High**
1. Do SHAP values reflect causality? Explain with the Titanic fare example. (→ No; fare captures class/cabin/crew confounders)
2. Describe the permutation feature importance algorithm step by step (7 steps, K repeats).
3. What does negative permutation importance mean? (→ noise, overfitting, redundancy, leakage — quiz slide)
4. Compare local vs global explainability with methods and use-cases.
5. Compare explainability analysis vs causal analysis (the two questions they answer).
6. Define confounder and give the umbrella/rain or Titanic examples.
7. If every passenger paid the same fare, what happens? (→ model less accurate; survival outcomes unchanged)

★★★★ **High**
8. Inherent (glass box) vs post-hoc (black box): definitions + 3 example models each.
9. Explain the Shapley value idea: players, payout, marginal contribution, fairness weighting.
10. Why is a Random Forest a black box although each tree is explainable?
11. Explainability–accuracy trade-off (decision tree vs neural network).
12. Why must PFI's baseline be computed on a held-out set, and why repeat K times?
13. Interpret a SHAP summary/beeswarm plot (what do bars, dots, colors mean?).
14. Why is prediction/correlation not enough for decision-making? (→ confounders; interventions)

★★★ **Medium**
15. The 5 purposes of explainability (justify/improve/trust/control/discover).
16. Stakeholder perspectives in loan approval (developer, domain/end user, auditor).
17. How does a decision tree reveal feature importance? (split order)
18. Logistic regression as an explainable model: role of weights, sigmoid, threshold.
19. When does explainability matter less? (low risk; speed/accuracy priority)
20. Compute a marginal contribution from a v(S) table; state the fare Shapley value (0.1667).
21. Define counterfactual analysis and interventions.
22. PFI advantages and limitations list.

---

# 7. Lecture Cheat Sheet

## Core Definitions
- **Explainability**: making model decisions understandable to humans (*why* + *how*, not just *what*)
- **Feature importance**: ranking of most influential features; goal = understand drivers of target
- **PFI**: performance drop when one feature is randomly shuffled (breaks feature–target link)
- **Shapley value**: fair, weighted average of a feature's marginal contributions over all subsets
- **Causal analysis**: cause-and-effect beyond correlation; supports interventions
- **Confounder**: hidden variable creating misleading correlations

## Key Comparisons

| Local | Global |
|---|---|
| One prediction; SHAP force, LIME, counterfactuals | Overall behavior; rankings, SHAP summary, surrogates |

| Glass box (inherent) | Black box (post-hoc) |
|---|---|
| Lin/Log regression, shallow DT, rules | NN, RF, XGBoost, high-dim SVM → SHAP/LIME |

| Explainability | Causality |
|---|---|
| "What influenced this decision?" | "What if we intervened?" |
| Like correlation; may credit confounders | Accounts for confounders |

## Formulas
- **LogReg**: z = wᵀx + b; ŷ = σ(z) = 1/(1+e^(−z)); threshold → {0,1}
- **PFI**: iⱼ = s − (avg over K of) sₖ,ⱼ → Importance = Baseline − After-shuffle
- **Shapley**: φᵢ = Σ_{S⊆N\{i}} [|S|!(n−|S|−1)!/n!] (v(S∪{i}) − v(S)); weight = fairness over orderings

## Pipelines
- **Explain workflow**: Train model → Explain model → Extract drivers
- **PFI**: Train → Baseline (held-out) → Pick feature → Shuffle → Re-evaluate → Drop = importance → repeat K times → all features → rank

## Algorithms & Tools
- Inherent: Decision Tree (split order), Logistic Regression (weights)
- Post-hoc: **Permutation importance** (model-agnostic), **SHAP**, **LIME**, surrogate models

## Visualizations
- SHAP **force plot** = local | SHAP **summary bar** = global mean|SHAP| | **Beeswarm** = per-point, red = high feature value | PFI **bar chart** = mean performance decrease

## The Golden Warnings
- ⚠️ **SHAP ≠ causality**; predictive models give correlational insights
- ⚠️ Negative PFI ⇒ noise/overfit/redundancy/leakage, not "useless"
- ⚠️ Titanic: sex → evacuation policy; fare → class/cabin/crew (confounders)

---

# 8. Lecture Summary

## Top 50 Most Probable Exam Questions
1. Define explainability. → Making model decisions understandable to humans (why + how).
2. What three questions does explainability address? → What, why, how of decisions.
3. List the 5 "explain to" purposes. → Justify, improve, build trust, control, discover.
4. When does explainability matter more? → Lives affected, regulation, trust/accountability.
5. When does it matter less? → Low-risk decisions; speed/accuracy priority.
6. Name 4 high-stakes fields. → Medicine, finance, legal/justice, military/security (also transport).
7. What is the explainability–accuracy trade-off? → Interpretable models less accurate; accurate models opaque.
8. Which model is more interpretable: DT or NN? → Decision tree.
9. Two business factors for model selection? → ROI, compliance.
10. What does the auditor want? → Fairness; no prohibited features (race); no disproportionate denials.
11. What does the end user ask? → "Why was my loan denied?"
12. Goal of feature importance analysis? → Highlight most influential features / understand target drivers.
13. Give 3 feature-importance use cases. → Loan-default indicators, churn behavioral signals, cyberattack network features.
14. Define local explainability. → Explain a single prediction for one instance.
15. Define global explainability. → Understand overall model behavior across all data.
16. 3 local methods? → SHAP force plot, LIME, counterfactuals.
17. 3 global methods? → Importance rankings, SHAP summary plot, global surrogate model.
18. What is LIME? → Linear approximation around one point.
19. What is a counterfactual explanation? → "What if income was higher?"
20. Define glass box / inherent explainability. → Transparent, self-explanatory; inspect logic directly.
21. 4 glass box examples? → Linear regression, logistic regression, shallow DT, rule-based.
22. 4 black box examples? → NN, random forest, XGBoost, high-dim SVM.
23. What is post-hoc explainability? → Separate techniques (SHAP/LIME) applied after training.
24. How does a DT reveal importance? → Order of the splits.
25. How does LogReg reveal logic? → Learnt weights.
26. LogReg pipeline components? → Weighted sum → sigmoid → threshold/quantizer → binary.
27. Is the threshold fixed? → No, adjustable.
28. Why is RF a black box? → Ensemble of trees (bagging + random subspace) hard to explain collectively.
29. Define permutation feature importance. → Performance drop when a feature is randomly shuffled.
30. PFI underlying hypothesis? → Shuffling breaks the feature–target relationship.
31. List the 7 PFI steps. → Train, baseline, pick feature, shuffle, re-evaluate, compute drop, repeat for all.
32. Why repeat K times? → Account for random variation.
33. Where is baseline computed? → Held-out set with scoring function (accuracy, R², F1).
34. PFI formula? → iⱼ = s − avg(sₖ,ⱼ) = baseline − shuffled score.
35. PFI advantages? → Model-agnostic; easy to explain/implement.
36. PFI limitations? → Correlated features mask importance; expensive; unstable on small data.
37. Negative importance meaning? → Noise, overfitting, redundancy, leakage, harmful dependency.
38. Titanic PFI top features? → Sex, then pclass.
39. Origin of Shapley values? → Cooperative game theory (Lloyd Shapley, Nobel 2012).
40. ML mapping? → Players = features; payout = prediction.
41. Marginal contribution formula? → f(S∪{i}) − f(S).
42. Purpose of the combinatorial weight? → Fairness by averaging all feature orderings.
43. Why weighting needed? → Contribution depends on what information is already available.
44. What does the beeswarm plot show? → Per-point SHAP values; red = high feature value.
45. Do SHAP values reflect causality? → No — explicit lecture warning.
46. Predictive importance vs actionability? → "Model uses it" vs "changing it changes the outcome".
47. Define confounder + example. → Hidden variable causing misleading correlation; rain (umbrella), evacuation policy (sex).
48. Titanic fare confounders? → Passenger class, cabin location, crew assistance.
49. Three key causal concepts? → Causal inference, counterfactual analysis, interventions.
50. Questions answered by explainability vs causality? → "What influenced this decision?" vs "What if we intervene?"

## Top 30 Definitions
1. **Explainability** — making model decisions understandable to humans
2. **XAI** — explainable AI: model highlights symptoms/patterns behind decisions
3. **Feature importance** — measure of a feature's influence on the model/target
4. **Local explainability** — explanation of a single prediction
5. **Global explainability** — overall model behavior across all data
6. **SHAP force plot** — local visualization for one prediction
7. **SHAP summary plot** — global bar plot of mean |SHAP|
8. **Beeswarm plot** — per-point SHAP values colored by feature value
9. **LIME** — local linear approximation around one point
10. **Counterfactual explanation** — "what if X changed?" explanation
11. **Global surrogate model** — simple model mimicking a black box
12. **Inherent explainability** — transparent-by-design (glass box)
13. **Post-hoc explainability** — interpretation tools applied after training
14. **Glass box** — transparent, self-explanatory model
15. **Black box** — opaque model needing interpretation tools
16. **Sigmoid** — function mapping z to probability (0,1)
17. **Quantizer/threshold** — converts probability to binary class
18. **Bagging** — bootstrap aggregating (part of Random Forest)
19. **Random subspace method** — random feature subsets per tree
20. **Permutation feature importance** — performance drop after shuffling a feature
21. **Baseline performance (s)** — held-out score before permutation
22. **Held-out set** — validation/test data not used in training
23. **Scoring function** — metric like accuracy, R², F1
24. **Model-agnostic** — applicable to any model type
25. **Shapley value** — fair average marginal contribution of a feature
26. **Marginal contribution** — f(S∪{i}) − f(S)
27. **Causal analysis** — study of cause-and-effect beyond correlation
28. **Causal inference** — determining whether one variable causes another
29. **Intervention** — applying/simulating an action to observe causal impact
30. **Confounder** — hidden variable creating misleading correlations

## Top 30 Keywords
Explainability • Trust • Compliance • ROI • High stakes • Stakeholders • Auditor • Trade-off • Interpretability • Accuracy • Feature importance • Target variable • Local • Global • LIME • Counterfactual • Surrogate model • Glass box • Black box • Post-hoc • SHAP • Shapley value • Marginal contribution • Cooperative game theory • Permutation • Shuffling • Baseline • Held-out set • Model-agnostic • K repetitions • Negative importance • Leakage • Sigmoid • Threshold • Bagging • Random subspace • Causality • Causal inference • Intervention • Confounder

## Top 20 Comparisons
1. Local vs Global explainability
2. Inherent (glass box) vs Post-hoc (black box)
3. Decision Tree vs Neural Network (interpretability vs accuracy)
4. Explainability vs Causality
5. Predictive importance vs Actionability
6. Correlation vs Causation
7. PFI vs SHAP
8. SHAP summary plot vs beeswarm plot
9. SHAP force plot (local) vs summary plot (global)
10. "What the model uses" vs "what changes the outcome"
11. Prediction goal vs understanding-drivers goal
12. Explainability matters more vs matters less situations
13. One decision tree vs Random Forest (explainable vs black box)
14. ROI vs compliance
15. Doctor with vs without explainable AI (hospital scenario)
16. What/How vs Why (rocket metaphor)
17. Baseline score vs post-shuffle score
18. Positive vs negative permutation importance
19. Fare as predictor vs fare as cause (Titanic)
20. Sex as predictor vs evacuation policy as cause (Titanic)

## Top 20 Formulas / Formal Items
1. z = Σ(wᵢxᵢ) + b
2. z = wᵀx + b
3. ŷ = σ(z) = 1/(1+e^(−(wᵀx+b)))
4. Threshold: ŷ ≥ t → 1, else 0
5. PFI: Importance = Baseline − New performance
6. PFI formal: iⱼ = s − Σₖ₌₁ᴷ sₖ,ⱼ (averaged over K)
7. Example: 90% − 78% = 12% importance
8. Marginal contribution: f(S∪{i}) − f(S)
9. Shapley: φᵢ(v) = Σ_{S⊆N\{i}} [|S|!(n−|S|−1)!/n!](v(S∪{i})−v(S))
10. Weight term: |S|!(n−|S|−1)!/n! (fairness over orderings)
11. Lecture result: φ_fare = 0.1667
12. Number of orderings for n features = n! (3 features → 6)
13. v(S) = model prediction with subset S
14. mean(|SHAP value|) = global importance metric
15. Pr(lung cancer | smoker) vs Pr(lung cancer | nonsmoker)
16. Scoring functions: accuracy, R², F1
17. Marginal contribution of Gender first: f({Gender}) − f({})
18. Marginal contribution of Gender last: f({Age,Income,Gender}) − f({Age,Income})
19. Random Forest = Bagging + Random Subspace + DT learning
20. Example table row: v({age})=0.4, v({age,fare})=0.6 → contribution 0.2

## Top 20 Diagrams/Images to Understand
1. Rocket WHAT/HOW/WHY (why-powered rocket flies) — slide 3
2. Explain-to cycle (justify→improve→trust→control→discover) — slide 5
3. Loan approval stakeholder diagram — slide 6
4. DT vs NN smiley trade-off table — slide 7
5. High-stakes fields collage — slide 8
6. Train-model pipeline (Step 1) — slide 11
7. Explain-model pipeline with X1…X6 arrows (Step 2) — slide 12
8. Local vs Global comparison boxes — slide 14
9. Glass jar vs black box icons — slide 15
10. Student pass/fail decision tree — slide 16
11. Logistic regression network diagram (sum→sigmoid→quantizer) — slide 17
12. Random Forest construction diagram — slide 18
13. PFI 7-step infographic (appears twice!) — slides 20, 22
14. "Steps 4–6 repeated K times" highlight — slide 22
15. Titanic PFI bar chart (F1) — slide 24
16. PFI pros/cons boxes — slide 25
17. SHAP marginal contribution table (fare) — slide 28
18. 6-permutation weighting table (Gender/Age/Income) — slide 30
19. SHAP summary bar vs beeswarm plots — slide 31
20. Mary diagnosis→treatment→death timeline — slides 36–37

## 10-Minute Revision Sheet
1. **Explainability** = human-understandable why/how. Needed when **stakes high** (lives, law, trust); skip when low-risk.
2. **5 purposes**: justify, improve, trust, control, discover. **4 stakeholders**: developer, domain user, end user, auditor.
3. **Trade-off**: interpretable (DT, LR) ↔ accurate (NN). Business: ROI + compliance.
4. **Local** (one prediction: force plot, LIME, counterfactual) vs **Global** (behavior: rankings, summary plot, surrogate).
5. **Glass box** (lin/log regression, shallow DT, rules — read weights/splits) vs **black box** (NN, RF, XGBoost — apply SHAP/LIME post-hoc). RF = bagging + random subspace ⇒ opaque ensemble.
6. **PFI**: shuffle one feature → break feature–target link → **Importance = baseline − new score**, on **held-out** data, repeat **K times**. Pros: model-agnostic, simple. Cons: correlated features, cost, instability. **Negative ⇒ noise/overfit/redundancy/leakage.**
7. **SHAP**: game theory; features = players, prediction = payout. φᵢ = weighted avg of **f(S∪{i})−f(S)** over all subsets; weight = fairness over all orderings. Titanic: sex > class > fare. Beeswarm: red = high value.
8. **⚠️ SHAP ≠ causality.** Predictive importance ("model uses it") ≠ actionability ("changing it changes outcome"). Same-fare thought experiment: model worse, reality unchanged.
9. **Causality**: inference + counterfactuals + interventions; accounts for **confounders**. Rain→umbrella+accidents; evacuation policy→sex↔survival; class/cabin/crew→fare↔survival.
10. **Two questions**: Explainability = "What influenced this decision?" Causality = "What if we intervened?"

---

# 9. Examples Repository

### 1. Hospital AI diagnosis (slide 4)
- **Topic**: Why explainability matters
- **Context**: AI diagnoses vague symptoms, recommends intensive, high-risk, costly treatment
- **Explanation**: Without explanations, doctor/patient can't see the *why*; trust collapses. With XAI, the model shows symptoms/risk factors used, letting the doctor validate against medical judgment and the patient decide informedly.
- **Why important**: Canonical "high stakes" motivating example.
- **Possible exam questions**: "List two problems without explainability and two benefits with it in the hospital scenario."

### 2. Loan approval — four stakeholders (slide 6)
- **Topic**: Perspectives on explainability
- **Context**: Developer, domain user, end user, auditor each ask different questions about a loan model.
- **Explanation**: Explainability serves debugging (developer), trust (domain user), individual justification (end user), fairness/regulatory checks (auditor — race excluded? minorities disproportionately denied?).
- **Why important**: Shows explainability is stakeholder-dependent; frequent matching-question material.
- **Possible exam questions**: Match stakeholder → question; "Which stakeholder cares about prohibited features?"

### 3. Decision Tree vs Neural Network trade-off (slide 7)
- **Topic**: Explainability–accuracy trade-off
- **Context**: Smiley-face table: DT interpretable but less accurate; NN accurate but opaque; plus ROI/compliance business framing.
- **Explanation**: Choosing a model balances clarity vs performance and profit vs legal standards.
- **Possible exam questions**: "Fill the trade-off table"; "Name the two business factors."

### 4. Feature-importance use cases (slide 9)
- **Topic**: Feature importance motivation
- **Context**: (a) financial indicators → loan defaults, (b) behavioral signals (clicks, scroll depth, session time) → conversion/churn, (c) network traffic features → cyberattacks.
- **Explanation**: In each case, the goal is identifying **drivers**, not deploying a predictor.
- **Possible exam questions**: "Give a use-case where understanding drivers matters more than prediction."

### 5. Disease-risk model, 2-step use-case (slides 11–12)
- **Topic**: Train-then-explain workflow
- **Context**: Multi-dimensional health data (EMR, claims) → ML model → predictions (hospitalization risk, fast CKD progressors, Long-COVID risk); then explain to extract risk factors X1…X6.
- **Explanation**: Step 1 learns risk factors implicitly; Step 2 queries the model to expose them.
- **Possible exam questions**: Pipeline ordering; "What are example predicted outcomes?"

### 6. Student pass/fail decision tree (slide 16)
- **Topic**: Inherently explainable models
- **Context**: Features: Study Hours, Social Media Time, Sport Time, Bedtime, Age → Pass/Fail tree; root = Study Hours.
- **Explanation**: The split order ranks importance — Study Hours first ⇒ most important.
- **Possible exam questions**: "Which feature is most important and why?"; trace a prediction through the tree.

### 7. Breast cancer logistic regression (slide 17)
- **Topic**: Inherently explainable models
- **Context**: Predict breast cancer from pathological image features (age, radius_mean, texture_mean); weights → weighted sum → sigmoid → threshold → {0,1}.
- **Explanation**: Learnt weights reveal the model's logic; threshold adjustable.
- **Possible exam questions**: Label the diagram; "How do we infer model logic?" (weights).

### 8. Fruit dataset Random Forest (slide 18)
- **Topic**: Black box models
- **Context**: Mass/Color/Texture/pH → Apple/Orange table fed into Bagging + Random Subspace + DT learning → Trees 1…N.
- **Explanation**: Each tree is explainable given its subset, but the ensemble's collective decision is not → post-hoc methods needed.
- **Possible exam questions**: "Why is RF a black box?"; name RF's three ingredients.

### 9. Age-shuffling PFI walkthrough (slides 20/22)
- **Topic**: Permutation feature importance
- **Context**: Baseline accuracy 90%; shuffle Age; new accuracy 78%; importance = 12%; then Income 7%, Balance 4%, Region 1%, Gender 0%.
- **Explanation**: Numeric demonstration of Importance = Baseline − New; ranking by drop.
- **Possible exam questions**: Compute importance from given accuracies; rank features.

### 10. Titanic permutation importance (slide 24)
- **Topic**: PFI in practice
- **Context**: RandomForestClassifier on Titanic, F1 scoring, n_repeats=10; bar chart: sex ≈ 0.20 > pclass > age > fare > sibsp > parch ≈ embarked (slightly negative).
- **Explanation**: Sex dominates survival prediction; tiny/negative bars = unimportant or noisy features.
- **Possible exam questions**: Read the chart; explain the slightly negative bars.

### 11. Titanic "fare" Shapley computation (slide 28)
- **Topic**: SHAP values
- **Context**: Table of subsets (∅, {age}, {sex}, …, {age,sex,sibsp}) with v(S), v(S∪{fare}), marginal contributions (0.2, 0.2, 0.2, …, 0.10) → φ_fare = 0.1667.
- **Explanation**: Concrete application of the Shapley formula with the fairness weighting.
- **Possible exam questions**: Compute a marginal contribution from a row; state what the weight term does.

### 12. House price with Age, Income, Gender (slides 29–30)
- **Topic**: SHAP weighting schema
- **Context**: 3 features → 6 permutations; table shows features "before Gender" and Gender's marginal contribution in each ordering.
- **Explanation**: A feature's contribution depends on what information is already available, so we average over all orderings — this is the fairness rationale for the combinatorial weights.
- **Possible exam questions**: "Why can't we just add features in one fixed order?"; fill a row of the permutation table.

### 13. Titanic SHAP plots (slide 31)
- **Topic**: Global vs local SHAP
- **Context**: Summary bar plot (sex > class > fare > age > embarked by mean|SHAP|) and beeswarm plot (per-passenger SHAP values, red = high feature value).
- **Explanation**: Same SHAP values viewed globally (magnitude ranking) and locally (direction + spread per point vs baseline).
- **Possible exam questions**: Identify which plot is global/local; interpret colors.

### 14. Mary's breast-cancer treatment timeline (slides 36–37)
- **Topic**: Causality motivation
- **Context**: Diagnosis (X) → Treatment → Death (Y). "A long survival time may be because of treatment!" Complications: aggressive treatment decided on predicted outcome; unrelated comorbidities; post-treatment unhealthy lifestyle. Related questions: smoking→breast cancer? obesity→drug response? Pr(lung cancer|smoker) vs nonsmoker.
- **Explanation**: Decision-making requires understanding **interventions**, not just model reasoning — prediction can't isolate the treatment's causal effect.
- **Possible exam questions**: "Why can't a predictive survival model tell us if treatment works?"

### 15. Umbrella–accidents confounder (slide 39)
- **Topic**: Confounders
- **Context**: Model: umbrella owners have more traffic accidents; umbrella ownership ranked important. Causal reality: **Rain** increases both umbrella use and accident risk; giving everyone an umbrella will **not** reduce accidents.
- **Explanation**: Classic misleading correlation via a hidden common cause.
- **Possible exam questions**: Identify the confounder; "Would intervening on umbrellas help?"

### 16. Titanic Sex vs Survival (slide 40)
- **Topic**: Explainability vs causality
- **Context**: Model says Sex most important (women survived more). Cause: **"women and children first" evacuation policy** determining lifeboat access; policy **not recorded in the dataset**.
- **Explanation**: Feature importance flagged a proxy; the true cause was an unobserved policy.
- **Possible exam questions**: "What is the confounder for sex→survival, and why is it special?" (unrecorded).

### 17. Titanic Fare vs Survival (slide 41)
- **Topic**: Explainability vs causality
- **Context**: High fare predicts survival; but paying more didn't directly save anyone — higher fare ↔ **higher passenger class** → cabins closer to lifeboats + priority access. Confounders: **passenger class, cabin location, crew assistance**.
- **Explanation**: Fare is a predictive proxy for structural advantages.
- **Possible exam questions**: List the three fare confounders; connect to the SHAP-≠-causality quiz.

---

# 10. Quiz Repository

### Quiz 1 (slide 21) — Negative permutation importance
- **Original question**: "After running the algorithm, I figured out that permutating a specific feature is slightly *increasing* performance. Why may this happen?"
- **Correct answer** (given on slide): Negative permutation importance does **not** mean the feature is useless. It means the model performs **better when the feature's information is destroyed**, suggesting **noise, overfitting, redundancy, leakage, or a harmful learned dependency**.
- **Explanation**: If shuffling a feature improves the score, the model had learned something counterproductive from it (or the feature duplicated other features / leaked target info), so destroying it helps slightly.
- **Related topic**: Permutation Feature Importance
- **Difficulty**: **Medium–Hard**

### Quiz 2 (slide 32) — SHAP says Fare is important
- **Original question**: "A highly accurate model predicts Titanic survival. Its explainability report (e.g., SHAP) shows that **Fare** is one of the most important features."
  - a) Fare directly affects survival because changing an important feature should change the prediction.
  - b) Fare is useful for prediction, but it may simply capture information about other factors that actually influenced survival.
  - c) Fare must have been included in the evacuation policy, otherwise the model would not rank it so highly.
  - d) Since Fare is highly important across many passengers, it is both predictive and causal.
- **Correct answer**: **(b)** *(inferred from the lecture content — the slide gives no explicit answer, but slides 34, 38, 41 state SHAP ≠ causality and fare's importance comes via confounders: passenger class, cabin location, crew assistance).*
- **Explanation**: SHAP measures **predictive** importance only. Fare correlates with passenger class and cabin location, which actually influenced lifeboat access. Options a, c, d wrongly assume causal meaning.
- **Related topic**: SHAP values, causality
- **Difficulty**: **Medium**

### Quiz 3 (slide 33) — Everyone pays the same fare
- **Original question**: "Imagine every passenger paid exactly the same fare, but everything remained unchanged. What would you expect?"
  - a) The survival rate would increase because Fare was an important feature.
  - b) The model would become less accurate, but the actual survival outcomes would likely remain unchanged.
  - c) The survival rate would become nearly identical across all passengers.
  - d) It is impossible to say because feature importance measures causal effect.
- **Correct answer**: **(b)** *(inferred — supported by the slide's own footnote distinguishing* **predictive importance** *("The model uses this") from* **actionability/causality** *("Changing this changes the outcome")).*
- **Explanation**: Fare is only an information source for the model. Removing its variation degrades the **model's** predictions but does not alter the real causal drivers (class, cabin location, evacuation policy), so real outcomes stay the same.
- **Related topic**: Predictive importance vs causality
- **Difficulty**: **Medium**

### Checkpoint/discussion questions (slides 36–37, "Mary" example)
- **Original questions**:
  1. "What is the likelihood this patient with breast cancer treatment will survive 5 years?"
  2. "What if we must decide to start an aggressive treatment based on the predicted outcome?"
  3. "What if the patient has developed unrelated comorbidities in-between and those led to death?"
  4. "What if the patient's lifestyle becomes unhealthy after the treatment?"
  5. "Does smoking cause breast cancer?" / "Does obesity cause low response for drug A?" / "Pr(lung cancer|smoker) vs Pr(lung cancer|nonsmoker)?"
- **Correct answer** *(inferred from lecture)*: These illustrate that a **predictive model cannot answer intervention questions** — survival time may be due to treatment, comorbidities, or lifestyle. Answering them requires **causal analysis** (causal inference, counterfactuals, interventions) that accounts for **confounders**; conditional probabilities alone are correlational.
- **Explanation**: The point of the example is that decision-making requires understanding **interventions, not just model reasoning**.
- **Related topic**: Introduction to Causal Analysis
- **Difficulty**: **Hard**

### Rhetorical review question (slide 38)
- **Original question**: "Why cannot we simply use predictive modeling or simply correlation for this?"
- **Correct answer** (given on slide): Explainability reveals **what influenced the model, not what changes the outcome**; predictive-model explanations are like **correlations** and may credit **confounders**; **causality accounts for confounders** to separate true causes from misleading correlations.
- **Related topic**: Explainability vs causality
- **Difficulty**: **Easy–Medium**

---