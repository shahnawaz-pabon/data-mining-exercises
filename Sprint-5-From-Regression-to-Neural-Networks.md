# 📖 Lecture: Sprint5_From regression to neural networks

---

# 1. Lecture Overview

## Main Topics
1. **Background** — Biological neurons vs. Artificial Neural Networks (single perceptron); Classical programming vs. ML vs. Deep Learning
2. **Regression** — Linear regression, assumptions, loss/objective functions (MSE), other types of regression (Polynomial, Ridge, Lasso, Elastic Net)
3. **Gradient Descent** — The gradient, the GD algorithm, learning rate, convexity, GD vs. SGD vs. Mini-Batch GD, learning rate decay
4. **Logistic Regression** — Sigmoid function, cross-entropy cost, gradient descent with the chain rule, multinomial classification (One-vs-Rest, SoftMax)
5. **Why do we need a Neural Network** — Limitations of logistic regression, hidden layers, forward propagation, backpropagation, activation functions, network size, training tricks, normalization

## Learning Objectives
- Understand that a **single perceptron is essentially linear/logistic regression** — Neural Networks are the *generalization* of linear regression.
- Fit models by minimizing a **cost function** using **Gradient Descent**.
- Convert regression → classification via **sigmoid** and **cross-entropy**.
- Understand **forward propagation** and **backpropagation** (chain rule).
- Know the practical "box of tricks": optimizers, learning rate schedules, regularization, normalization.

## Important Takeaways (repeated/highlighted in the lecture)
- 🔑 **"Neural networks are the generalization of linear regression."** (Repeated as a cartoon on slide 2.)
- 🔑 **"A neural network can be explained as several logistic regression blocks — we call these blocks neurons."** (Bottomline slide 51.)
- 🔑 **Fitting a model = finding parameters θ (weights ω and bias b) that minimize the cost function.**
- 🔑 In **convex problems (like linear regression with MSE), GD is guaranteed to reach the global minimum**.
- 🔑 **Learning rate**, **network size**, and **activation function** are all **hyperparameters** that need fine tuning.
- 🔑 **Golden Rule of Neural Nets: "Neural Networks are the second-best way to do everything!"**

---

# 2. Topic-wise Notes

---

## Topic 1: Biological Neurons vs. Artificial Neural Networks (Perceptron)

### Definition
An **artificial neuron (perceptron)** is a computing unit inspired by biological neurons. It multiplies inputs by **weights**, adds a **bias**, sums everything at a **summing junction**, and passes the result through an **activation function** to produce an output.

### Simple Example
Just as a brain neuron fires when signals arriving through synapses cross a threshold, a perceptron "fires" (outputs a value) when the weighted sum of its inputs passes through the activation function.

### Why is it Important?
It is the fundamental building block of all neural networks and is the bridge between regression and deep learning.

### Key Points
- **Inputs:** x₁, x₂, …, xₙ
- **Synaptic weights (including bias):** w₁, …, wₙ + bias
- **Summing junction:** Σ combines weighted inputs + bias
- **Activation function φ(.)** produces the output
- Biological: Neuron 1 → **Axon** → **Synapse** → Neuron 2
- **A single perceptron is not that different from linear or logistic regression!** (Emphasized slide 7)

### Common Mistakes
- Thinking a perceptron is fundamentally different from regression — with identity activation it *is* linear regression; with sigmoid activation it *is* logistic regression.
- Forgetting the **bias** is part of the parameters.

### Comparison

| Biological Neuron | Artificial Neuron |
|---|---|
| Dendrites receive signals | Inputs x₁…xₙ |
| Synapse strength | Synaptic weights w₁…wₙ |
| Cell body integrates | Summing junction Σ |
| Action potential/firing threshold | Activation function φ(.) |
| Axon transmits output | Output |

### Diagram / Image Explanation
**Slide 4 diagram (single perceptron):** Inputs x₁…xₙ each multiplied by weight w₁…wₙ, plus bias, entering Σ (summing junction), then φ(.) (activation function), then output.
- *Possible exam question:* Label the components of a perceptron.
- *Common mistake:* Placing the activation function before the summing junction.

### Important Keywords
- **Perceptron** — a single artificial neuron.
- **Synaptic weights** — trainable multipliers on inputs.
- **Bias** — constant added to weighted sum.
- **Summing junction** — Σ that aggregates weighted inputs.
- **Activation function φ** — transforms the sum into the output.

### Memory Tips
**"W-S-A-O"**: **W**eigh → **S**um → **A**ctivate → **O**utput.

---

## Topic 2: Deep Learning vs. Machine Learning vs. Classical Programming

### Definition
Classical programming uses hand-written rules; machine learning learns patterns from data using **hand-designed (pre-calculated) features**; deep learning learns patterns using **self-estimated features** it extracts on its own.

### Simple Example
Spam detection: classical = write "if contains word X" rules; ML = you compute features (word counts) and train a model; DL = feed raw emails, the network learns its own features.

### Why is it Important?
This distinction is *the* answer to "What makes DL so special?" — a highlighted lecture question.

### Key Points
- **Classical approach:** Study problem → **Write rules** → Evaluate → Launch (a priori rule definition).
- **ML approach:** Data + **Hand-Designed Features** → Train ML algorithm → Evaluate → Launch. Algorithms recognize patterns using **pre-calculated features**.
- **DL approach:** Data → Train ML algorithm → Evaluate → Launch. Algorithms recognize patterns using **self-estimated features** (no hand-designed feature step!).
- **Deep learning models use three or more layers — typically hundreds or thousands of layers.**
- "Non-deep" feedforward NN = one hidden layer; Deep NN = multiple hidden layers.

### Common Mistakes
- Saying DL needs hand-crafted features — the whole point is it does **not**.
- Thinking "deep" means many neurons — it means many **layers**.

### Comparison

| Aspect | Classical Programming | Machine Learning | Deep Learning |
|---|---|---|---|
| Rules | Written a priori by humans | Learned from data | Learned from data |
| Features | N/A | **Hand-designed / pre-calculated** | **Self-estimated** |
| Feature engineering needed? | — | Yes | No |
| Layers | — | Shallow | 3+ (often hundreds/thousands) |

### Diagram / Image Explanation
**Slide 6 (three pipelines side by side):** Shows that the "Hand-Designed Features" box (marked with ✗) disappears in the DL pipeline.
- *Possible exam question:* "What is the key difference between the ML and DL pipeline?" → Feature extraction is automatic in DL.

**Slide 5 (non-deep vs. deep network):** Left = input, ONE hidden layer, output. Right = input, hidden layers 1–3, output.
- *Exam question:* "How many layers make a network deep?" → 3 or more.

### Important Keywords
- **Hand-designed features** — features manually engineered by humans (ML).
- **Self-estimated features** — features the network learns itself (DL).
- **Hidden layer** — layer between input and output.
- **Deep neural network** — network with 3+ layers.

### Memory Tips
**"Rules → Features → Nothing"**: Classical needs rules, ML needs features, DL needs neither — just data.

---

## Topic 3: Linear Regression

### Definition
**Regression** is a method to model the relationship between a **dependent variable (output)** and one or more **independent variables (input)**. The goal is to **predict continuous values** using independent variables (features or attributes).

### Simple Example
**Predict house prices using house characteristics (features):** area, rooms, location → price. (Lecture's running example: 120 m², 3 rooms, central → 500K.)

### Why is it Important?
Linear regression is the foundation from which the entire lecture builds up to neural networks — it is literally a perceptron with identity activation.

### Key Points
- Data = **historical observations**.
- Model: **y = b + Σᵢ xᵢwᵢ**, in vector form **ŷ = xω + b**.
- **To fit the model = to find the best parameters that have an acceptable error margin but still reliable.**
- If the model works, we rely on it for future **predictions**: ŷ = f(θ̂).
- Training needs a **loss function** (single prediction error) and an **objective/cost function** (error for all data points).
- As a perceptron: activation function = **identity (1)**.

### Common Mistakes
- Confusing **loss** (one data point) with **cost/objective** (all M data points).
- Confusing dependent (output y) vs. independent (input x) variables.
- Using linear regression to predict categories (that's classification/logistic regression).

### Comparison

| Regression | Classification |
|---|---|
| Predicts **continuous** values | Predicts **discrete classes** |
| e.g., house **price** | e.g., benign/malignant |
| Linear regression, MSE | Logistic regression, cross-entropy |
| Identity activation | Sigmoid activation |

### Formula Explanation

**Model: y = b + Σᵢ xᵢ wᵢ**
- **y** = output (prediction), **b** = bias, **xᵢ** = i-th input, **wᵢ** = weight on i-th input, **i** = index over input connections.
- *Intuition:* a weighted sum of features plus a base value.
- *When used:* predicting any continuous target.
- *Common mistake:* forgetting the bias term b.

**Vectorized: ŷ = xω + b** — same model written with vectors (x = feature vector, ω = weight vector).

### Diagram / Image Explanation
**Slides 8–10:** Table of house data (area, rooms, location → price) next to the model equation / perceptron diagram with activation function "1" (identity).
- *Why important:* shows linear regression IS a perceptron with identity activation.
- *Exam question:* "What is the activation function of a linear regression perceptron?" → identity (1 / none).

**Slide 11 (regression line plot):** dataset points, regression line, dashed vertical lines = **prediction errors** (residuals).
- *Exam question:* "What do the dashed lines represent?" → the prediction errors between actual points and the regression line.

### Important Keywords
- **Dependent variable** — the output y we predict.
- **Independent variables** — input features/attributes x.
- **Fit the model** — find best parameters with acceptable error but still reliable.
- **Parameters θ** — weights (ω) and bias (b).
- **Prediction error / residual** — vertical distance from point to regression line.

### Memory Tips
"**y** depends on **x**" → y = **de**pendent, x = **in**dependent.
Linear regression = perceptron wearing an "identity" mask.

---

## Topic 4: Assumptions of Linear Regression

### Definition
Five conditions that must hold for linear regression results to be valid: linearity, independence, homoscedasticity, normality of errors, and no multicollinearity.

### Simple Example
Predicting house prices: size relates linearly to price (linearity), houses selected randomly across the city (independence), price errors spread equally across price levels (homoscedasticity), most errors small (normality), and number of bedrooms/bathrooms shouldn't both be used if highly correlated (no multicollinearity).

### Why is it Important?
Classic exam list — professors love asking to name and explain the 5 assumptions.

### Key Points
- **Linearity:** Relationship between X and Y is linear (e.g., size relates to price).
- **Independence:** Observations are independent (select houses randomly across the city).
- **Homoscedasticity:** Constant variance of residuals (equally difficult to predict price everywhere; errors spread equally across price levels).
- **Normality:** Errors are normally distributed (most errors small, big misses rare).
- **No multicollinearity:** Predictors are not too highly correlated (e.g., no. of bedrooms and no. of bathrooms can be correlated).

### Common Mistakes
- Confusing **homoscedasticity** (constant error variance) with **normality** (error distribution shape).
- Thinking normality applies to the features — it applies to the **errors/residuals**.
- Multicollinearity is about **predictors correlating with each other**, not with the target.

### Comparison

| Assumption | About what? | Violation example |
|---|---|---|
| Linearity | X–Y relationship | Curved price-size relation |
| Independence | Observations | Sampling only one street |
| Homoscedasticity | Residual variance | Errors grow with price |
| Normality | Residual distribution | Many huge misses |
| No multicollinearity | Predictors amongst themselves | Bedrooms ≈ bathrooms |

### Important Keywords
- **Homoscedasticity** — constant variance of residuals across all levels.
- **Multicollinearity** — predictors highly correlated with each other.
- **Residual** — error between actual and predicted value.

### Memory Tips
**L.I.H.N.M.** → "**L**ions **I**n **H**ouses **N**ever **M**eow" (Linearity, Independence, Homoscedasticity, Normality, no Multicollinearity).

---

## Topic 5: Loss Function, Objective/Cost Function & MSE

### Definition
A **loss function** measures how far off a **single prediction** is from its actual value. The **objective function (cost function)** summarizes the error produced by the model **for all M data points**. In linear regression, the common cost function is the **Mean Squared Error (MSE)**.

### Simple Example
If a house costs 500K and the model predicts 450K, the squared loss for that house is (500−450)² = 2500 (in K²). Average such losses over all houses (÷2M) → MSE.

### Why is it Important?
Every training procedure in this lecture is "minimize a cost function" — this is THE central concept, backed by a dedicated quiz.

### Key Points
- Loss (single point): **(y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)²** — goal: **minimize that error**.
- Cost: **MSE = 𝒥(θ) = (1/2M) Σᵢ₌₁ᴹ (y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)²**
- **Fitting the model** = finding parameters **θ** (weights **ω** and bias **b**) that **minimize this error**.
- MSE plotted against a single parameter is a **parabola** (quadratic function); the minimum point is **θ̂** — the parameter value with minimal loss.
- MSE is **quadratic**: the further from the optimum, the bigger (quadratically) the error.
- We can't just use brute force (np.min) because it doesn't know the structure of the cost function and is impractical for continuous, high-dimensional problems → we need an optimization algorithm that **exploits the mathematical structure** (gradient descent).

### Common Mistakes
- Mixing loss (1 point) and cost (all M points).
- Forgetting the **1/2M** factor in the lecture's MSE (the 2 simplifies the derivative).
- Thinking the parabola is plotted over x — it is plotted over the **parameter θ**.

### Comparison

| Loss Function | Cost / Objective Function |
|---|---|
| One single prediction | All M data points |
| (y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)² | (1/2M) Σ (y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)² |
| Per-example error | Summarized / averaged error |

### Formula Explanation

**MSE = 𝒥(θ) = (1/2M) Σᵢ₌₁ᴹ (y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)²**
- **M** = number of data points; **N** = number of features.
- **y⁽ⁱ⁾** = actual value of i-th data point; **ŷ⁽ⁱ⁾** = predicted value.
- **θ** = parameters {ω, b}. Note ŷ = xω + b, so the parameters are hidden inside ŷ.
- *Intuition:* average squared distance between prediction and truth; squaring punishes big errors more and makes the function smooth and convex.
- *When used:* linear regression training.
- *Interpretation:* lower MSE = better fit; the minimum of the parabola = best parameter θ̂.
- *Common mistake:* not knowing **where the parameters are** in the equation — they are inside ŷ (quiz question!).

### Diagram / Image Explanation
**Slides 14–16 (cost parabola):** x-axis = θ (parameter), y-axis = cost. Red parabola with the minimum marked θ̂ = "parameter value with the minimal loss." Note: "For simplification we assume we have only one parameter to optimize."
- *Exam questions:* Why parabola? (quadratic function). Where are the parameters? (inside ŷ). Why squared and not absolute? (quadratic growth away from optimum).
- *Common mistake:* reading the x-axis as data x instead of parameter θ.

### Important Keywords
- **Loss function** — error of one prediction.
- **Objective/cost function** — total (averaged) error over all data.
- **MSE** — Mean Squared Error, the standard linear regression cost.
- **θ̂ (theta hat)** — the optimal (estimated) parameter value; **hat = estimated value**, kept even after convergence because it remains an *estimate*, not the true value.
- **Convex/quadratic** — bowl-shaped; single global minimum.

### Memory Tips
"**L**oss = **L**one point; **C**ost = **C**omplete dataset."
The ½ in 1/2M exists to cancel the 2 from the derivative of the square.

---

## Topic 6: The Gradient ("The Path and the Pace")

### Definition
The **gradient** is the **derivative of a function** with respect to its input variables. It tells us the **direction of steepest ascent** and its **magnitude** tells us how steep the slope is.

### Simple Example
Standing on a hill in fog: the gradient points uphill in the steepest direction; how strongly it "pulls" tells you how steep it is. To descend (minimize), you step in the **opposite** direction.

### Why is it Important?
The gradient is the engine of gradient descent, backpropagation, and all NN training.

### Key Points
1. **Direction of Steepest Ascent** — the gradient points where the function increases the most; moving along it increases the function value as fast as possible.
2. **Magnitude of Change** — the length of the gradient vector tells us how steep the slope is. **Large gradient = function changing rapidly**; small gradient = changing slowly.
- On the cost parabola: left of the minimum → **negative derivative**; right → **positive derivative**.
- Because we want to **minimize**, we move **against** the gradient (minus sign in the update rule).

### Common Mistakes
- Saying the gradient points to the minimum — it points to the steepest **ascent**; GD uses its **negative**.
- Forgetting the gradient's magnitude naturally shrinks near the minimum (slope flattens), so **steps become smaller over time even with a fixed learning rate** (lecture "question for thought").

### Formula Explanation
Gradient = ∂𝒥/∂θ — slope of the cost with respect to the parameters.
- Negative slope → increase θ to reduce cost; positive slope → decrease θ.
- Common mistake: sign confusion — the minus in the update rule already handles direction.

### Diagram / Image Explanation
**Slide 18 (parabola with tangent lines):** negative derivative on the left branch, positive derivative on the right branch, minimum θ̂ at the bottom.
- *Exam question:* "What is the sign of the derivative left/right of the minimum, and what happens exactly at it?" → negative / positive / zero.

### Important Keywords
- **Gradient** — derivative w.r.t. inputs; direction + magnitude of steepest ascent.
- **Steepest ascent** — direction of maximal increase.
- **Derivative sign** — tells which way to move the parameter.

### Memory Tips
"**The Path and the Pace**" (lecture's own phrase): direction = path, magnitude = pace.

---

## Topic 7: Gradient Descent (GD)

### Definition
**Gradient descent is an iterative optimization algorithm used to minimize functions.** It calculates the **gradient (slope)** of the loss function and updates the weights **in the direction that reduces the loss**.

### Simple Example
Rolling a ball down the cost bowl step by step: start at a random point θ₀, take steps downhill proportional to the slope until you reach the bottom (convergence/minimum cost).

### Why is it Important?
GD is how ALL models in this lecture (linear regression, logistic regression, neural networks) are trained. It appears on ~15 slides.

### Key Points — **Gradient Descent Algorithm** (memorize!)
1. **Select a random initial value for the parameters (θ₀) and select a learning rate (α).**
2. **Repeat until convergence:** θⱼ ← θⱼ − α · ∂/∂θⱼ 𝒥(θ)  (j corresponds to model parameters)
- For 2+ parameters, repeat for **each** parameter: ω = ω − α ∂𝒥/∂ω and b = b − α ∂𝒥/∂b ("we repeat this for each weight in case we have multiple covariates/features").
- **Step size decreases over time** because the gradient magnitude shrinks near the minimum (flatter slope).
- **We still write θ̂ (hat) after convergence** because the result remains an *estimate* of the best parameter.
- **In convex problems (like linear regression with MSE), it is GUARANTEED that we move toward the global minimum since it is convex.** (Highlighted!)
- In non-convex functions, GD may end in a **local minimum** or **saddle point**.
- In NN version: stop at **convergence or maximum number of iterations reached**.

**Derivatives for linear regression (given in lecture; don't memorize derivation, know the shape):**
- ∂𝒥/∂b = −(1/M) Σ (y⁽ⁱ⁾ − (ωx⁽ⁱ⁾ + b))
- ∂𝒥/∂ω = −(1/M) Σ (y⁽ⁱ⁾ − (ωx⁽ⁱ⁾ + b)) · xᵀ⁽ⁱ⁾
- Obtained by applying the **chain rule** to the cost function.

### Benefits and Limitations of Gradient Descent (Homework summary slide)

| Benefits | Limitations |
|---|---|
| Easy to implement (simple iterative steps along negative gradient) | Slow convergence (entire dataset each iteration) |
| Scales with large parameter sets | Sensitive to learning rate (too large → overshoot, too small → very slow) |
| Efficient for convex problems (finds global minimum) | Can get stuck in local minima of **non-convex** functions |
| Simplicity → widely used | Computationally expensive for large datasets & models |

### Common Mistakes
- Forgetting the **minus** sign in the update rule.
- Believing GD always finds the global minimum — only guaranteed for **convex** functions.
- Confusing "learning rate" with "gradient": α is a fixed hyperparameter you choose; the gradient is computed.

### Formula Explanation

**Update rule: θⱼ ← θⱼ − α · (∂/∂θⱼ) 𝒥(θ)**
- **θⱼ** = j-th model parameter, **α** = learning rate, **∂𝒥/∂θⱼ** = gradient of cost w.r.t. that parameter.
- *Intuition:* move each parameter a small step against the slope.
- *When used:* every training iteration until convergence.
- *Interpretation:* big slope → big step; near minimum slope→0 → steps shrink → convergence.
- *Common mistake:* updating with **+** instead of **−**; forgetting to update **every** parameter j.

### Diagram / Image Explanation
**Slides 20–21 (GD steps on parabola):** θ₀ = random initial value (initial weight) on the right; arrows = **steps** hopping down the curve; bottom = **"Convergence or minimum cost"** at θ̂.
- *Exam question:* Why do steps get smaller? → gradient magnitude decreases as slope flattens near minimum.
**Slide 26 (3D bowl surface):** Cost function J(ω,b) with two parameters (a weight and a bias); blue circles = parameter trajectory moving toward the optimum (red circle). Caption: convex ⇒ guaranteed global minimum.
**Slide 74 (non-convex 1D loss):** P1 = **Local Min**, P2 = **Saddle Point**, P3 = **Global Min** — deep networks often face vanishing gradients and get stuck in saddle points.
- *Exam question:* Identify P1/P2/P3 on the curve.

### Important Keywords
- **Iterative optimization algorithm** — improves parameters step by step.
- **Convergence** — the point when updates stop improving the cost.
- **θ₀** — random initial parameter value.
- **Convex function** — bowl-shaped; GD guaranteed to find global minimum.
- **Local minimum / saddle point / global minimum** — types of critical points in non-convex landscapes.
- **Vanishing gradients** — gradients become too small in deep networks.

### Memory Tips
**"Init, Rate, Repeat"** — initialize randomly, pick learning rate, repeat update until convergence.
Update rule song: "**new θ = old θ minus α times slope**."

---

## Topic 8: Learning Rate (α) and Learning Rate Decay

### Definition
The **learning rate controls how much we update at each iteration** — it controls the steps toward the optimal point. **Learning rate decay** reduces α over time for fast learning at the start and stable steps near convergence.

### Simple Example
Walking down a hill: giant leaps (too high α) make you overshoot and bounce across the valley; baby steps (too low α) take forever; the right stride gets you down quickly and safely.

### Why is it Important?
"**Learning rate choice has a great impact on the convergence process**" — shown twice in the lecture (slides 30 & 95), a near-certain exam topic.

### Key Points
- **Unfortunately, there is no single rule to give the best learning rate α.**
- **Too high:** drastic updates, parameters "**slosh back and forth across the ravine**" (divergence/oscillation).
- **Too low:** too many updates before reaching the optimal point (very slow).
- **Just right:** smooth, efficient convergence.
- Learning rate is a **hyperparameter**.
- **Learning Rate Decay** — "a solution to the learning rate dilemma": reduce α over time → fast learning in the beginning, slower more stable steps as the algorithm converges. Helps prevent overshooting near the optimum.
- **Decay techniques (formulas given):**
  - **Step decay:** α = α₀ × factor^(epoch/decay step) — reduce α by a factor every set number of epochs.
  - **Exponential decay:** α = α₀ × e^(−decay rate × epoch).
  - **1/t decay:** α = α₀ / (1 + decay rate × epoch) — decrease by inverse of epoch number.

### Common Mistakes
- Thinking there is a universal best learning rate — there isn't.
- Confusing learning-rate-decay–induced smaller steps with the natural step shrinkage from smaller gradients (both happen!).
- Mixing up the three decay formulas (step = factor powers; exponential = e-power; 1/t = division).

### Comparison

| Learning rate too LOW | Just right | Too HIGH |
|---|---|---|
| Many tiny steps | Smooth descent | Oscillation across ravine |
| Very slow convergence | Efficient convergence | May diverge / never converge |

### Formula Explanation
- **Step decay: α = α₀ × factor^(epoch/decay step)** — α₀ = initial rate; drops in discrete stairs.
- **Exponential decay: α = α₀ × e^(−decay rate × epoch)** — smooth exponential curve.
- **1/t decay: α = α₀ / (1 + decay rate × epoch)** — hyperbolic decrease.
- *Interpretation (plot):* step decay = staircase; exponential = fastest smooth drop; 1/t = smooth but slower than exponential later.

### Diagram / Image Explanation
**Slide 30/95 (three parabolas):** Too low = many tiny red arrows; Just right = few decisive arrows to the minimum; Too high = arrows zig-zag wildly across the bowl.
- *Exam question:* Given a picture of oscillating updates, diagnose the problem → learning rate too high.
**Slide 96 (decay schedules plot):** learning rate vs. epochs for Step Decay (blue staircase), Exponential Decay (red dashed), 1/t Decay (green dash-dot).

### Important Keywords
- **Learning rate (α)** — step size hyperparameter.
- **Learning rate schedule / decay** — plan for reducing α over epochs.
- **Step / exponential / 1/t decay** — the three decay techniques.
- **Overshooting** — jumping past the minimum when α too big.

### Memory Tips
**"Goldilocks α"**: too hot (high) bounces, too cold (low) crawls, just right converges.
Decay trio: **S-E-T** → Step, Exponential, one-over-**T**.

---

## Topic 9: Other Types of Linear Regression (Polynomial, Ridge, Lasso, Elastic Net)

### Definition
Extensions of linear regression: **polynomial regression** adds polynomial feature terms to capture non-linear relationships; **Ridge (L2)** and **Lasso (L1)** add penalty terms to address multicollinearity and overfitting; **Elastic Net** combines both.

### Simple Example
Modeling crop yield vs. temperature (rises then falls): a straight line fails, but adding x² (polynomial regression) captures the curve.

### Why is it Important?
Shows regularization vocabulary (L1/L2) reused later in NN training tricks; classic comparison exam question.

### Key Points
- **Polynomial Regression:** extends linear regression with polynomial terms (x², x³); captures non-linear relationships; **still a linear model in parameters**; e.g., Y = w₀ + w₁x + w₂x² + ε.
- **Ridge (L2 Regularization):** shrinks coefficients by adding penalty **λΣwⱼ²**.
- **Lasso (L1 Regularization):** can shrink some coefficients **to zero** → **feature selection**.
- Both Ridge & Lasso address **multicollinearity** and **overfitting**.
- **Elastic Net:** combines Lasso and Ridge; useful when predictors are highly correlated.

### Common Mistakes
- Saying polynomial regression is non-linear — it is non-linear in **x** but **linear in the parameters**.
- Swapping L1/L2: **L1 = Lasso = zeros/selection**; **L2 = Ridge = shrinks**.

### Comparison

| | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Penalty | L2: λΣwⱼ² | L1: λΣ|wⱼ| | L1 + L2 combined |
| Coefficients | Shrunk toward 0 | Some exactly **0** | Both effects |
| Feature selection | No | **Yes** | Partial |
| Best when | Multicollinearity | Sparse features | Highly correlated predictors |

### Formula Explanation
**Y = w₀ + w₁x + w₂x² + ε** — polynomial regression: w's are still linear coefficients; ε = noise term.
**Ridge penalty λΣwⱼ²** — λ controls penalty strength; large weights are punished quadratically.

### Diagram / Image Explanation
**Slide 32 plot:** polynomial regression fit with Truth / Estimate / CB (confidence band) curves through scattered points — shows a smooth non-linear fit.

### Important Keywords
- **Polynomial regression** — linear regression + polynomial terms.
- **Ridge (L2)** — squared-weight penalty, shrinks coefficients.
- **Lasso (L1)** — absolute-weight penalty, zeroes coefficients (feature selection).
- **Elastic Net** — Lasso + Ridge combination.
- **Regularization** — penalizing weights to prevent overfitting.

### Memory Tips
**"LassO gives zerO"** (L1 → 0 coefficients). Ridge has a "2" sound → L**2** squared penalty.

### Applications of Linear Regression (slide 33 — know a few!)
Real Estate (house prices), Salary Estimation, Sales Forecasting, Risk Assessment (insurance claims), Agriculture (crop yield), Finance & Investment (stock returns), Education Analytics (student performance).

---

## Topic 10: Logistic Regression & the Sigmoid Function

### Definition
**Logistic regression** converts the linear model into a binary classifier by passing the weighted sum z = wᵀx + b through the **sigmoid function**, producing a probability in (0,1); a **threshold (quantizer)** converts the probability into a binary prediction {0,1}.

### Simple Example
**Breast cancer prediction** (lecture's running example): features extracted from pathological images (age, radius_mean, texture_mean…) → probability of malignant (1) vs. benign (0).

### Why is it Important?
Logistic regression = one sigmoid neuron; it is the exact building block ("yellow box") of neural networks.

### Key Points — Notation (slide 35, know it!)
- Data point: **x ∈ Rᴺ** — single point with N features, x = [x₁,…,x_N]ᵀ.
- Dataset: **X ∈ Rᴹˣᴺ** — matrix, each row a data point x⁽ⁱ⁾, M total points.
- True binary label: **y ∈ {0,1}**; labels vector **y = [y⁽¹⁾,…,y⁽ᴹ⁾]ᵀ**.
- Weights: **w ∈ Rᴺ** — a vector of **trainable** weights for the N features.
- Output: **ŷ ∈ (0,1)** — predicted probability for a single data point; **ŷ ∈ Rᴹ** for all points.

### Pipeline (memorize the order!)
**x → weighted function z = Σ(wᵢxᵢ)+b = wᵀx+b → activation function Sigmoid σ → ŷ (probability) → Quantizer (thresholding of the probability) → binary estimate {0,1}**
- "We do this with a specific **threshold** to convert probabilities to predictions; **the threshold can be adjusted**."

### Common Mistakes
- Calling logistic regression a regression for continuous values — despite the name it's a **classifier**.
- Forgetting the **quantizer/threshold** step after sigmoid.
- Saying sigmoid outputs {0,1} — it outputs a **probability in (0,1)**; the threshold makes it binary.

### Comparison

| Linear Regression | Logistic Regression |
|---|---|
| Predicts continuous value | Predicts probability → class {0,1} |
| Activation: identity (1) | Activation: **sigmoid σ** |
| Cost: MSE | Cost: **cross-entropy** |
| Output ∈ R | Output ∈ (0,1) |
| No thresholding | Quantizer/threshold step |

### Formula Explanation

**σ(z) = 1 / (1 + e^(−z))**, with **z = wᵀx + b**, so **ŷ = σ(z) = 1/(1+e^(−(wᵀx+b)))**
- **z** = weighted sum (logit); **e** = Euler's number.
- *Intuition:* squashes any real number into (0,1); z=0 → 0.5; large positive z → ≈1; large negative → ≈0.
- *When used:* binary classification output; interpreting output as **probability of one of two alternatives (0,1)**.
- *Interpretation:* ŷ = probability of class 1 (malignant); 1−ŷ = probability of class 0 (benign).
- *Common mistakes:* wrong sign in exponent; thinking output can be exactly 0 or 1 (it's an open interval).

### Diagram / Image Explanation
**Slide 40 (sigmoid curve):** S-shaped curve from 0 to 1, crossing **0.5 at z = 0**, x-range −6…6. "Logistic sigmoid function commonly referred to as just Sigmoid function. Its output can be interpreted as a probability of one of two alternatives (0,1)."
- *Exam question:* value of σ(0)? → 0.5. Range? → (0,1).
**Slides 36–39 (logistic regression as network):** inputs (bias node "1", Age x₀, Radius_mean x₁, Texture_mean x_N) → weights → Σ → sigmoid → ŷ → quantizer → binary estimates.
- *Common mistake:* omitting the bias node "1".

### Important Keywords
- **Sigmoid σ** — S-shaped squashing function to (0,1).
- **z (weighted function)** — wᵀx + b.
- **Quantizer** — thresholds the probability into {0,1}.
- **Threshold** — adjustable cut-off probability.
- **Binary label y ∈ {0,1}** — e.g., benign 0 / malignant 1.

### Memory Tips
Pipeline: "**Sum, Squash, Snap**" — sum (z), squash (sigmoid), snap to 0/1 (quantizer).
σ(0)=0.5 → "**zero gives fifty-fifty**."

---

## Topic 11: Cross-Entropy Cost Function

### Definition
The **cross-entropy cost function** measures classification error by penalizing confident wrong probability predictions using logarithms. For binary problems: sum of entropy for predicting class 1 plus entropy for predicting class 0.

### Simple Example
**Fruit baskets (lecture example):** minimizing cross-entropy of a strawberries/apples mix = minimize entropy of predicting strawberries + minimize entropy of predicting apples.

### Why is it Important?
It is THE cost function for logistic regression AND for neural networks in this lecture (reused on slide 70).

### Key Points
- Binary: **𝒥(ω,b) = −(1/M) Σᵢ₌₁ᴹ (y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1−y⁽ⁱ⁾) log(1−ŷ⁽ⁱ⁾))**
- y⁽ⁱ⁾ = **true label (0 or 1)** for the i-th data point; ŷ⁽ⁱ⁾ = **predicted probability** from the model.
- Only one of the two terms is active per point (y=1 → first term; y=0 → second term).
- The log function (binary log shown in lecture plot) → log of small probability = large negative → big penalty for confident wrong answers.
- **Multinomial cross-entropy:** **𝒥(ω,b) = −(1/M) Σᵢ Σₖ (yᵢ<k> log(ŷᵢ<k>))** — yᵢ<k> = **true probability distribution**, ŷᵢ<k> = **model's predicted probability** over K classes.

### Common Mistakes
- Forgetting the leading **minus** sign.
- Using MSE for classification instead of cross-entropy.
- Confusing y (true label) with ŷ (predicted probability).

### Comparison

| MSE | Cross-Entropy |
|---|---|
| Regression | Classification |
| Squared distance | Log-probability penalty |
| (1/2M)Σ(y−ŷ)² | −(1/M)Σ(y log ŷ + (1−y)log(1−ŷ)) |
| Continuous targets | Probability outputs |

### Formula Explanation

**𝒥(ω,b) = −(1/M) Σ (y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1−y⁽ⁱ⁾) log(1−ŷ⁽ⁱ⁾))**
- Variables: M = # data points; y = true label 0/1; ŷ = predicted probability.
- *Intuition:* when y=1 we want ŷ→1 (log ŷ→0, no penalty); when the model is confident and wrong, log→−∞ → huge cost.
- *When used:* training logistic regression and binary NN classifiers.
- *Interpretation:* smaller = better calibrated, more accurate probabilities.
- *Common mistake:* dropping the (1−y)log(1−ŷ) term — you need BOTH classes' entropies (basket picture!).

### Diagram / Image Explanation
**Slide 41 (baskets + log plot):** Mixed basket → strawberries basket + apples basket illustrates decomposing cross entropy into per-class entropies; the log₂(x) plot explains why probabilities near 0 are heavily punished.
- *Exam question:* Explain the two terms of binary cross entropy using the basket analogy.

### Important Keywords
- **Cross-entropy** — log-based classification cost.
- **Binary log function** — log₂ shown in the lecture plot.
- **True probability distribution** — one-hot true labels in multinomial CE.
- **One-hot** — encoding where the true class = 1, others = 0 (appears in the SoftMax regression figure).

### Memory Tips
"**Minus, Mean, Mix of two logs**" — the 3 M's of binary cross-entropy.

---

## Topic 12: Gradient Descent for Logistic Regression (Chain Rule)

### Definition
To train logistic regression with GD, the gradient of the cross-entropy cost with respect to the weights is found by **applying the chain rule** through the computation chain x → z → ŷ → 𝒥.

### Simple Example
Like peeling an onion: differentiate cost w.r.t. ŷ, then ŷ w.r.t. z (sigmoid derivative), then z w.r.t. ω, and multiply the three.

### Why is it Important?
This exact chain-rule mechanic is what becomes **backpropagation** in neural networks.

### Key Points (the three links — given in lecture)
- **∂𝒥/∂ω = (∂𝒥/∂ŷ) · (∂ŷ/∂z) · (∂z/∂ω)**  ← chain rule
- Derivative of cost w.r.t. ŷ: **∂𝒥/∂ŷ = −y/ŷ + (1−y)/(1−ŷ)**
- **Sigmoid derivative:** **∂ŷ/∂z = ŷ(1−ŷ)** (also written σ′(x) = σ(x)(1−σ(x)))
- Derivative of z w.r.t. ω: **∂z/∂ω = x** (or xᵀ)
- **Combined result:** **∂𝒥/∂ω = (ŷ − y)·xᵀ**
- Over all samples: **∂𝒥/∂ω = (1/M) Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾) x⁽ⁱ⁾ᵀ**
- **Update rule:** **ω := ω − α · (1/M) Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾) x⁽ⁱ⁾ᵀ**

### Common Mistakes
- Forgetting the sigmoid derivative shortcut **σ′ = σ(1−σ)** — very likely exam formula!
- Writing the combined gradient as (y − ŷ)x instead of **(ŷ − y)x** (sign flip).
- Skipping a link in the chain.

### Formula Explanation
**σ′(x) = σ(x)·(1 − σ(x))**
- *Intuition:* the sigmoid's slope is maximal at z=0 (0.25) and → 0 at the tails — this is why sigmoid can cause **vanishing gradients**.
- *When used:* every backprop step through a sigmoid.
**(ŷ − y)·xᵀ** — elegant final gradient: error times input.
- *Interpretation:* the weight change is proportional to how wrong we were (ŷ−y) and to the input that caused it (x).

### Diagram / Image Explanation
**Slides 42–44 (computation chain):** x → [z = wᵀx + b | σ(z)] → ŷ, with the three derivative boxes and the green arrow combining them.
- *Exam question:* Write the chain rule for ∂𝒥/∂ω and name each factor.

### Important Keywords
- **Chain rule** — decompose derivative through intermediate variables.
- **Sigmoid derivative** — ŷ(1−ŷ).
- **Combined gradient** — (ŷ−y)xᵀ.

### Memory Tips
Chain: "**Cost→ŷ→z→ω**" — remember "**CYZW**" path.
Final gradient = "**error × input**."

---

## Topic 13: Multinomial (Multilabel) Classification — One-vs-Rest & SoftMax

### Definition
When there are **more than two classes**, we either train **one separate logistic regression per class (one-vs-rest)** or use the **SoftMax function** to fit a single **SoftMax regression** model outputting a probability distribution over all classes.

### Simple Example
Classifying fruit into 3 shapes (stars/circles/squares in lecture figure): one-vs-rest builds 3 binary boundaries (stars vs. rest, etc.); SoftMax gives e.g. [0.02, 0.90, 0.05, 0.01, 0.02] over 5 classes summing to 1.

### Why is it Important?
Answers "What if we have more than two classes?" (lecture quiz question) and explains multi-output NN layers.

### Key Points
- **1. One vs. others (one-vs-rest):** "a separate logistic regression model for each class."
- **2. SoftMax regression:** **SoftMax = normalized exponential function** (probabilities of different classes); **maps a vector of continuous values to a vector of probabilities summing to 1**.
- **softmax(zᵢ) = e^(zᵢ) / Σⱼ₌₁ᴷ e^(zⱼ)**
- Lecture worked example: input [1.30, 5.10, 2.20, 0.70, 1.10] → output [0.02, 0.90, 0.05, 0.01, 0.02].
- SoftMax regression uses **cross-entropy** with **one-hot true labels**, and **argmax** picks the final class.

### Common Mistakes
- Using sigmoid for multi-class mutually exclusive problems instead of SoftMax.
- Forgetting SoftMax outputs **sum to 1**.
- Thinking one-vs-rest needs one model total — it needs **K models** for K classes.

### Comparison

| One-vs-Rest | SoftMax Regression |
|---|---|
| K separate binary logistic models | One single model |
| Each: class k vs. all others | Joint probability distribution |
| Sigmoid per model | SoftMax over K outputs |
| Probabilities don't sum to 1 | Probabilities **sum to 1** |

| Sigmoid | SoftMax |
|---|---|
| 2 classes (binary) | K classes (multinomial) |
| Single probability | Probability vector |
| 1/(1+e^(−z)) | e^(zᵢ)/Σe^(zⱼ) |

### Formula Explanation
**softmax(zᵢ) = e^(zᵢ) / Σⱼ₌₁ᴷ e^(zⱼ)**
- **zᵢ** = score for class i; **K** = number of classes.
- *Intuition:* exponentiate (all positive, big scores amplified) then normalize (divide by sum → probabilities).
- *Interpretation:* largest z gets the largest probability (0.90 for 5.10 in the example).
- *Common mistake:* forgetting the denominator sums over **all** classes.

### Diagram / Image Explanation
**Slide 46 (one-vs-rest):** one scatter of 3 classes fanning into 3 binary-boundary plots.
**Slide 47 (SoftMax regression architecture):** inputs → weights wₘ,ₖ → K summing junctions → SOFTMAX FUNCTION → one-hot targets t₁…tₖ → argmax → y; trained with Cross-Entropy and One-Hot True Labels.
- *Exam question:* Order the SoftMax pipeline components.

### Important Keywords
- **One-vs-rest** — one binary classifier per class.
- **SoftMax** — normalized exponential turning scores into a probability distribution.
- **One-hot labels** — target encoding for multinomial CE.
- **argmax** — pick the class with the highest probability.

### Memory Tips
"**Soft**Max **soft**ly spreads probability; the winner still takes **max**."

---

## Topic 14: Limitations of Logistic Regression (Why do we need a Neural Network?)

### Definition
Six weaknesses of logistic regression that motivate neural networks, chiefly its restriction to **linear decision boundaries**.

### Simple Example
Image classification: pixel relationships are complex and non-linear, so a single linear boundary in pixel space performs poorly — a NN with hidden layers is needed.

### Why is it Important?
This slide is the pivot of the entire lecture — "Why do we need a neural network then?" is a near-guaranteed exam question.

### Key Points (memorize the 6!)
1. **Linearity of Decision Boundaries** — only finds linear boundaries; limits performance when data is not linearly separable.
2. **Limited Feature Interaction** — does not inherently capture complex interactions between features unless manually engineered.
3. **Capacity for Complexity** — works for simple tasks but struggles with higher complexity (image or audio classification).
4. **Sensitivity to Outliers and Noise** — outliers disproportionately influence the cost function.
5. **Scalability with Large Datasets** — does not scale as effectively to large and high-dimensional data.
6. **Inherent Limitations in Multinomial Classification** — primarily binary; extending (one-vs-all, SoftMax) can be less efficient.

### Common Mistakes
- Claiming logistic regression can't do multiclass at all — it can (OvR/SoftMax), just **less efficiently**.
- Forgetting outlier sensitivity.

### Important Keywords
- **Linear decision boundary** — a straight line/hyperplane separating classes.
- **Linearly separable** — classes divisible by a linear boundary.
- **Feature interaction** — combined effects of multiple features.

### Memory Tips
**"L-I-C-S-S-M"**: **L**inear boundaries, feature **I**nteraction, **C**apacity, **S**ensitivity to outliers, **S**calability, **M**ultinomial limits.

---

## Topic 15: Neural Networks — Hidden Layers & Architecture

### Definition
A **neural network** adds one or more **hidden layers** of neurons between input and output to model complex relationships. **Bottomline (lecture): a neural network can be explained as several logistic regression blocks — with some caveats and extensions; we call these blocks neurons.**

### Simple Example
Breast cancer example: inputs Age=34, Radius_mean=2, Texture_mean=4 pass through 2 hidden sigmoid neurons (outputs .4 and .2) into an output neuron → 0.6 = **probability of cancer**.

### Why is it Important?
Hidden layers give NNs the power to overcome logistic regression's linear-boundary limitation.

### Key Points
- Layer roles: **Input layer** (input features / independent variables) → **Hidden layer(s)** → **Output layer** (prediction / "dependent variable").
- The **yellow box** around input+one neuron in the lecture figure **corresponds to a logistic regression**.
- **More output neurons** → multilabel classification (outputs y₁…y_K).
- **More layers** → deeper model for more complex patterns (hidden layers 1, 2, 3 with activations a, b, c).
- **Why go deep?** Creating a series of **complex transformations** from input to output → the algorithm learns to extract **hierarchical features** that help prediction (edges → parts → whole animals in the elephant/kangaroo/penguin figure; "increasingly complex features").

### Common Mistakes
- Counting the input layer as a "processing" layer.
- Thinking deeper is always better (→ overfitting, computational cost — see network size topic).

### Comparison

| Single Perceptron / Logistic Regression | Neural Network |
|---|---|
| No hidden layer | ≥1 hidden layer |
| Linear decision boundary | Non-linear boundaries |
| Hand-engineered feature interactions | Learns hierarchical features |
| Few parameters | Many parameters |

### Diagram / Image Explanation
**Slides 50–51 (adding a hidden layer):** input neurons (34, 2, 4) with weights (.6, .1, .7, .2, .3, .2) → two hidden sigmoid neurons (.4, .2) → weights (.5, .8) → output neuron → 0.6 probability of cancer. Yellow box = logistic regression block.
- *Exam question:* "What does the yellow box correspond to?" → a logistic regression.
**Slide 52:** multiple output neurons y₁…y_K for multilabel classification (input x₁…x_N, hidden a₁…a_D).
**Slides 53–54:** 3 hidden layers (a, b, c) then output — deeper for more complex patterns.
**Slide 55 (hierarchical features image):** photos → simple strokes → parts → animal shapes → labels (Penguin/Elephant/Kangaroo); sides labeled unsupervised/supervised learning, "increasingly complex features."
- *Exam question:* "What does each successive layer learn?" → increasingly complex/hierarchical features.

### Important Keywords
- **Hidden layer** — intermediate layer between input and output.
- **Neuron** — a logistic-regression-like block within the network.
- **Hierarchical features** — increasingly complex features learned layer by layer.
- **Multilabel classification** — multiple output neurons.

### Memory Tips
"**NN = LEGO of logistic regressions**" — stack the same block to build complexity.

---

## Topic 16: Forward Propagation

### Definition
**Forward propagation** is the pass of the input through the network layer by layer — computing each layer's weighted sum z and activation a — until the prediction ŷ is produced and the **loss function is computed** against the ground truth y.

### Simple Example
Cancer example: (Age, Radius_mean, Texture_mean) → hidden layer computes z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾, a⁽¹⁾ = g⁽¹⁾(z⁽¹⁾) → output layer computes z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾, a⁽²⁾ = g⁽²⁾(z⁽²⁾) = ŷ = probability of cancer → compare with true label y in the loss.

### Why is it Important?
It is step ① of NN training; backprop (step ②) cannot happen without it.

### Key Points — Notation (slide 68 "Let's put it all together" — memorize!)
- **ℓ** = number of layers.
- **n⁽ⁱ⁾** = number of units in layer i; **n⁽⁰⁾** = units in layer 0 (input); **n⁽ˡ⁾** = units in the output layer = number of outputs (predictions).
- **a⁽ⁱ⁾** = activations in layer i; **ŷ = a⁽ˡ⁾**.
- **g⁽ⁱ⁾** = activation function in layer i.
- **W⁽ⁱ⁾, b⁽ⁱ⁾** = the parameters of layer i.
- **z⁽ⁱ⁾ = W⁽ⁱ⁾a⁽ⁱ⁻¹⁾ + b⁽ⁱ⁾** and **a⁽ⁱ⁾ = g⁽ⁱ⁾(z⁽ⁱ⁾)**.
- Full 2-layer forward pass (slide 57): **ŷ = g⁽²⁾(W⁽²⁾ g⁽¹⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾)** — nested composition: z⁽¹⁾ → a⁽¹⁾ → z⁽²⁾ → a⁽²⁾.
- Loss: **J(W⁽¹⁾, W⁽²⁾, b⁽¹⁾, b⁽²⁾) = ℒ(ŷ, y)** — the cost depends on ALL parameters.
- **As NNs grow:** we move from **scalar operations** (single input/output) → **matrix operations** (many neurons at once); we need **gradients (∇)** to update weights via backpropagation; we use **capital letters** for vectors/matrices for compact, scalable notation.
- Dimension example (slide 69): l = 4; a⁽³⁾ ∈ R⁴; b⁽¹⁾ ∈ R³; W⁽²⁾ ∈ R⁵ˣ³; ∇_b⁽¹⁾J ∈ R³; ∇_W⁽²⁾J ∈ R⁵ˣ³ — **gradients have the same shape as the parameters they update**.

### Common Mistakes
- Writing z⁽ⁱ⁾ = W⁽ⁱ⁾x + b⁽ⁱ⁾ for hidden layers > 1 — it takes the **previous layer's activation** a⁽ⁱ⁻¹⁾, not x.
- Forgetting ŷ = a⁽ˡ⁾ (output activation IS the prediction).
- Mismatching gradient dimensions with parameter dimensions.

### Formula Explanation
**z⁽ⁱ⁾ = W⁽ⁱ⁾a⁽ⁱ⁻¹⁾ + b⁽ⁱ⁾; a⁽ⁱ⁾ = g⁽ⁱ⁾(z⁽ⁱ⁾)**
- The universal layer recipe: linear transform + non-linear activation, repeated.
- *Interpretation:* each layer re-represents the previous layer's output.
**ŷ = g⁽²⁾(W⁽²⁾ g⁽¹⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾)**
- *Intuition:* the whole network is one nested function; this nesting is why the **chain rule** is needed for gradients.

### Diagram / Image Explanation
**Slides 56–57 (forward propagation diagram):** inputs (34, 2, 4) with bias nodes (1); per-layer weights W⁽¹⁾,b⁽¹⁾ (elements w₀₀⁽¹⁾ etc.) and W⁽²⁾,b⁽²⁾; hidden activations .4/.2; output ŷ = probability of cancer; then "Compute loss function" with ground truth y.
- *Exam question:* Given the diagram, write the equations of the forward pass in order.
**Slide 70:** green→blue→red network with arrow "① Forward propagation" + cross-entropy cost function.

### Important Keywords
- **Forward propagation** — input→output computation pass.
- **a⁽ⁱ⁾ / z⁽ⁱ⁾ / g⁽ⁱ⁾ / W⁽ⁱ⁾, b⁽ⁱ⁾** — activation, pre-activation, activation function, parameters of layer i.
- **Ground truth labels** — the true y ("dependent variable").
- **Matrix operations / capital letters** — vectorized notation for many neurons at once.

### Memory Tips
Layer mantra: "**z then a, z then a…**" until ŷ.
"**Hat comes out of the last layer**": ŷ = a⁽ˡ⁾.

---

## Topic 17: Backward Propagation (Backpropagation)

### Definition
**Backpropagation** computes the gradients of the cost with respect to **every** weight and bias in the network by applying the **chain rule** backwards from output to input, then updates all parameters with gradient descent.

### Simple Example
After forward pass gives ŷ and loss, the error signal travels backwards: output layer parameters (W⁽²⁾, b⁽²⁾) get their gradients first, then hidden layer parameters (W⁽¹⁾, b⁽¹⁾) via the chain rule.

### Why is it Important?
It is step ② of NN training — the mechanism that makes deep learning possible.

### Key Points
- **Update steps (all four parameters):**
  - W⁽²⁾ := W⁽²⁾ − α∇_W⁽²⁾ J(W⁽¹⁾,W⁽²⁾,b⁽¹⁾,b⁽²⁾)
  - b⁽²⁾ := b⁽²⁾ − α∇_b⁽²⁾ J(…)
  - W⁽¹⁾ := W⁽¹⁾ − α∇_W⁽¹⁾ J(…)
  - b⁽¹⁾ := b⁽¹⁾ − α∇_b⁽¹⁾ J(…)
  (η is used interchangeably with α for the learning rate on slide 60.)
- **To find the partial derivatives we use the chain rule.**
- Full chain (slide 72): **∂𝒥/∂ω⁽¹⁾ = (∂𝒥/∂ŷ)·(∂ŷ/∂g⁽ˡ⁾)·(∂g⁽ˡ⁾/∂z⁽ˡ⁾)·(∂g⁽ˡ⁻¹⁾/∂z⁽ˡ⁻¹⁾)·…·(∂z⁽¹⁾/∂ω⁽¹⁾)** — multiply local derivatives through every layer back to the first weights.
- **Local gradient concept (slide 60 figure):** at each node f with inputs x,y and output z: downstream gradient ∂L/∂x = ∂L/∂z · ∂z/∂x — each node multiplies the incoming gradient by its local gradient.
- **Worked example (slide 61):** ∇_z⁽²⁾J = a⁽²⁾ − ŷ; ∇_W⁽²⁾J = ∇_z⁽²⁾J · ∇_W⁽²⁾z⁽²⁾ = **(a⁽²⁾ − ŷ)·a⁽¹⁾ᵀ**; uses ∂ℒ/∂ŷ = (ŷ−y)/((1−ŷ)ŷ) and σ′(x) = σ(x)(1−σ(x)).
- GD on the NN cost: choose learning rate + random init θ:(ω,b); repeat **until convergence or maximum number of iterations reached**: θⱼ ← θⱼ − α ∂𝒥/∂θⱼ.

### Common Mistakes
- Updating only the output layer — **every** layer's W and b are updated.
- Applying chain rule in forward order — gradients flow **backwards**.
- Forgetting that all update steps use the SAME cost J (which depends on all parameters).

### Comparison

| Forward Propagation | Backpropagation |
|---|---|
| Input → Output | Output → Input |
| Computes z, a, ŷ, loss | Computes gradients ∇ |
| Uses W, b values | Updates W, b values |
| Function composition | Chain rule decomposition |

### Formula Explanation
**θⱼ ← θⱼ − α ∂𝒥/∂θⱼ (for every layer's W and b)** — the same GD rule scaled to all NN parameters.
**∂L/∂x = ∂L/∂z · ∂z/∂x** ("local gradient") — the building block of backprop: multiply upstream gradient by the node's local derivative.
- *Common mistake:* mixing which factor is "local" (node's own derivative) vs. upstream (from the loss side).

### Diagram / Image Explanation
**Slide 59 (backprop over the cancer network):** same forward diagram plus pink "Update steps" box with the four update equations.
**Slide 60 (local gradient node):** circle f with inputs x, y and output z; green arrows forward, red arrows backward carrying ∂L/∂z, ∂L/∂x = ∂L/∂z·∂z/∂x, ∂L/∂y.
**Slide 72:** network with backward arrow "② Backpropagation" and the long chain-rule product.
- *Exam question:* Order training: forward propagation → compute cost → backpropagation → update parameters → repeat.

### Important Keywords
- **Backpropagation** — computing all parameter gradients via the chain rule backwards.
- **∇ (nabla / gradient)** — vector/matrix of partial derivatives.
- **Local gradient** — a node's own derivative multiplied with upstream gradients.
- **Update steps** — one GD update per parameter tensor.

### Memory Tips
"**Forward computes, backward corrects.**"
Training loop: **F-C-B-U** (Forward, Cost, Backward, Update).

---

## Topic 18: Choosing the Network Size (Hyperparameter)

### Definition
**Network size** refers to the **number of layers and neurons per layer** in a neural network. It is **fixed before training** and remains constant throughout — making it a **hyperparameter**.

### Simple Example
For simple tabular data, a small network with one hidden layer may suffice; for image classification, a much deeper, larger network is needed.

### Why is it Important?
The green "Remember" box (**"Network size is a hyperparameter that should be chosen and fine tuned carefully"**) marks this as exam-critical.

### Key Points
- **Fixed Before Training** — chosen before training starts, constant throughout ⇒ hyperparameter.
- **Influences Capacity** — size directly impacts the network's capacity to learn patterns; larger networks capture more complex relationships.
- **Balancing Complexity** — **too many layers/neurons → overfitting; too few → underfitting**.
- **Computational Cost** — larger networks increase computational demands (training time, resources).

### Tips for Choosing the Right Network Size (slide 64 — memorize the 5)
1. **Start Small** — easier to add layers/neurons than to remove them.
2. **Gradually Increase Complexity** — add incrementally **based on validation performance**.
3. **Consider Task Complexity** — simpler task → smaller network; complex → larger.
4. **Experiment with Dropout** — for large networks, dropout reduces overfitting, improves generalization.
5. **Use Early Stopping** — halt training when performance plateaus to prevent overfitting.
- **Remember:** network size is not the only hyperparameter — also consider **regularization**, **early stopping**, and the **activation functions**.

### Common Mistakes
- Calling network size a learned parameter — it is a **hyper**parameter (not learned by GD).
- "Bigger is always better" — bigger risks overfitting and cost.

### Comparison

| Too small network | Too large network |
|---|---|
| **Underfitting** | **Overfitting** |
| Can't capture patterns | Memorizes training data |
| Cheap to train | Computationally expensive |

| Parameter | Hyperparameter |
|---|---|
| Learned during training (W, b) | Set **before** training |
| Updated by gradient descent | Tuned by experimentation/validation |
| e.g., weights, biases | e.g., network size, learning rate, activation function |

### Important Keywords
- **Network size** — # layers + # neurons per layer.
- **Hyperparameter** — setting fixed before training.
- **Capacity** — ability to learn complex patterns.
- **Underfitting / Overfitting** — too little / too much capacity.
- **Dropout, Early stopping** — remedies for large-network overfitting.

### Memory Tips
Sizing rule: "**Start small, grow with validation.**"

---

## Topic 19: Activation Functions

### Definition
**Activation functions (AF)**, inspired by biological **action potentials**, determine whether a neuron should be **activated ("fired")** based on whether the weighted input meets a threshold. Mathematically they add the **non-linearity** the network needs to learn complex patterns.

### Simple Example
ReLU max(0, x): a neuron outputs its input if positive, otherwise 0 — like a neuron that only fires above threshold.

### Why is it Important?
"**The choice of AF directly influences the network's ability to learn complex patterns, stabilize training, and capture non-linear relationships**." AF is a **hyperparameter** — "Remember" box.

### Key Points — Characteristics of activation functions (memorize the 5!)
1. **Non-Linearity** — adds complexity; learn beyond simple linear relationships.
2. **Differentiability** — enables gradient calculation, essential for learning via backpropagation.
3. **Bounded Output** — controls output range, preventing extreme values that destabilize training.
4. **Sparsity** — focuses the network on essential features by setting some neuron outputs to zero, improving efficiency.
5. **Computational Efficiency** — reduces computation time; faster, more scalable training/inference.

### The six activation functions listed (slide 67 — know formulas!)
| Function | Formula |
|---|---|
| **Sigmoid** | σ(x) = 1/(1+e^(−x)) |
| **tanh** | tanh(x) |
| **ReLU** | max(0, x) |
| **Leaky ReLU** | max(0.1x, x) |
| **Maxout** | max(w₁ᵀx + b₁, w₂ᵀx + b₂) |
| **ELU** | x if x ≥ 0; α(eˣ − 1) if x < 0 |

- Any of these can replace σ(zᵢ) in the neuron.
- **Remember:** "Activation function is a hyperparameter that needs experimentation and fine tuning, much like learning rates and layer sizes."

### Common Mistakes
- Omitting the activation → network collapses to a linear model regardless of depth.
- Confusing Leaky ReLU (0.1x for negatives) with ReLU (0 for negatives).
- Forgetting sigmoid is bounded (0,1) while ReLU is unbounded above.

### Comparison

| Sigmoid | ReLU |
|---|---|
| (0,1) bounded | [0, ∞) unbounded above |
| Smooth, saturates (vanishing gradients) | No saturation for x>0 |
| Probabilistic output layer | Standard hidden-layer choice |
| More computation (exp) | Very cheap (max) |

### Diagram / Image Explanation
**Slide 66 (action potential curve):** voltage vs. time; labels Threshold (−55 mV), Resting state (−70 mV), Depolarization, Repolarization, Failed initiations, Refractory period, Stimulus — biological inspiration for "fire or not."
- *Exam question:* What biological concept inspired activation functions? → the action potential/firing threshold.
**Slide 67 (six AF plots):** shapes of Sigmoid (S in (0,1)), tanh (S in (−1,1)), ReLU (hinge at 0), Leaky ReLU (slight negative slope), Maxout, ELU (smooth negative saturation).

### Important Keywords
- **Action potential** — biological neuron firing event.
- **Non-linearity** — key property enabling complex learning.
- **Differentiability** — needed for backprop.
- **Sparsity** — zeros in outputs (ReLU family).
- **ReLU / Leaky ReLU / ELU / tanh / Maxout / Sigmoid** — the lecture's six AFs.

### Memory Tips
Characteristics: **"No BS DC"** → **N**on-linearity, **B**ounded, **S**parsity, **D**ifferentiability, **C**omputational efficiency.
ReLU = "**Re**ject negatives, **L**et positives through **U**nchanged."

---

## Topic 20: Training Difficulties of Deep Models

### Definition
Deep networks are hard to train because of **optimization difficulty** (vanishing gradients, saddle points, local minima), **overfitting** (high-capacity models memorize training data), and **training instability** (exploding or vanishing gradients).

### Simple Example
Temperature-vs-date data (lecture figure): a huge model traces every noisy point (yellow jagged line = overfit); a good model follows the seasonal curve (red); an underfit model is too smooth/shifted (green).

### Why is it Important?
Motivates the entire "box of tricks" — likely scenario-question material.

### Key Points
- **Optimization Difficulty:** deep networks often face **vanishing gradients** and get stuck in **saddle points** (P1 local min, P2 saddle point, P3 global min figure).
- **Overfitting:** "**High-capacity models easily memorize training data, reducing generalization.**"
- **Training Stability:** hard due to **exploding or vanishing gradients** (explosion icon on backprop path; 3D cliff-shaped error surface figure where updates jump wildly).

### A Box of Tricks for Training Neural Networks (slide 79 — memorize the 3 groups!)
**Optimization Tricks:**
- *Choose the right optimizer:* tailored to task (e.g., **SGD, Adam**).
- *Learning Rate Decay:* dynamically reduces step size during training.
- *Weight Initialization:* ensures stable convergence with appropriate starting weights.
- *Gradient Clipping:* prevents **exploding gradients** in deep or recurrent networks.

**Regularization Tricks:**
- *Dropout:* randomly disables neurons to prevent overfitting.
- *L1/L2 Regularization:* penalizes large or non-sparse weights to improve generalization.
- *Early Stopping:* stops training when validation performance stagnates.
- *Batch/Layer Normalization:* stabilizes training and improves convergence.

**Architectural Choices:**
- *Choosing the right architecture:* adapt model complexity to the task (e.g., **CNNs for images, RNNs for sequences**).

### Common Mistakes
- Mixing up remedies: gradient clipping fixes **exploding** gradients (not vanishing); dropout fixes **overfitting** (not slow convergence).
- Confusing saddle point (flat, not a minimum) with local minimum.

### Comparison

| Problem | Trick(s) |
|---|---|
| Exploding gradients | Gradient clipping |
| Overfitting | Dropout, L1/L2, Early stopping |
| Unstable/slow convergence | Weight init, Batch/Layer Norm, LR decay, right optimizer |
| Wrong inductive bias | Architecture choice (CNN images, RNN sequences) |

### Diagram / Image Explanation
**Slide 74:** 1D loss curve with P1 (Local Min), P2 (Saddle Point), P3 (Global Min).
**Slides 75–76 (overfitting):** temperature scatter; overlaid fits: jagged yellow (overfit), red (good), smooth green (underfit).
- *Exam question:* identify which curve overfits and why (memorizes noise → poor generalization).
**Slide 78 (cliff error surface):** exploding-gradient jump depicted by purple arrows leaping off a cliff-shaped surface.

### Important Keywords
- **Vanishing gradients** — gradients shrink to ~0 in deep nets; learning stalls.
- **Exploding gradients** — gradients blow up; unstable jumps.
- **Saddle point** — flat point that is neither min nor max.
- **Overfitting / generalization** — memorizing training data vs. performing on new data.
- **Dropout / gradient clipping / early stopping / batch normalization / weight initialization** — the tricks.
- **Adam** — an SGD extension optimizer, recommended default.

### Memory Tips
Box of tricks: **"O-R-A"** — **O**ptimization, **R**egularization, **A**rchitecture.

---

## Topic 21: GD vs. SGD vs. Mini-Batch GD

### Definition
Three variants of gradient descent differing in how much data is used per update: **GD** averages gradients over the **entire dataset** each iteration; **SGD** updates parameters using **one data point** at a time; **Mini-Batch GD** divides the data into **smaller batches** and averages gradients per batch.

### Simple Example
Lecture motivation: **30,000 genes (features), 1 million data points** → GD must compute 1,000,000 terms per parameter per step; over 1000 epochs that's **≥ 30,000,000,000,000 (30 trillion) terms**! SGD/mini-batch make this tractable.

### Why is it Important?
Full comparison slides + two "Homework: Summary" slides = professor considers this crucial.

### Key Points
- **GD (Batch GD):** average gradients over entire dataset each iteration. No fluctuation in learning; stable but often **slow convergence**; **sensitive to local minima**. Each epoch = one weight update. Cost curve: smooth decrease (but smooth ≠ guaranteed convergence).
- **SGD:** picks **one data point** each step, uses only it to calculate derivatives and update parameters. Adds **randomness/fluctuations** that can **speed up convergence** but may cause **very noisy updates**. Each step = gradient of one point + one update; each epoch = stepping through each data point individually. Cost curve: lots of fluctuations.
  - **Benefits:** faster updates (more frequent), **avoids local minima** (random fluctuations help escape — especially helpful in non-convex problems), **memory efficiency** (doesn't need whole dataset loaded at once).
  - **However:** **sensitive to noisy updates or outliers** ("this outlier will cost us some time if it got chosen") and slow, as it requires an update after each iteration.
- **Mini-Batch GD:** the **best of both worlds** — divides data into small batches, averages gradients per batch. Fluctuates (small-sample averaging) but achieves **faster, more stable convergence than SGD** — "**more stable estimates for the parameters in fewer steps**." Each step = gradient over a mini-batch + one update; each epoch = several mini-batch steps covering the entire dataset. Cost curve: relatively small fluctuations vs. SGD.
- SGD's fluctuations can help with **non-convex optimization** (escape local minima; the non-convex vs. convex figure).
- In SGD the cost for one point is 𝒥 = (y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)² (no sum over M).

### Common Mistakes
- Saying SGD is always faster — per-update yes, but noisy; mini-batch usually converges best.
- Confusing **step** and **epoch** (epoch = one pass over ALL data).
- Forgetting GD's smooth cost curve "does not mean it will converge."

### Comparison (high-probability exam table!)

| | **GD (Batch)** | **SGD** | **Mini-Batch GD** |
|---|---|---|---|
| Data per update | Entire dataset | **One data point** | A small batch |
| Cost curve | Smooth | Heavy fluctuations | Small fluctuations |
| Speed per epoch | Slow (1 update) | Many updates | Several updates |
| Stability | Very stable | Very noisy | Balanced |
| Local minima | Sensitive (can get stuck) | Fluctuations help escape | Balanced |
| Memory | Whole dataset | Single point | One batch |
| Outliers | Averaged out | **Sensitive** | Dampened |

### Diagram / Image Explanation
**Slide 85 (SGD scatter):** randomly picked data points (green) among dataset points — one point per step drives the update.
**Slide 86 (non-convex vs. convex):** GD trajectory vs. non-gradient / hybrid algorithms on both landscapes — fluctuations help in non-convex terrain.
**Slide 87 (outlier):** red outlier point "will cost us some time if it got chosen" → SGD outlier sensitivity.
**Slide 88 (mini-batches):** blue/red/orange point groups = different mini-batches used per update.
**Slide 89 (green blocks):** GD = one big block; SGD = many single rows; MBGD = several medium blocks.
**Slide 90 (three cost curves):** GD smooth / SGD jagged / MBGD mildly jagged vs. epochs.
- *Exam question:* Match each cost-vs-epoch curve to GD/SGD/MBGD.

### Important Keywords
- **Epoch** — one full pass over the dataset.
- **Batch / mini-batch** — subset of data per update.
- **Stochastic** — random single-sample-based.
- **Fluctuations** — noise in updates that can help escape local minima.

### Memory Tips
"**All, One, Some**" → GD uses **All**, SGD uses **One**, Mini-batch uses **Some**.

---

## Topic 22: Choosing the Right Optimizer

### Definition
An optimizer is the algorithm that applies the gradients to update the parameters; different optimizers (SGD, Adam) suit different tasks.

### Simple Example
Training an image classifier: start with **Adam** because it performs well without much tuning.

### Why is it Important?
Practical recommendation slide with a "Remember" box — likely short-answer question.

### Key Points (slide 97)
- **Start with Adam (a stochastic gradient descent extension)** for most tasks — it generally performs well **without much tuning**.
- **Use learning rate decay with SGD** when training large models — often improves stability.
- **Tune learning rates:** learning rates and decay parameters significantly impact training speed and stability.
- **Remember:** experimentation helps choose the best option, but **monitor how the cost evolves with time during training**; different tasks may favor different optimizers or decay schedules.

### Common Mistakes
- Thinking Adam is unrelated to SGD — Adam **is an SGD extension**.
- Setting an optimizer and never monitoring the cost curve.

### Important Keywords
- **Adam** — adaptive SGD-extension optimizer, good default.
- **Optimizer** — algorithm applying gradient updates.

### Memory Tips
"**When in doubt, Adam out(performs).**"

---

## Topic 23: Normalizing Input

### Definition
**Input normalization** rescales features using the training data's mean (μ) and variance (σ²) so features are centered and comparable, which stabilizes and speeds up training.

### Simple Example
Feature ranging 0–15 with mean ~4 becomes centered around 0 after x_normalized = (x − μ)/σ² — the scatter plot shifts from the upper-right region to be centered on the origin.

### Why is it Important?
The highlighted rule: **"μ and σ become part of the model and must be the same in production"** — a classic data-leakage/deployment exam question.

### Key Points
- **μ = (1/M) Σ x⁽ⁱ⁾** (mean); **σ² = (1/M) Σ (x⁽ⁱ⁾ − μ)²** (variance).
- **x_normalized = (x − μ)/σ²** (lecture's formula).
- Pipeline (slide 99 figure): train data → processing → modelling → final model; the **processing parameters (e.g., normalization) and processing steps** are saved and applied to **test data** before prediction.
- Compute μ, σ on **training data only**; reuse identical values for test/production data.

### Common Mistakes
- Re-computing μ and σ on the test set → data leakage / inconsistent predictions.
- Forgetting normalization parameters are **part of the model** artifact.

### Comparison

| Training phase | Production/Test phase |
|---|---|
| Compute μ, σ from train data | **Reuse** train μ, σ |
| Fit processing + model | Apply saved processing + model |

### Formula Explanation
**x_normalized = (x − μ)/σ²** — subtract mean (center at 0), divide by variance (scale).
- *Common mistake:* not realizing μ, σ must be frozen after training.

### Diagram / Image Explanation
**Slide 99:** Left: before/after scatter plots (data moves to be centered at origin). Right: train/test pipeline diagram with "processing parameters (e.g. normalization) and processing steps" circle feeding the test-data processing box.
- *Exam question:* "Why must test data use the training μ and σ?" → they are part of the model; using different values makes predictions inconsistent.

### Important Keywords
- **Normalization** — centering and scaling inputs.
- **μ (mean), σ² (variance)** — processing parameters learned from train data.
- **Processing parameters** — saved with the model for production.

### Memory Tips
"**Fit on train, freeze for the rest.**"

---

## Topic 24: Building a Neural Network from Scratch (Pipeline) & The Golden Rule

### Definition
The end-to-end recipe for constructing and training a neural network, from framework choice to hyperparameter optimization.

### Simple Example
Using a framework: define a model with 2 hidden layers + dropout, pick cross-entropy cost, choose Adam with LR decay, train with backprop, then tune hyperparameters.

### Why is it Important?
Pipeline-ordering questions are explicitly listed in the lecture's final slides.

### Key Points — the 6 steps in order (slide 101, memorize!)
1. **Choose Framework**
2. **Build the model** (network size, layers or batch normalization, dropout, …)
3. **Define the cost function**
4. **Select an optimizer and learning rate**
5. **Apply backpropagation**
6. **Do hyperparameters optimization**

- **The Golden Rule of Neural Nets: "Neural Networks are the second-best way to do everything!"** — meaning NNs are versatile general-purpose tools, but a specialized method usually beats them on a specific problem.

### Common Mistakes
- Putting optimizer selection before cost definition (cost comes first — the optimizer minimizes it).
- Treating hyperparameter optimization as a first step — it is the final tuning loop.

### Important Keywords
- **Framework** — software library for building NNs.
- **Hyperparameter optimization** — final tuning of size, LR, AF, etc.
- **Golden Rule** — NNs are the *second-best* way to do *everything*.

### Memory Tips
**"F-B-C-O-B-H"**: "**F**ine **B**akers **C**ook **O**nly **B**etter **H**oney" → Framework, Build, Cost, Optimizer, Backprop, Hyperparameters.

---

# 3. Topic-wise Exam Practice

## Perceptron & Background
1. **MCQ:** The components of a single perceptron in order are:
   A) Activation → Sum → Weights B) Weights → Sum → Activation C) Sum → Weights → Activation
   **Answer:** B (inputs × weights → summing junction → activation function → output).
2. **True/False:** Deep learning requires hand-designed features. **Answer:** False (self-estimated features).
3. **Fill in the blank:** Deep learning models use ______ or more layers, but typically hundreds or thousands. **Answer:** three.
4. **One-word:** What biological structure inspired the weights of a perceptron? **Answer:** Synapses.
5. **Short answer:** What makes DL special vs. ML? **Answer:** DL algorithms recognize patterns using **self-estimated** features; ML needs **pre-calculated (hand-designed)** features.

## Linear Regression
6. **Definition question:** Define regression. **Answer:** A method to model the relationship between a dependent variable (output) and one or more independent variables (input), to predict continuous values.
7. **MCQ:** In y = b + Σxᵢwᵢ, b is: A) weight B) bias C) input D) index. **Answer:** B.
8. **Scenario:** You want to predict crop yield from rainfall, fertilizer, temperature. Which analysis type? **Answer:** (Linear) regression — continuous target.
9. **Matching:** Match assumption to description: Homoscedasticity / Normality / No multicollinearity ↔ (a) errors normally distributed, (b) constant variance of residuals, (c) predictors not too correlated. **Answer:** Homoscedasticity–b, Normality–a, No multicollinearity–c.
10. **True/False:** Polynomial regression is a non-linear model in its parameters. **Answer:** False — still linear in parameters.
11. **Two-word answer:** Which regression can shrink coefficients exactly to zero? **Answer:** Lasso regression (L1).
12. **MCQ:** Ridge regression adds which penalty? A) λΣ|wⱼ| B) λΣwⱼ² C) dropout D) e^w. **Answer:** B.

## Loss/Cost & MSE
13. **Short answer:** Difference between loss and cost function? **Answer:** Loss = error of a single prediction; cost/objective = error summarized over all M data points.
14. **Formula interpretation:** In MSE = (1/2M)Σ(y⁽ⁱ⁾−ŷ⁽ⁱ⁾)², where are the parameters? **Answer:** Inside ŷ, since ŷ = xω + b; θ = {ω, b}.
15. **MCQ:** The MSE-vs-parameter plot is a parabola because: A) data is parabolic B) MSE is a quadratic function C) learning rate D) sigmoid. **Answer:** B.
16. **Short answer:** Why not simply use np.min to find the best parameters? **Answer:** Brute force doesn't know the structure of the cost function and is impractical for continuous, high-dimensional optimization; we need an algorithm exploiting the mathematical structure (GD).

## Gradient & Gradient Descent
17. **Definition:** What two things does the gradient tell us? **Answer:** Direction of steepest ascent and magnitude (how steep/fast the function changes).
18. **Fill in the blank:** GD update rule: θⱼ ← θⱼ ______. **Answer:** − α (∂/∂θⱼ)𝒥(θ).
19. **MCQ (lecture quiz):** A loss function is perfectly convex; running GD, are you guaranteed the global minimum? A) Yes, always B) No C) Only if LR small D) Only with SGD. **Answer:** A — in convex problems GD is guaranteed to move toward the global minimum. *(Caveat: with a badly chosen too-high learning rate it can still oscillate — the lecture's highlighted statement is that convexity guarantees moving toward the global minimum.)*
20. **Short answer:** Why does step size decrease with time even with fixed α? **Answer:** The gradient magnitude shrinks as the slope flattens near the minimum.
21. **Short answer:** Why keep the hat on θ̂ after convergence? **Answer:** It remains an estimate of the optimal parameter, not the true value.
22. **Diagram question:** On a non-convex loss with P1, P2, P3 — identify each. **Answer:** P1 local minimum, P2 saddle point, P3 global minimum.
23. **True/False:** GD's smooth cost curve guarantees convergence. **Answer:** False ("does not mean it will converge").

## Learning Rate
24. **Scenario:** Cost oscillates wildly and even increases across iterations. Diagnosis? **Answer:** Learning rate too high (parameters slosh back and forth across the ravine).
25. **MCQ:** Which decay reduces α by a factor every fixed number of epochs? A) Exponential B) Step C) 1/t. **Answer:** B.
26. **Fill in blank:** Exponential decay: α = α₀ × ______. **Answer:** e^(−decay rate × epoch).
27. **True/False:** There is one universal rule to pick the best learning rate. **Answer:** False.

## Logistic Regression
28. **One-word:** Activation function of logistic regression? **Answer:** Sigmoid.
29. **Formula interpretation:** σ(0) = ? **Answer:** 0.5.
30. **MCQ:** Output range of sigmoid: A) [0,1] B) (0,1) C) (−1,1) D) [−1,1]. **Answer:** B.
31. **Pipeline ordering:** Order: quantizer, weighted sum, sigmoid, binary output, input. **Answer:** input → weighted sum (z = wᵀx+b) → sigmoid → (probability ŷ) → quantizer → binary output {0,1}.
32. **Short answer:** What does the quantizer do? **Answer:** Thresholds the predicted probability into a binary estimate {0,1}; the threshold can be adjusted.
33. **Fill in blank:** Binary cross-entropy: 𝒥(ω,b) = −(1/M)Σ( ______ + ______ ). **Answer:** y⁽ⁱ⁾log(ŷ⁽ⁱ⁾) and (1−y⁽ⁱ⁾)log(1−ŷ⁽ⁱ⁾).
34. **Formula:** Sigmoid derivative? **Answer:** σ′ = σ(1−σ) (i.e., ∂ŷ/∂z = ŷ(1−ŷ)).
35. **Formula:** Combined gradient of cost w.r.t. w in logistic regression? **Answer:** ∂𝒥/∂ω = (ŷ−y)·xᵀ.
36. **Scenario:** Breast cancer prediction with labels benign=0/malignant=1 from image features. Algorithm? **Answer:** Logistic regression (binary classification).

## Multinomial Classification
37. **Two-word:** Two approaches for multiclass logistic regression? **Answer:** One-vs-rest; SoftMax (regression).
38. **Formula interpretation:** What property does SoftMax output satisfy? **Answer:** Vector of probabilities summing to 1.
39. **MCQ:** SoftMax is a: A) normalized exponential function B) linear function C) step function D) distance metric. **Answer:** A.
40. **Fill in the blank:** One-vs-rest fits a separate ______ for each class. **Answer:** logistic regression model.
41. **Given z = [1.30, 5.10, 2.20, 0.70, 1.10], which class does softmax favor?** **Answer:** class 2 (score 5.10 → probability 0.90).

## Limitations & NN Architecture
42. **List question:** Name 4 limitations of logistic regression. **Answer:** Linear decision boundaries; limited feature interaction; limited capacity for complexity; sensitive to outliers/noise; poor scalability to large/high-dimensional data; inefficient multinomial extension (any 4).
43. **Short answer (lecture bottomline):** How can a neural network be explained in terms of logistic regression? **Answer:** As several logistic regression blocks (with caveats/extensions) — these blocks are called neurons.
44. **One-word:** What does each successive hidden layer learn? **Answer:** (Increasingly complex) hierarchical features.
45. **MCQ:** Adding more OUTPUT neurons addresses: A) deeper patterns B) multilabel classification C) overfitting D) faster training. **Answer:** B.

## Forward & Backpropagation
46. **Fill in blanks:** z⁽ⁱ⁾ = ______; a⁽ⁱ⁾ = ______. **Answer:** W⁽ⁱ⁾a⁽ⁱ⁻¹⁾ + b⁽ⁱ⁾; g⁽ⁱ⁾(z⁽ⁱ⁾).
47. **One-word:** ŷ equals the activation of which layer? **Answer:** last (output) layer: ŷ = a⁽ˡ⁾.
48. **Ordering:** Order NN training: update parameters, forward propagation, backpropagation, compute cost. **Answer:** forward propagation → compute cost → backpropagation → update parameters (repeat).
49. **Short answer:** Which mathematical rule finds partial derivatives in backprop? **Answer:** The chain rule.
50. **Formula interpretation:** In the local gradient picture, ∂L/∂x = ? **Answer:** ∂L/∂z · ∂z/∂x (upstream gradient × local gradient).
51. **Dimension check:** If W⁽²⁾ ∈ R⁵ˣ³, what is the shape of ∇_W⁽²⁾J? **Answer:** R⁵ˣ³ — gradients match parameter shapes.
52. **True/False:** Backprop only updates the last layer's weights. **Answer:** False — every layer's W and b are updated.
53. **Short answer:** GD for NNs repeats until? **Answer:** Convergence **or** maximum number of iterations reached.

## Network Size & Activation Functions
54. **True/False:** Network size is learned during training. **Answer:** False — it is a hyperparameter fixed before training.
55. **Short answer:** Too many layers/neurons cause ______; too few cause ______. **Answer:** overfitting; underfitting.
56. **List:** 5 tips for choosing network size. **Answer:** Start small; gradually increase complexity (via validation); consider task complexity; experiment with dropout; use early stopping.
57. **Matching:** Match: ReLU / Leaky ReLU / ELU / Sigmoid ↔ (a) 1/(1+e^(−x)), (b) max(0,x), (c) max(0.1x,x), (d) x if x≥0 else α(eˣ−1). **Answer:** ReLU–b, Leaky ReLU–c, ELU–d, Sigmoid–a.
58. **List:** 5 characteristics of activation functions. **Answer:** Non-linearity, differentiability, bounded output, sparsity, computational efficiency.
59. **Short answer:** Why is differentiability of AFs essential? **Answer:** It enables gradient calculation for learning via backpropagation.
60. **One-word:** What biological phenomenon inspired AFs? **Answer:** Action potential (firing threshold).

## Training Tricks
61. **Matching:** Match problem→trick: exploding gradients / overfitting / images / sequences ↔ gradient clipping / dropout (or L1/L2, early stopping) / CNN / RNN. **Answer:** as listed.
62. **MCQ:** Which is NOT a regularization trick? A) Dropout B) Early stopping C) Gradient clipping D) L1/L2. **Answer:** C (it's an optimization trick).
63. **Fill in blank:** Deep networks often face ______ gradients and get stuck in ______ points. **Answer:** vanishing; saddle.
64. **Definition:** Overfitting (per lecture). **Answer:** High-capacity models easily memorize training data, reducing generalization.

## GD vs. SGD vs. Mini-Batch
65. **MCQ:** Which uses exactly one data point per update? A) GD B) SGD C) MBGD. **Answer:** B.
66. **Diagram question:** Cost curve with heavy fluctuations vs. epochs belongs to? **Answer:** SGD.
67. **Short answer:** Two benefits of SGD. **Answer:** Faster (more frequent) updates; helps avoid local minima via random fluctuations; memory efficiency (any two).
68. **Short answer:** Main drawback of SGD regarding data quality? **Answer:** Sensitive to noisy updates/outliers — an outlier costs time if chosen.
69. **Short answer:** Why is mini-batch "best of both worlds"? **Answer:** Averages small batches → more stable estimates than SGD in fewer steps, faster than full GD.
70. **Numeric scenario (lecture):** 30,000 features, 1M data points, 1000 epochs — approximately how many terms does full GD compute? **Answer:** At least 30,000,000,000,000 (30 trillion).
71. **One-word:** Recommended default optimizer? **Answer:** Adam.

## Normalization & Pipeline
72. **Formula:** Lecture's normalization formula? **Answer:** x_normalized = (x − μ)/σ², with μ = (1/M)Σx⁽ⁱ⁾, σ² = (1/M)Σ(x⁽ⁱ⁾−μ)².
73. **Scenario:** A deployed model performs oddly; the team normalized production data with production-computed μ, σ. Problem? **Answer:** μ and σ are part of the model — must reuse the training values in production.
74. **Pipeline ordering:** Order the 6 NN-from-scratch steps. **Answer:** Choose framework → Build the model → Define cost function → Select optimizer & learning rate → Apply backpropagation → Hyperparameter optimization.
75. **Quote completion:** "Neural Networks are the ______ way to do ______!" **Answer:** second-best; everything.

---

# 4. Flashcards

1. **Neural networks are the generalization of ______?** → Linear regression.
2. **Perceptron components in order?** → Inputs → weights (+bias) → summing junction Σ → activation function φ → output.
3. **How many layers makes "deep"?** → 3 or more (typically hundreds/thousands).
4. **DL vs. ML feature difference?** → DL: self-estimated features; ML: hand-designed/pre-calculated features.
5. **Define regression.** → Models relationship between dependent variable (output) and independent variables (input) to predict continuous values.
6. **Linear regression model formula?** → y = b + Σxᵢwᵢ (ŷ = xω + b).
7. **What does "fit the model" mean?** → Find best parameters with acceptable error margin but still reliable.
8. **5 assumptions of linear regression?** → Linearity, Independence, Homoscedasticity, Normality, No multicollinearity.
9. **Homoscedasticity?** → Constant variance of residuals across levels.
10. **Loss vs. cost function?** → Loss = single prediction error; cost = all M data points.
11. **MSE formula?** → 𝒥(θ) = (1/2M)Σ(y⁽ⁱ⁾−ŷ⁽ⁱ⁾)².
12. **Why is MSE curve a parabola?** → It is a quadratic function.
13. **Where are the parameters in MSE?** → Inside ŷ = xω + b; θ = weights ω and bias b.
14. **Gradient definition?** → Derivative of a function w.r.t. its inputs: direction of steepest ascent + magnitude of change.
15. **GD update rule?** → θⱼ ← θⱼ − α(∂/∂θⱼ)𝒥(θ).
16. **GD algorithm 2 steps?** → 1) Random init + choose learning rate; 2) Repeat update until convergence.
17. **Convex + GD guarantee?** → Guaranteed to move toward the global minimum.
18. **Learning rate role?** → Controls how much we update at each iteration (step size).
19. **LR too high?** → Parameters slosh back and forth across the ravine.
20. **LR too low?** → Too many updates before reaching the optimum.
21. **3 LR decay techniques?** → Step decay, exponential decay, 1/t decay.
22. **Ridge penalty?** → L2: λΣwⱼ².
23. **Lasso specialty?** → L1 can shrink coefficients to zero → feature selection.
24. **Elastic Net?** → Combines Lasso + Ridge; for highly correlated predictors.
25. **Polynomial regression?** → Adds x², x³ terms; captures non-linear relations; still linear in parameters.
26. **Sigmoid formula?** → σ(z) = 1/(1+e^(−z)).
27. **σ(0)?** → 0.5.
28. **Sigmoid output range?** → (0,1) — interpreted as probability of one of two alternatives.
29. **z (weighted function)?** → z = Σ(wᵢxᵢ) + b = wᵀx + b.
30. **Quantizer?** → Thresholds probability into binary estimate {0,1}; threshold adjustable.
31. **Binary cross-entropy?** → 𝒥 = −(1/M)Σ(y log ŷ + (1−y)log(1−ŷ)).
32. **Multinomial cross-entropy?** → 𝒥 = −(1/M)ΣᵢΣₖ yᵢ<k> log(ŷᵢ<k>).
33. **Chain rule for logistic regression?** → ∂𝒥/∂ω = (∂𝒥/∂ŷ)(∂ŷ/∂z)(∂z/∂ω).
34. **Sigmoid derivative?** → σ′ = σ(1−σ); ∂ŷ/∂z = ŷ(1−ŷ).
35. **Combined logistic gradient?** → (ŷ−y)·xᵀ.
36. **Two multiclass strategies?** → One-vs-rest; SoftMax regression.
37. **SoftMax formula?** → softmax(zᵢ) = e^(zᵢ)/Σⱼe^(zⱼ); outputs sum to 1.
38. **6 limitations of logistic regression?** → Linear boundaries, limited feature interaction, limited capacity, outlier sensitivity, poor scalability, inefficient multinomial handling.
39. **NN bottomline?** → NN = several logistic regression blocks = neurons.
40. **Why go deep?** → Series of complex transformations → learns hierarchical features.
41. **Forward prop layer equations?** → z⁽ⁱ⁾ = W⁽ⁱ⁾a⁽ⁱ⁻¹⁾+b⁽ⁱ⁾; a⁽ⁱ⁾ = g⁽ⁱ⁾(z⁽ⁱ⁾); ŷ = a⁽ˡ⁾.
42. **Backprop uses which rule?** → The chain rule (backwards, layer by layer).
43. **Local gradient formula?** → ∂L/∂x = ∂L/∂z · ∂z/∂x.
44. **Network size?** → Number of layers + neurons per layer; a hyperparameter fixed before training.
45. **Too big / too small network?** → Overfitting / underfitting.
46. **5 network-size tips?** → Start small; increase gradually via validation; match task complexity; dropout; early stopping.
47. **5 AF characteristics?** → Non-linearity, differentiability, bounded output, sparsity, computational efficiency.
48. **ReLU / Leaky ReLU?** → max(0,x) / max(0.1x,x).
49. **ELU?** → x if x≥0; α(eˣ−1) if x<0.
50. **AF inspired by?** → Biological action potentials (fire if threshold met).
51. **3 deep-training problems?** → Optimization difficulty (vanishing gradients, saddle points), overfitting, training instability (exploding/vanishing gradients).
52. **Gradient clipping fixes?** → Exploding gradients.
53. **Dropout does?** → Randomly disables neurons to prevent overfitting.
54. **Early stopping?** → Stop when validation performance stagnates/plateaus.
55. **Batch/Layer Norm?** → Stabilizes training, improves convergence.
56. **CNN vs RNN use?** → CNNs for images; RNNs for sequences.
57. **GD / SGD / MBGD data per step?** → All / one point / a mini-batch.
58. **SGD benefits?** → Faster updates, avoids local minima, memory efficiency.
59. **SGD drawbacks?** → Sensitive to outliers/noise; needs update per iteration.
60. **MBGD advantage?** → More stable parameter estimates in fewer steps (best of both worlds).
61. **Default optimizer advice?** → Start with Adam (an SGD extension); LR decay with SGD for large models.
62. **Normalization formulas?** → μ=(1/M)Σx; σ²=(1/M)Σ(x−μ)²; x_norm=(x−μ)/σ².
63. **Normalization production rule?** → μ and σ are part of the model — same values in production.
64. **NN-from-scratch pipeline?** → Framework → Build model → Cost function → Optimizer+LR → Backprop → Hyperparameter optimization.
65. **Golden Rule of Neural Nets?** → "Neural Networks are the second-best way to do everything!"
66. **Epoch definition?** → One complete pass through the entire dataset.
67. **P1/P2/P3 on non-convex loss?** → Local min / saddle point / global min.
68. **Why hat on ŷ, θ̂?** → Denotes estimated values (estimates remain estimates after convergence).

---

# 5. Rapid Revision

- **NN = generalization of linear regression**; single perceptron ≈ linear/logistic regression.
- Perceptron: inputs → **weights+bias** → **Σ** → **activation** → output.
- Classical = rules; ML = hand-designed features; **DL = self-estimated features**; deep = **3+ layers**.
- Regression predicts **continuous** values: **ŷ = xω + b**; data = historical observations.
- **Fit = find best parameters** (θ = ω, b) minimizing error.
- Assumptions: **Linearity, Independence, Homoscedasticity, Normality, No multicollinearity**.
- **Loss** = one point (y−ŷ)²; **Cost = MSE = (1/2M)Σ(y−ŷ)²** — quadratic → parabola → convex.
- **Gradient** = derivative: steepest-ascent direction + magnitude.
- **GD:** random init + learning rate; repeat θⱼ ← θⱼ − α∂𝒥/∂θⱼ until convergence. Convex ⇒ global minimum guaranteed.
- **α too high → oscillation; too low → too slow.** Decay: step / exponential / 1/t.
- Ridge L2 shrinks; **Lasso L1 zeroes (feature selection)**; Elastic Net = both; polynomial adds x², x³ (still linear in parameters).
- **Logistic regression:** z = wᵀx+b → **sigmoid** (probability) → **quantizer** (threshold) → {0,1}.
- σ(z) = 1/(1+e^(−z)); σ(0)=0.5; range (0,1); **σ′ = σ(1−σ)**.
- **Cross-entropy:** −(1/M)Σ(y log ŷ + (1−y)log(1−ŷ)); multinomial: −(1/M)ΣΣ y<k> log ŷ<k>.
- Chain rule: ∂𝒥/∂ω = ∂𝒥/∂ŷ · ∂ŷ/∂z · ∂z/∂ω = **(ŷ−y)xᵀ**.
- Multiclass: **one-vs-rest** (K models) or **SoftMax** (normalized exponential; probabilities sum to 1).
- Logistic limits: **linear boundaries**, weak feature interaction, low capacity, outlier-sensitive, poor scaling, weak multinomial → need NN.
- **NN = stacked logistic regression blocks (neurons)**; deeper → **hierarchical features**.
- **Forward:** z⁽ⁱ⁾=W⁽ⁱ⁾a⁽ⁱ⁻¹⁾+b⁽ⁱ⁾, a⁽ⁱ⁾=g⁽ⁱ⁾(z⁽ⁱ⁾), ŷ=a⁽ˡ⁾ → cost.
- **Backward:** chain rule, local gradients, update EVERY W⁽ⁱ⁾, b⁽ⁱ⁾ with −α∇.
- **Network size = hyperparameter**: too big → overfit; too small → underfit. Start small, grow via validation, dropout, early stopping.
- **AF characteristics:** non-linearity, differentiability, bounded output, sparsity, efficiency. Sigmoid, tanh, **ReLU max(0,x)**, Leaky ReLU max(0.1x,x), Maxout, ELU.
- Deep-training problems: **vanishing/exploding gradients, saddle points, overfitting**.
- Tricks: optimizer (SGD/**Adam**), LR decay, weight init, **gradient clipping** | **dropout**, L1/L2, **early stopping**, batch/layer norm | architecture (CNN images, RNN sequences).
- **GD = all data (smooth, slow); SGD = 1 point (noisy, escapes local minima, outlier-sensitive); Mini-batch = best of both** (stable estimates, fewer steps).
- **Normalize with train μ, σ²; reuse SAME values in production** — they're part of the model.
- Pipeline: **Framework → Model → Cost → Optimizer/LR → Backprop → Hyperparameter optimization**.
- Golden Rule: NNs are the **second-best way to do everything**.

---

# 6. High Probability Exam Questions

### ★★★★★ Very High
1. Write and explain the **gradient descent update rule** and the 2-step GD algorithm. (Repeated on 6+ slides)
2. **GD vs. SGD vs. Mini-Batch GD** — compare data per update, stability, convergence curves, pros/cons. (Two homework-summary slides!)
3. **Loss vs. cost function; write the MSE formula** and explain each symbol; why is it quadratic/parabolic? (Direct lecture quiz)
4. Convex loss + GD: **are you guaranteed the global minimum?** (Explicit lecture MCQ — Yes for convex; highlighted "guaranteed... global minimum since it is convex")
5. **Effect of learning rate too high / too low / just right** (+ diagram interpretation). (Shown twice)
6. **Sigmoid function:** formula, range, σ(0), why the output is a probability; sigmoid derivative σ(1−σ).
7. **Binary cross-entropy formula** and the meaning of y and ŷ.
8. **Limitations of logistic regression / why do we need neural networks?**
9. **Forward propagation equations** (z⁽ⁱ⁾, a⁽ⁱ⁾, ŷ = a⁽ˡ⁾) and **backpropagation via chain rule** with update steps for all parameters.
10. **NN = several logistic regression blocks (neurons)** — explain.

### ★★★★ High
11. **Chain rule for logistic regression:** three factors and combined result (ŷ−y)xᵀ.
12. **5 assumptions of linear regression** with house-price examples.
13. **One-vs-rest vs. SoftMax**; SoftMax formula and its sum-to-1 property.
14. **Network size as hyperparameter**; overfitting vs. underfitting; 5 sizing tips.
15. **Activation functions:** 5 characteristics + formulas of ReLU/Leaky ReLU/Sigmoid/ELU.
16. **Box of tricks:** match problems (exploding gradients, overfitting…) to tricks (clipping, dropout, early stopping, batch norm).
17. **Normalization:** formulas + why μ, σ must be identical in production.
18. **Why does GD step size decrease over time? Why keep the hat on θ̂?** (Lecture "questions for thought")
19. **Ridge vs. Lasso vs. Elastic Net** comparison.
20. **P1/P2/P3 diagram:** local min, saddle point, global min; vanishing gradients in deep nets.

### ★★★ Medium
21. Classical vs. ML vs. DL pipelines (hand-designed vs. self-estimated features).
22. Perceptron anatomy and biological analogy (synapse→weight, action potential→activation).
23. Why brute-force np.min can't replace GD.
24. Notation: x ∈ Rᴺ, X ∈ Rᴹˣᴺ, y ∈ {0,1}, w ∈ Rᴺ, ŷ ∈ (0,1); gradient shapes match parameter shapes.
25. The 30,000-genes / 1M-points / 30-trillion-terms motivation for SGD.
26. NN-from-scratch 6-step pipeline ordering.
27. Learning rate decay formulas (step, exponential, 1/t).
28. Adam recommendation & monitoring the cost during training.
29. Hierarchical features / "why do we go deep" image.
30. Golden Rule: "NNs are the second-best way to do everything."

---

# 7. Lecture Cheat Sheet

## Core Definitions
- **Regression:** model relation between dependent (output) & independent (input) variables → predict continuous values.
- **Fit the model:** find best parameters θ = {ω, b} with acceptable error, still reliable.
- **Loss:** single-prediction error. **Cost (objective):** all-M-points error.
- **Gradient:** derivative — steepest ascent direction + magnitude.
- **Gradient Descent:** iterative optimization minimizing functions via slope-based weight updates.
- **Logistic regression:** linear model + sigmoid → probability → threshold → class.
- **Neural network:** several logistic regression blocks (neurons) with hidden layers.
- **Network size:** #layers + #neurons/layer — a hyperparameter.
- **Overfitting:** high-capacity model memorizes training data, poor generalization.

## Key Formulas
| Name | Formula |
|---|---|
| Linear model | ŷ = xω + b = b + Σxᵢwᵢ |
| MSE cost | 𝒥(θ) = (1/2M)Σ(y⁽ⁱ⁾−ŷ⁽ⁱ⁾)² |
| GD update | θⱼ ← θⱼ − α(∂/∂θⱼ)𝒥(θ) |
| Sigmoid | σ(z) = 1/(1+e^(−z)), z = wᵀx+b |
| Sigmoid derivative | σ′ = σ(1−σ) |
| Binary cross-entropy | −(1/M)Σ(y log ŷ + (1−y)log(1−ŷ)) |
| Multinomial CE | −(1/M)ΣᵢΣₖ y<k> log ŷ<k> |
| Logistic gradient | ∂𝒥/∂ω = (ŷ−y)xᵀ |
| SoftMax | e^(zᵢ)/Σⱼe^(zⱼ) (sums to 1) |
| Forward layer | z⁽ⁱ⁾ = W⁽ⁱ⁾a⁽ⁱ⁻¹⁾+b⁽ⁱ⁾; a⁽ⁱ⁾ = g⁽ⁱ⁾(z⁽ⁱ⁾); ŷ = a⁽ˡ⁾ |
| Chain rule (backprop) | ∂𝒥/∂ω⁽¹⁾ = ∂𝒥/∂ŷ · ∂ŷ/∂g⁽ˡ⁾ · … · ∂z⁽¹⁾/∂ω⁽¹⁾ |
| LR decays | step: α₀·factor^(epoch/step); exp: α₀·e^(−rate·epoch); 1/t: α₀/(1+rate·epoch) |
| Normalization | x_norm = (x−μ)/σ²; μ=(1/M)Σx; σ²=(1/M)Σ(x−μ)² |
| Ridge / Lasso | +λΣwⱼ² (L2) / +λΣ|wⱼ| (L1→zeros) |
| ReLU / Leaky / ELU | max(0,x) / max(0.1x,x) / x or α(eˣ−1) |

## Comparisons at a Glance
- **Regression vs. Classification:** continuous vs. discrete; MSE vs. cross-entropy; identity vs. sigmoid.
- **Loss vs. Cost:** 1 point vs. M points.
- **Ridge vs. Lasso:** shrink vs. zero-out (feature selection).
- **Sigmoid vs. SoftMax:** binary vs. K-class (sum to 1).
- **One-vs-rest vs. SoftMax:** K models vs. 1 model.
- **GD vs. SGD vs. MBGD:** all / one / batch; smooth / noisy / mildly noisy.
- **Forward vs. Backward:** compute prediction vs. compute gradients.
- **Parameter vs. Hyperparameter:** learned (W, b) vs. preset (size, α, AF).
- **Overfitting vs. Underfitting:** too big vs. too small network.

## Pipelines
- **Logistic regression:** x → z = wᵀx+b → sigmoid → ŷ → quantizer → {0,1}.
- **NN training loop:** Forward prop → compute cost → Backprop (chain rule) → update all W, b → repeat.
- **Build NN from scratch:** Framework → Build model → Cost function → Optimizer + LR → Backprop → Hyperparameter optimization.
- **Normalization deployment:** fit μ, σ on train → save as processing parameters → apply SAME to test/production.

## Frequently Confused
- Gradient points **up** (ascent); GD steps **down** (minus sign).
- Sigmoid outputs probability **(0,1)**, not the class; the **quantizer** gives the class.
- **L1 = Lasso = zeros**; L2 = Ridge = shrink.
- Gradient clipping ↔ **exploding** gradients (not vanishing).
- Polynomial regression: non-linear in x, **linear in parameters**.
- Homoscedasticity = variance of **residuals** (not features).
- **Adam is an SGD extension**, not a separate paradigm.

## Golden Rule
**"Neural Networks are the second-best way to do everything!"**

---

# 8. Lecture Summary

## Top 50 Most Probable Exam Questions
1. Define regression and its goal.
2. Give the linear regression model formula and label all symbols.
3. What does "fitting the model" mean?
4. List the 5 assumptions of linear regression.
5. Explain homoscedasticity with the house-price example.
6. What is multicollinearity? Give the lecture example.
7. Loss vs. cost/objective function — define both.
8. Write the MSE formula; what does each symbol mean?
9. Why does the cost curve have a parabola shape?
10. Where are the parameters in the MSE equation?
11. Why is MSE squared rather than an absolute mean?
12. Why can't np.min replace gradient descent?
13. Define the gradient — its two pieces of information.
14. What is the sign of the derivative left/right of the minimum?
15. State the gradient descent algorithm (2 steps).
16. Write the GD update rule and explain α.
17. Why does GD step size decrease over time?
18. Why keep the hat on θ̂ after convergence?
19. Convex loss + GD: guaranteed global minimum? Explain.
20. What happens with a too-high / too-low learning rate?
21. Name and write the 3 learning-rate decay formulas.
22. What is polynomial regression, and is it linear in parameters?
23. Compare Ridge, Lasso, Elastic Net.
24. Name 3 applications of linear regression.
25. Define logistic regression's pipeline in order.
26. Write the sigmoid formula; give σ(0) and the output range.
27. What is the quantizer and can the threshold change?
28. Write the binary cross-entropy cost; define y⁽ⁱ⁾ and ŷ⁽ⁱ⁾.
29. Explain cross-entropy with the strawberry/apple basket example.
30. Write the multinomial cross-entropy formula.
31. Write the chain rule for ∂𝒥/∂ω in logistic regression and its three factors.
32. State the sigmoid derivative.
33. Give the combined gradient (ŷ−y)xᵀ and the update rule over M samples.
34. Describe one-vs-rest classification.
35. Write the SoftMax formula and its key property.
36. List 4+ limitations of logistic regression.
37. Why do we need neural networks?
38. What does the "yellow box" in the hidden-layer figure correspond to?
39. State the lecture's bottomline about NNs and logistic regression blocks.
40. Why do we go deep? (hierarchical features)
41. Write the forward propagation equations for a 2-layer network.
42. Define ℓ, n⁽ⁱ⁾, a⁽ⁱ⁾, g⁽ⁱ⁾, W⁽ⁱ⁾, b⁽ⁱ⁾ and the relation ŷ = a⁽ˡ⁾.
43. Explain backpropagation and the role of the chain rule / local gradients.
44. Write the 4 update steps of a 2-layer NN.
45. Why is network size a hyperparameter? Consequences of too big/small?
46. Give the 5 tips for choosing network size.
47. List the 5 characteristics of activation functions and 4 example AFs with formulas.
48. Compare GD, SGD, and mini-batch GD (definitions, curves, pros/cons).
49. Give normalization formulas and the production rule for μ and σ.
50. Order the 6 steps to build a NN from scratch + state the Golden Rule.

## Top 30 Definitions
1. **Regression** — modeling dependent-vs-independent variable relationships to predict continuous values.
2. **Dependent variable** — the output y.
3. **Independent variables** — input features/attributes x.
4. **Perceptron** — a single artificial neuron: weighted sum + bias + activation.
5. **Activation function** — decides if a neuron "fires"; adds non-linearity.
6. **Bias** — constant term added to the weighted sum.
7. **Fitting the model** — finding parameters with acceptable error but reliable.
8. **Loss function** — error of a single prediction.
9. **Cost/objective function** — summarized error over all data points.
10. **MSE** — mean squared error, standard regression cost.
11. **Gradient** — derivative giving steepest-ascent direction and magnitude.
12. **Gradient descent** — iterative optimization minimizing functions via gradient-based updates.
13. **Learning rate** — controls how much we update each iteration.
14. **Convergence** — when updates reach minimum cost.
15. **Convex function** — bowl shape; GD guaranteed toward global minimum.
16. **Saddle point** — flat critical point, neither min nor max.
17. **Learning rate decay** — reducing α over time for stability.
18. **Ridge (L2)** — shrinks coefficients via squared penalty.
19. **Lasso (L1)** — can zero coefficients → feature selection.
20. **Logistic regression** — sigmoid-based binary classifier.
21. **Sigmoid** — 1/(1+e^(−z)); maps to (0,1) probability.
22. **Quantizer** — thresholds probability into binary class.
23. **Cross-entropy** — log-based classification cost.
24. **SoftMax** — normalized exponential mapping scores to probabilities summing to 1.
25. **Hidden layer** — layer between input and output; enables complexity.
26. **Forward propagation** — computing prediction layer by layer.
27. **Backpropagation** — computing all gradients via chain rule backwards.
28. **Hyperparameter** — setting fixed before training (size, α, AF).
29. **Overfitting** — memorizing training data, reducing generalization.
30. **Epoch** — one full pass over the dataset.

## Top 30 Keywords
1. Synaptic weights 2. Summing junction 3. Self-estimated features 4. Hand-designed features 5. Deep neural network 6. Parameters θ (ω, b) 7. Residual/prediction error 8. Homoscedasticity 9. Multicollinearity 10. Parabola/quadratic 11. θ̂ (estimate) 12. Steepest ascent 13. θ₀ random initialization 14. Local/global minimum 15. Step/exponential/1/t decay 16. Elastic Net 17. z = wᵀx + b 18. Threshold 19. One-hot labels 20. One-vs-rest 21. argmax 22. Linear decision boundary 23. Neuron (logistic block) 24. Hierarchical features 25. a⁽ⁱ⁾, z⁽ⁱ⁾, g⁽ⁱ⁾ 26. Local gradient 27. Vanishing/exploding gradients 28. Dropout / early stopping / gradient clipping / batch normalization 29. Adam / SGD / mini-batch 30. Normalization parameters μ, σ

## Top 20 Comparisons
1. Regression vs. Classification
2. Loss vs. Cost function
3. Classical vs. ML vs. DL approaches
4. Biological vs. Artificial neuron
5. Non-deep vs. Deep network
6. Linear vs. Logistic regression
7. MSE vs. Cross-entropy
8. Sigmoid vs. SoftMax
9. One-vs-rest vs. SoftMax regression
10. Ridge vs. Lasso
11. Lasso/Ridge vs. Elastic Net
12. Convex vs. Non-convex loss
13. LR too low vs. just right vs. too high
14. Step vs. Exponential vs. 1/t decay
15. GD vs. SGD
16. SGD vs. Mini-batch GD
17. Forward vs. Backward propagation
18. Parameter vs. Hyperparameter
19. Overfitting vs. Underfitting
20. ReLU vs. Leaky ReLU vs. ELU vs. Sigmoid

## Top 20 Formulas
1. y = b + Σxᵢwᵢ
2. ŷ = xω + b
3. Loss: (y⁽ⁱ⁾−ŷ⁽ⁱ⁾)²
4. MSE: (1/2M)Σ(y⁽ⁱ⁾−ŷ⁽ⁱ⁾)²
5. GD: θⱼ ← θⱼ − α ∂𝒥/∂θⱼ
6. ∂𝒥/∂b = −(1/M)Σ(y−(ωx+b))
7. ∂𝒥/∂ω = −(1/M)Σ(y−(ωx+b))xᵀ
8. z = wᵀx + b
9. σ(z) = 1/(1+e^(−z))
10. σ′ = σ(1−σ)
11. BCE: −(1/M)Σ(y log ŷ + (1−y)log(1−ŷ))
12. Multinomial CE: −(1/M)ΣΣ y<k> log ŷ<k>
13. Chain: ∂𝒥/∂ω = ∂𝒥/∂ŷ · ∂ŷ/∂z · ∂z/∂ω
14. ∂𝒥/∂ω = (ŷ−y)xᵀ
15. softmax(zᵢ) = e^(zᵢ)/Σe^(zⱼ)
16. z⁽ⁱ⁾ = W⁽ⁱ⁾a⁽ⁱ⁻¹⁾+b⁽ⁱ⁾; a⁽ⁱ⁾ = g⁽ⁱ⁾(z⁽ⁱ⁾)
17. ŷ = g⁽²⁾(W⁽²⁾g⁽¹⁾(W⁽¹⁾x+b⁽¹⁾)+b⁽²⁾)
18. Decays: α₀·factor^(epoch/step); α₀e^(−rate·epoch); α₀/(1+rate·epoch)
19. x_norm = (x−μ)/σ²
20. Ridge λΣwⱼ²; ReLU max(0,x); Leaky max(0.1x,x)

## Top 20 Diagrams/Images
1. Biological neuron vs. single perceptron (slide 4)
2. Non-deep vs. deep feedforward network (slide 5)
3. Classical/ML/DL pipeline comparison (slide 6)
4. House-price data table + model (slides 8–10)
5. Regression line with prediction errors (slide 11)
6. Cost parabola with θ̂ (slides 14–16)
7. Parabola with negative/positive derivative tangents (slide 18)
8. GD steps hopping to convergence with θ₀ (slides 20–21)
9. 3D convex bowl J(ω,b) with trajectory (slide 26)
10. Convex vs. non-convex 3D surfaces (quiz slides 27/31)
11. Too low / just right / too high LR arrows (slides 30/95)
12. Logistic regression network with quantizer (slides 36–39)
13. Sigmoid curve crossing 0.5 at 0 (slide 40)
14. Cross-entropy fruit baskets + log plot (slide 41)
15. One-vs-rest 3-boundary figure & SoftMax architecture (slides 46–47)
16. Hidden layer "yellow box = logistic regression" (slides 50–51)
17. Hierarchical features animals figure (slide 55)
18. Forward/backward propagation diagrams + local gradient node (slides 56–60, 70–72)
19. P1/P2/P3 loss curve; overfitting temperature fits; exploding-gradient cliff (slides 74–78)
20. GD/SGD/MBGD blocks + three convergence curves (slides 89–90); normalization pipeline (slide 99)

## 10-Minute Revision Sheet
1. **NN = generalization of linear regression; neuron = logistic regression block.**
2. **ŷ = xω + b**; fit = minimize **MSE = (1/2M)Σ(y−ŷ)²** (loss = 1 point, cost = all points).
3. **5 assumptions:** Linearity, Independence, Homoscedasticity, Normality, No multicollinearity.
4. **GD:** init random θ₀ + pick α; repeat **θ ← θ − α∂𝒥/∂θ** until convergence; convex ⇒ global min guaranteed; steps shrink as slope flattens.
5. **α:** too high → sloshing; too low → slow; decay (step/exp/1/t) helps.
6. **Logistic:** z = wᵀx+b → **σ = 1/(1+e^(−z))** ∈ (0,1) → quantizer → {0,1}; cost = **cross-entropy**; gradient = **(ŷ−y)xᵀ**; σ′ = σ(1−σ).
7. **Multiclass:** one-vs-rest or **SoftMax** (e^z/Σe^z, sums to 1, one-hot + CE + argmax).
8. **Logistic limits** (linear boundaries etc.) → **NN with hidden layers** → hierarchical features.
9. **Training = forward (z, a, ŷ, cost) + backward (chain rule, update all W,b)**; gradients match parameter shapes.
10. **Hyperparameters:** network size, learning rate, activation function. Overfit ↔ too big; underfit ↔ too small.
11. **AFs:** non-linear, differentiable, bounded, sparse, efficient — ReLU max(0,x) etc.
12. **Tricks:** clipping (exploding), dropout/L1L2/early stop (overfit), batch norm (stability), Adam (default optimizer).
13. **GD all / SGD one (noisy, escapes local minima, outlier-sensitive) / mini-batch some (best of both)**.
14. **Normalize with train μ, σ² — reuse in production.**
15. Pipeline: **Framework → Model → Cost → Optimizer → Backprop → Hyperparameter tuning.** Golden rule: NNs = second-best at everything.

---

# Examples Repository

### 1. House Price Prediction (Linear Regression)
- **Topic:** Linear Regression
- **Original lecture context:** Slides 8–10 — "Predict house prices using house characteristics (features)"; table: area 120/80/100, rooms 3/3/6, location central/peripheral, price 500K/300/200K.
- **Simple explanation:** Historical house observations (features → price) train a model ŷ = xω + b that predicts prices of new houses.
- **Why important:** The running example defining data = historical observations, features, dependent variable, and model reliability for future predictions.
- **Possible exam questions:** Identify dependent/independent variables in the table; why is this regression and not classification?; write the model for 3 features.

### 2. House-Price Illustrations of the 5 Regression Assumptions
- **Topic:** Assumptions of Linear Regression
- **Original context:** Slide 12 — size↔price (linearity), houses selected randomly across the city (independence), price errors spread equally across price levels (homoscedasticity), most errors small (normality), bedrooms vs. bathrooms correlated (multicollinearity).
- **Simple explanation:** Each abstract assumption is grounded in the housing scenario.
- **Why important:** The exam will likely ask for assumptions **with examples**.
- **Possible exam questions:** Give an example of a multicollinearity violation in housing data; what does "equally difficult to predict the price" refer to?

### 3. 30,000 Genes Disease Prediction (Scale Motivation)
- **Topic:** GD → SGD motivation
- **Original context:** Slides 82–84 — 30,000 gene features, thousands→1 million data points; smallest logistic model needs ≥30,000 parameters; 1,000,000 terms per parameter per step; 1000 epochs → ≥30,000,000,000,000 terms; adding layers → >2,000,000 parameters.
- **Simple explanation:** Full-dataset gradient computation explodes combinatorially with features × data × epochs, making batch GD impractical.
- **Why important:** Justifies SGD and mini-batch GD.
- **Possible exam questions:** Why is batch GD impractical for genomic data?; compute the number of gradient terms for M points, N parameters, E epochs.

### 4. Breast Cancer Prediction (Logistic Regression & NN)
- **Topic:** Logistic Regression → Neural Networks
- **Original context:** Slides 35–39, 50–51, 56–59 — dataset with radius_0, texture_0, perimeter_2, age, cancer_type (0/1); network inputs Age=34, Radius_mean=2, Texture_mean=4; output probability of cancer 0.6; "Logistic Regression to predict breast cancer using features extracted from pathological images"; Benign 0 / malignant 1.
- **Simple explanation:** Real medical binary classification: features → sigmoid probability of malignancy → threshold to diagnose.
- **Why important:** THE running example for logistic regression, forward propagation and backpropagation.
- **Possible exam questions:** Given features and weights, compute z and ŷ; interpret ŷ = 0.6; what do labels 0/1 encode?

### 5. Action or Horror? (Binary Classification Teaser)
- **Topic:** Logistic Regression introduction
- **Original context:** Slide 34 — the popcorn-cat asks "Is this Action or Horror?" opening Episode 2.
- **Simple explanation:** Classifying a movie into one of two classes is a binary classification task — solved by logistic regression.
- **Why important:** Frames classification as predicting one of two alternatives.
- **Possible exam questions:** Is genre prediction (2 genres) regression or classification? Which model applies?

### 6. Strawberry & Apple Baskets (Cross-Entropy)
- **Topic:** Cross-entropy cost
- **Original context:** Slide 41 — minimize cross entropy = minimize entropy of predicting strawberries + minimize entropy of predicting apples; binary log plot alongside.
- **Simple explanation:** Binary cross-entropy is the sum of two per-class terms — one per fruit — each punished by the log of the predicted probability.
- **Why important:** Intuition for the two terms of the BCE formula.
- **Possible exam questions:** Map each basket to a term in the BCE formula; why does log punish confident wrong predictions?

### 7. SoftMax Numeric Example
- **Topic:** SoftMax regression
- **Original context:** Slide 47 — [1.30, 5.10, 2.20, 0.70, 1.10] → softmax → [0.02, 0.90, 0.05, 0.01, 0.02].
- **Simple explanation:** Exponentiating and normalizing turns raw scores into probabilities summing to 1; the biggest score dominates.
- **Why important:** Concrete demonstration of SoftMax properties — could be a compute-style question.
- **Possible exam questions:** Which class wins and with what probability?; verify the outputs sum to 1; what happens to small scores?

### 8. Penguin/Elephant/Kangaroo Hierarchical Features
- **Topic:** Why go deep
- **Original context:** Slide 55 — animal photos at the input; strokes → contours → animal shapes → labels; "increasingly complex features"; unsupervised/supervised learning sides.
- **Simple explanation:** Each deeper layer composes simpler features into more abstract ones until class-discriminative shapes emerge.
- **Why important:** The canonical picture of hierarchical feature learning.
- **Possible exam questions:** What do early vs. late layers learn?; define hierarchical features.

### 9. Temperature-vs-Date Overfitting
- **Topic:** Overfitting
- **Original context:** Slides 75–76 — scatter of daily temperature 2020-07→2022-01; overlaid jagged (overfit), red (good) and smooth green (underfit) fits.
- **Simple explanation:** A too-flexible model traces every noisy point instead of the seasonal trend — it memorizes rather than generalizes.
- **Why important:** Visual definition of overfitting: "high-capacity models easily memorize training data, reducing generalization."
- **Possible exam questions:** Identify overfit/underfit/good curves; which tricks remedy the jagged fit? (dropout, L1/L2, early stopping).

### 10. Outlier Costing SGD Time
- **Topic:** SGD sensitivity
- **Original context:** Slide 87 — red outlier point above the data: "This outlier will cost us some time if it got chosen."
- **Simple explanation:** Because SGD updates from one point, a picked outlier produces a large misleading update.
- **Why important:** Concrete illustration of SGD's outlier sensitivity.
- **Possible exam questions:** Why is SGD sensitive to outliers while GD is less so? (GD averages over all data.)

### 11. Mini-batch Colored-Groups Picture
- **Topic:** Mini-batch GD
- **Original context:** Slide 88 — blue/red/orange point groups: "Different mini-batches of data will be used instead to update the parameter instead of individual updates" → "more stable estimates for the parameters in fewer steps."
- **Simple explanation:** Each update averages over a small colored group of points, damping noise vs. SGD.
- **Why important:** Visual for the best-of-both-worlds claim.
- **Possible exam questions:** Explain how mini-batching stabilizes updates.

### 12. Normalization Before/After Scatter + Production Pipeline
- **Topic:** Normalizing input
- **Original context:** Slide 99 — scatter shifting to be origin-centered after (x−μ)/σ²; train/test pipeline where processing parameters feed test processing; "μ and σ become part of the model and must be the same in production."
- **Simple explanation:** Normalization is fit on training data and the exact parameters are reused downstream.
- **Why important:** Deployment-correctness rule; classic trap question.
- **Possible exam questions:** What goes wrong if production data is normalized with its own μ, σ?

### 13. Applications of Linear Regression (7 domains)
- **Topic:** Applications
- **Original context:** Slide 33 — Real Estate (house prices from size/location/bedrooms), Salary Estimation (experience/education/role), Sales Forecasting (ad spend/seasonality/trends), Risk Assessment (insurance claims from age/vehicle/history), Agriculture (crop yield from rainfall/fertilizer/temperature), Finance & Investment (stock returns from P/E, volume), Education Analytics (student performance from attendance/hours/socio-economics).
- **Simple explanation:** Regression fits any continuous-prediction domain.
- **Why important:** Application-identification questions.
- **Possible exam questions:** Give 3 applications of linear regression and their features/targets.

---

# Quiz Repository

### Quiz 1 — Why does the cost function have this shape (parabola or paraboloid)?
- **Original question:** "Why does it have this shape (parabola or paraboloid)?" (Slide 15)
- **Correct answer (lecture, slide 17):** Because it is a **quadratic function**.
- **Explanation:** MSE squares the errors; squaring a linear function of θ yields a quadratic in θ → parabola (1 parameter) / paraboloid (2+).
- **Related topic:** Loss/Cost functions (MSE).
- **Difficulty:** Easy

### Quiz 2 — Where are the parameters in the equations?
- **Original question:** "Where are the parameters in the equations below?" (Slide 15)
- **Correct answer (lecture):** In ŷ⁽ⁱ⁾, since ŷ = xω + b, where ω, b are the parameters represented as θ (weights ω and bias b). In the plot we assume one single parameter θ for simplification.
- **Explanation:** The cost only appears to depend on y and ŷ; the parameters hide inside the prediction ŷ.
- **Related topic:** MSE / model fitting.
- **Difficulty:** Medium

### Quiz 3 — Why a squared function and not simply the absolute mean or any other function?
- **Original question:** (Slide 15)
- **Correct answer (lecture):** MSE is a quadratic function — the further you go from the optimum, the bigger (quadratically) the MSE gets.
- **Explanation:** Quadratic growth punishes large deviations more strongly and gives a smooth curve suited to gradient-based optimization.
- **Related topic:** MSE.
- **Difficulty:** Medium

### Quiz 4 — How would you find the best parameters? Why not simply np.min?
- **Original question:** "How would you optimize or find the best parameters that give us the minimum loss (the smallest error between prediction and actual values)? Why don't we simply use np.min?" (Slide 16)
- **Correct answer (lecture):** Because np.min() (or any brute-force search) does not know the structure of the cost function and is not practical for continuous, high-dimensional optimization problems. We need optimization algorithms that **exploit the mathematical structure** of the cost function to find minima much faster.
- **Explanation:** GD uses the gradient (structure) to walk directly downhill instead of blindly evaluating infinitely many parameter combinations.
- **Related topic:** Gradient Descent motivation.
- **Difficulty:** Medium

### Quiz 5 — Why does the step size decrease with time?
- **Original question:** "Some questions for thought: Why does the step size decrease with time?" (Slide 21)
- **Inferred Answer:** Because the update is θ ← θ − α·gradient and the gradient's magnitude shrinks as the slope flattens approaching the minimum — so even with a constant learning rate, steps naturally get smaller.
- **Explanation:** Near θ̂ the derivative → 0, making α × gradient → 0.
- **Related topic:** Gradient Descent / gradient magnitude.
- **Difficulty:** Medium

### Quiz 6 — Why do we still add "hat" to the parameters even after the optimization converges?
- **Original question:** (Slide 21)
- **Inferred Answer:** The hat denotes an **estimate**. Even after convergence, θ̂ is the value estimated from data by an optimization procedure — not the true underlying parameter — so it keeps the hat.
- **Explanation:** Statistical convention: ŷ, θ̂ mark estimated/predicted quantities.
- **Related topic:** Notation / model fitting.
- **Difficulty:** Medium

### Quiz 7 — Convex loss + gradient descent: guaranteed global minimum? (appears twice, slides 27 & 31)
- **Original question:** "A loss function is perfectly convex. You run gradient descent. Are you guaranteed to find the global minimum? A. Yes, always B. No, not necessarily C. Only if learning rate is small D. Only with stochastic gradient descent"
- **Correct answer (per lecture emphasis):** **A — Yes** (slide 26: "In convex problems (like linear regression with MSE), it is guaranteed that we will move toward the global minimum since it is convex."). *Note:* strictly, an appropriately chosen learning rate is assumed — a wildly too-large α can oscillate, which is why C is a tempting distractor; the lecture's highlighted takeaway is the convexity guarantee. **Marked partly as Inferred Answer** since the slide deck itself doesn't print the letter.
- **Explanation:** A convex function has a single global minimum and no local minima/saddle traps, so following the negative gradient always moves toward it.
- **Related topic:** Convexity, GD.
- **Difficulty:** Medium

### Quiz 8 — What about having more parameters (weight w and bias b, or several weights)?
- **Original question:** (Slide 22, "We need to go deeper" meme)
- **Correct answer (lecture, slides 23–26):** Apply the same GD but update **each** parameter with its own partial derivative: ω = ω − α∂𝒥/∂ω and b = b − α∂𝒥/∂b; repeated for each weight when there are multiple covariates/features.
- **Explanation:** GD generalizes dimension-wise; the cost becomes a surface (paraboloid) and updates move both coordinates toward the bowl's bottom.
- **Related topic:** GD for multiple parameters.
- **Difficulty:** Medium

### Quiz 9 — What if we have more than two classes?
- **Original question:** (Slide 45, Multinomial classification)
- **Correct answer (lecture, slides 46–47):** Either **one-vs-rest** (a separate logistic regression model for each class) or **SoftMax regression** (SoftMax function producing a probability vector summing to 1, trained with cross-entropy on one-hot labels).
- **Explanation:** Binary sigmoid can't handle K classes directly; these two strategies extend it.
- **Related topic:** Multinomial classification.
- **Difficulty:** Easy

### Quiz 10 — Choosing the right network size: how deep should we go, how big each layer?
- **Original question:** "How deep should we go? And how big should each layer be?" (Slide 62)
- **Correct answer (lecture, slides 63–64):** Network size is a **hyperparameter** fixed before training: start small, gradually increase complexity based on validation performance, consider task complexity, use dropout for large networks, use early stopping; balance capacity — too big → overfitting, too small → underfitting.
- **Related topic:** Network size.
- **Difficulty:** Medium

### Quiz 11 — Homework: Benefits and Limitations of Gradient Descent
- **Original question:** Summarize benefits and limitations of GD (Slide 91 — "Homework: Summary").
- **Correct answer (lecture):** Benefits: easy to implement; scales with large parameter sets; efficient for convex problems (finds global minimum); simplicity. Limitations: slow convergence (whole dataset per iteration); sensitivity to learning rate; can get stuck in local minima of non-convex functions; computationally expensive for large datasets and models.
- **Related topic:** Gradient Descent.
- **Difficulty:** Medium

### Quiz 12 — Homework: What is Stochastic Gradient Descent?
- **Original question:** (Slide 92 — "Homework: Summary")
- **Correct answer (lecture):** SGD updates model parameters after **each data point** (or small batch) rather than all data points; core idea — use each data point's gradient individually to take small steps minimizing the loss. Benefits: faster updates, avoids local minima (random fluctuations, helpful in non-convex problems), memory efficiency. However: sensitive to noisy updates/outliers; slow as it requires an update after each iteration.
- **Related topic:** SGD.
- **Difficulty:** Medium

### Quiz 13 — Model still not working after scaling up: should we follow the same procedure with 2,000,000 parameters?
- **Original question:** "We did this but the model is still not working well!!! So we added more layers… now we have more than 2,000,000 parameters, should we follow the same procedure again?" (Slide 84)
- **Inferred Answer:** No — full-batch GD becomes computationally infeasible; use **stochastic** or **mini-batch gradient descent** (introduced immediately after) to update from single points or small batches.
- **Explanation:** This rhetorical question motivates SGD/mini-batch as the scalable alternative.
- **Related topic:** SGD / Mini-batch GD.
- **Difficulty:** Medium

---