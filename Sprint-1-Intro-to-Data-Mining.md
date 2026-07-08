# 📘 Sprint 1 | Introduction to Data Mining

> **How to use this file:** Study Parts 1–2 to *understand*, then drill Parts 3–4 (cheat sheet + question banks) to *pass*. Every **bold keyword** is a likely exam word. This lecture is an *introduction* — the exam will test **definitions, comparisons, classifications (data types, task types), and the project lifecycle**.

---

# PART 0 — The Big Picture (read this first)

This whole lecture answers 5 questions:

1. **What is Data Mining / Data Science?** (an interdisciplinary field to extract knowledge from data)
2. **Where is it used?** (almost every industry)
3. **How is a project done?** (the 7-stage lifecycle)
4. **What are the inputs?** (data — its *types* and its *representation*)
5. **What tasks & tools do we use?** (classification, clustering… with Python)

If you can retell those 5 answers, you understand the lecture. Everything below adds the exam-precise detail.

**Note on the slides:** Slides 1–9 are course logistics (professor intro, classroom rules, grading). Slides 10–36 are the real content. I cover *all* of them, but spend the most effort on the high-yield content slides.

---

# PART 1 — CONTENT SLIDES (Full Teaching)

---

## 🔹 SECTION A — Course Logistics (Slides 1–9)
*(Low exam value, but occasionally a professor sneaks in one MCQ about course rules. Memorize the 3 highlighted facts.)*

### 1. Core Idea
This is an *Introduction to Data Mining* course taught in **English**, using **Python** and common statistical Python frameworks. It uses **interactive teaching** — you solve small tasks in class.

### 2. Key Points (⭐ the 3 exam-worthy facts)
- ⭐ **80% active participation in exercises is MANDATORY** for exam eligibility.
- ⭐ Once assigned to a group, **switching groups is NOT allowed**.
- ⭐ The module is taught in **English** and programmed in **Python**.
- Blocks start at **10:00**; attending all blocks is *strongly recommended*.

### 3. Definitions
- **Pre-read topics** = math/ML background you should review: **vectors & matrices, matrix multiplication, vector norms, eigenvalues, derivatives, gradients, Jacobian matrix, exponential & logarithmic functions, Bayes' theorem, mean, variance, probability distribution, covariance matrix, PCA (dimensionality reduction)**.
- **Key ML terminology (pre-read):** Model, Backpropagation, Preprocessing, Feature, Target, Training, Performance evaluation, Overfitting/Underfitting.

### 4. Why It Matters
Data mining needs a math + programming foundation. The professor lists the prerequisites so you know what to revise.

### 6. Common Mistakes
- Thinking participation is optional — it is **80% mandatory**.
- Confusing "recommended" (attend blocks) with "mandatory" (80% exercises).

---

## 🔹 SECTION B — What is Data Mining? (Slides 10–13) ⭐⭐⭐ HIGH YIELD

### 1. Core Idea
**Data Mining** is the science of digging useful **knowledge and insights** out of large amounts of data. It is **interdisciplinary** — it borrows from computer science, statistics, and the subject area (domain) the data comes from. Think of it as *"turning raw data into useful answers."*

### 2. Key Points
- ⭐ **Official definition (memorize word-for-word feel):** *"Data mining is an **interdisciplinary** field that uses **scientific methods, processes, algorithms, and systems** to extract **knowledge and insights** from **structured and unstructured data** across a broad range of **application domains**."*
- The **Data Science Venn diagram** overlaps these fields:
  - **Computer Science** + **Math & Statistics** + **Domain Knowledge** = **Data Science** at the center.
  - Sub-overlaps: **Machine Learning** (CS ∩ Stats), **Software Development** (CS ∩ Domain), **Empirical Research** (Stats ∩ Domain).
- Famous quote (Josh Wills): *"A **Data Scientist** is a person who is **better at statistics than any software engineer**, and **better at software engineering than any statistician**."* → captures the interdisciplinary blend.

### 3. Definitions (exam-ready)
- **Data Mining:** Interdisciplinary field using scientific methods, processes, algorithms & systems to extract knowledge/insights from structured & unstructured data across many application domains.
- **Interdisciplinary:** Combining knowledge and methods from *several different fields* (e.g., CS + statistics + domain expertise).
- **Data Science:** The intersection of computer science, math/statistics, and domain knowledge used to gain insights from data. *(Often used interchangeably with data mining in this course.)*
- **Structured data:** organized, schema-based data. **Unstructured data:** no predefined format.
- **Application domain:** the real-world field the data belongs to (health, finance, etc.).

### 4. Why It Matters
Defines the *entire subject*. Every downstream topic (data types, lifecycle, tasks) exists to serve this goal: extract knowledge from data. The interdisciplinary point is a favorite exam question.

### 5. Easy Example
A hospital has 10 years of patient records. **Data mining** finds a hidden pattern: *"patients with high blood sugar + smoking history recover 3 days slower."* That *insight* came from combining **CS** (writing the code), **statistics** (the pattern math), and **domain knowledge** (a doctor knowing which features matter).

### 6. Common Mistakes
- ❌ Thinking data mining = only computer science. ✅ It's **interdisciplinary** (CS + stats + domain).
- ❌ Confusing "data" with "insight." Data is raw; the *insight/knowledge* is the output.
- ❌ Forgetting it handles **both structured AND unstructured** data.

### 9. Memory Trick
**"MAPS"** for the definition ingredients: **M**ethods, **A**lgorithms, **P**rocesses, **S**ystems → used to extract knowledge.
For the Venn diagram: **"C-M-D"** = **C**omputer science, **M**ath/stats, **D**omain knowledge = Data Science.
Josh Wills: *"Better than the engineer at stats, better than the statistician at engineering."*

---

## 🔹 SECTION C — Applications of Data Mining (Slide 14) ⭐⭐

### 1. Core Idea
Data mining is used **almost everywhere**. The slide gives 9 industry areas, each with example tasks. You don't need all 27 examples — remember the *areas* and 1–2 examples each.

### 2. Key Points — the 9 domains + signature examples
| Domain | Example tasks (remember 1–2) |
|---|---|
| **Manufacturing & Industry** | Predictive maintenance, quality control, supply chain optimization |
| **E-commerce & Retail** | Recommendation systems, dynamic pricing, inventory mgmt |
| **Social Media & Internet** | Spam filtering, fake news detection, influencer identification |
| **Cybersecurity** | Intrusion detection, malware detection, identity-theft prevention |
| **Science & Research** | Genomics/bioinformatics, climate analysis, astronomy |
| **Business & Marketing** | Customer segmentation, churn prediction, sentiment analysis |
| **Finance & Banking** | **Fraud detection**, risk management, algorithmic trading |
| **Healthcare & Medicine** | Disease prediction/diagnosis, drug discovery, patient monitoring |
| **Smart Cities & Transport** | Traffic prediction, route optimization, accident analysis |

### 4. Why It Matters
Shows data mining is *general-purpose*. Exam may give an example ("detecting credit-card fraud") and ask which **domain** or which **task type** it is.

### 5. Easy Example
Netflix suggesting your next show = **Recommendation system** → **E-commerce/Retail** (and the task type is *Recommendation*).

### 6. Common Mistakes
- Mixing up **domain** (the industry, e.g., Finance) with **task** (the technique, e.g., Anomaly Detection). Fraud detection is a *finance domain* problem solved by an *anomaly-detection task*.

### 9. Memory Trick
Fraud → Finance. Spam → Social media. Disease → Healthcare. Match the noun to the obvious industry.

---

## 🔹 SECTION D — Data Mining Project Lifecycle (Slides 15–17, 23, 25) ⭐⭐⭐ VERY HIGH YIELD
*(This diagram is repeated 5× in the slides — that repetition screams "EXAM QUESTION.")*

### 1. Core Idea
A data mining / data science project follows an **ordered sequence of stages** from start to end. A project *"may involve all or some of these stages,"* and it is **iterative** (arrows loop back to the data). Everything revolves around the central **database/data**.

### 2. Key Points — the 7 stages IN ORDER ⭐
1. **Problem understanding** — define what you're trying to solve.
2. **Explore and process** — look at the data, clean & preprocess it.
3. **Modelling & Analysis** — build the model / run algorithms.
4. **Communicate results** — present findings (storytelling!).
5. **Deploy & operationalize** — put the model into real use.
6. **Test and Monitoring** — check it keeps working over time.
7. **End.**

- The **data(base)** sits in the center — every stage reads from / writes to it.
- ⭐ Key phrases: *"may involve **all or some** of these stages"* and it is **iterative** (you can loop back).

### 3. Definitions
- **Project lifecycle:** the ordered set of stages a data mining project passes through from problem to deployment.
- **Deploy / operationalize:** making the finished model available for real-world use.
- **Monitoring:** continuously checking a deployed model still performs correctly.
- **Preprocessing:** cleaning and transforming raw data before modelling (part of *Explore and process*).

### 4. Why It Matters
It's the *roadmap* of the entire course (later sprints follow these stages). Guaranteed exam material — likely "put these stages in order" or "which stage does X belong to?"

### 5. Easy Example
Predicting student dropout:
1. **Problem understanding**: "predict who will drop out."
2. **Explore/process**: gather grades, attendance; clean missing values.
3. **Modelling**: train a classifier.
4. **Communicate**: show the dean a dashboard.
5. **Deploy**: integrate into the student portal.
6. **Monitor**: check accuracy each semester.

### 6. Common Mistakes
- ❌ Putting **Communicate results** *after* deployment. ✅ Order: Model → **Communicate** → Deploy → Monitor.
- ❌ Thinking it's strictly linear. ✅ It's **iterative** and you may skip stages ("all or some").
- ❌ Forgetting **Problem understanding** is FIRST (not data collection).

### 9. Memory Trick
**"Please Explore Models, Communicate, Deploy, Test"** →
**P**roblem, **E**xplore, **M**odelling, **C**ommunicate, **D**eploy, **T**est.
Mnemonic sentence: **"Purple Elephants Make Cute Dancing Tigers."**

---

## 🔹 SECTION E — Domain Knowledge (Slide 16) ⭐⭐

### 1. Core Idea
**Domain knowledge** = understanding the real-world subject the data comes from. It is *crucial* to a successful project because it guides you at every step.

### 2. Key Points — Domain knowledge does 3 things:
1. **Guides successful preprocessing** (you know which data is valid/relevant).
2. **Influences the modeling** (you pick the right features & methods).
3. **Provides the context to explain the results** (you can interpret findings correctly).
- ⭐ **Key takeaway (quote):** *"Always dive into the domain behind the data — it's the key to engineering meaningful **features**, focusing your analysis, and telling a **story** your audience understands."*

### 3. Definitions
- **Domain knowledge:** expertise about the specific field/subject the data belongs to (e.g., medicine for health data).
- **Feature engineering:** creating meaningful input variables (features) from raw data — done well only with domain knowledge.

### 4. Why It Matters
The same numbers mean different things in different fields. Domain knowledge prevents wrong conclusions and enables good **feature engineering**.

### 5. Easy Example
A "temperature of 39" is *normal* for a car engine but a *dangerous fever* for a human. Only **domain knowledge** tells you how to treat the value.

### 6. Common Mistakes
- Thinking a good algorithm alone gives good results. ✅ Without domain knowledge you may engineer meaningless features and misread results.

### 9. Memory Trick
Domain knowledge = **"P-M-C"**: **P**reprocessing, **M**odeling, **C**ontext (explaining results).

---

## 🔹 SECTION F — Data Representation (Slides 18–22) ⭐⭐⭐ HIGH YIELD

### 1. Core Idea
Computers only understand **numbers**. So before modelling, we must turn *any* kind of data (text, images, audio, molecules, graphs) into an **informative numerical representation** (usually **vectors**). Choosing a *good* representation is a **fundamental step** in data mining.

### 2. Key Points
- ⭐ **Definition:** *"To train a model, we must create an **informative and numerical representation of the data** that can be manipulated by electronic devices."*
- Data exists in **many formats**: social media, medical **imaging**, **EHR** (electronic health records), **genomics**, **knowledge graphs**, **bioactive molecules**, **textual data**, **vital signals** (like ECG).
- **Representation techniques by data type** (⭐ memorize this table):

| Data type | How it's represented |
|---|---|
| **Text** | **Word Embedding** (real-valued vectors); **TF-IDF**; **Word2Vec / GloVe** |
| **Images** | **Pixel values** |
| **Molecules** | **Graph** — each **atom = node**, each **bond = edge** |
| **Audio / speech** | **Spectrogram** (frequency spectrum over time) |
| **Graph/network data** | **Node embeddings** (e.g., **Node2Vec**) |

### 3. Definitions (exam-ready)
- **Data representation:** converting raw data into an informative, numerical form (vectors) a computer can process.
- **Word Embedding:** representing words as real-valued **vectors** so that **semantically similar words are positioned closely**; captures contextual relationships; can be **pre-trained** on large corpora (e.g., **Word2Vec, GloVe**).
- **TF-IDF (Term Frequency–Inverse Document Frequency):** converts text into numerical vectors reflecting **how important a word is** in a document. **TF** = frequency of a word *within one document*; **IDF** = how rare the word is *across the whole corpus*. Score = **TF × IDF**.
- **Spectrogram:** a visual representation of an audio signal's **frequency spectrum over time**; useful for speech/music analysis.
- **Node embedding (Node2Vec):** continuous **vector** features for nodes in a **graph** that capture the network's **structural information**.

### 4. Why It Matters
No model can run on raw text or images. The *quality* of the representation heavily determines the *quality* of results — "garbage representation → garbage model."

### 5. Easy Example (TF-IDF intuition)
The word "the" appears in *every* document → high TF but **IDF ≈ 0** → low importance. The word "diabetes" appears in *few* documents → high IDF → **high TF-IDF** → it's a *meaningful* word. That's how TF-IDF finds important words.

### 6. Common Mistakes
- ❌ TF-IDF gives common words high scores. ✅ **Common words get LOW scores** (IDF pulls them down); **rare, informative words get HIGH scores.**
- ❌ Word embeddings are random numbers. ✅ They place **similar words close together** in vector space.
- ❌ Confusing **TF** (within one document) with **IDF** (across all documents/corpus).
- ❌ Thinking representation is optional — it's a **fundamental step**.

### 9. Memory Trick
- **TF-IDF** = **"rare = important."** A word common everywhere = boring; a word rare across docs but frequent in *this* doc = key.
- **Embeddings** = *"meaning becomes distance"* — close vectors = similar meaning.
- Molecule = graph: **atoms are dots (nodes), bonds are lines (edges).**

---

## 🔹 SECTION G — Main Data Mining Tasks (Slide 24) ⭐⭐⭐ VERY HIGH YIELD
*(A classic exam table — "match the task to its definition/example." Learn all 11.)*

### 1. Core Idea
These are the **types of problems** data mining solves. Each has a one-line definition and a typical example. This is the single most testable table in the lecture.

### 2. Key Points — the 11 tasks ⭐

| Task | Definition | Example |
|---|---|---|
| **Classification** | Sorting data into **predefined categories** (labels) | Spam detection |
| **Association Analysis** | Finding **relationships between items** in large datasets | "People who buy bread also buy butter" |
| **Regression** | Predicting **continuous (numeric) values** | Stock prices, patient recovery time |
| **Clustering** | Grouping data by **similarity, WITHOUT predefined labels** | Customer segmentation |
| **Optimization** | Finding the **best solution within constraints** | Route planning, resource allocation |
| **Recommendation** | Suggesting items based on **user preferences** | Netflix / Amazon suggestions |
| **Anomaly Detection** | Identifying **unusual patterns / outliers** | Fraud detection, fault prediction |
| **Personalization** | **Tailoring** user experiences | Personalized marketing, adaptive learning |
| **Decision Support** | Informing **decision-making** via analysis | Medical diagnostics, business intelligence |
| **Simulation** | **Modeling complex systems** to predict behavior | Weather modeling, financial markets |
| **Text Mining** | Understanding/mining/generating **human language** | Sentiment analysis, translation |

### 3. Definitions — the 4 you MUST nail (most tested)
- **Classification:** predicting a **category/label** from a *fixed set* (supervised, **labeled** data). E.g., email → {spam, not-spam}.
- **Regression:** predicting a **continuous number** (supervised). E.g., predict house price = €340,500.
- **Clustering:** grouping similar items with **NO predefined labels** (unsupervised). E.g., auto-group customers.
- **Anomaly Detection:** flagging rare/unusual points (**outliers**). E.g., a €9,000 charge on a card that usually spends €50.

### 4. Why It Matters
This vocabulary is the backbone of the whole course and every downstream sprint (Sprints 5–7 = Classification, Clustering, Regression). Exam *will* test the difference between these.

### 5. Easy Example (to separate the confusing three)
- Predict **"cat or dog"** from a photo → **Classification** (category).
- Predict **"tomorrow's temperature = 24.5°C"** → **Regression** (number).
- **Auto-group** 1000 photos into piles that look alike, with no labels given → **Clustering**.

### 6. Common Mistakes ⭐ (examiners LOVE these traps)
- ❌ **Classification vs Regression:** Classification = **categories/labels**; Regression = **continuous numbers**. (Predicting a *price* = regression, not classification!)
- ❌ **Classification vs Clustering:** Classification uses **predefined labels** (supervised); Clustering has **NO labels** (unsupervised).
- ❌ **Clustering vs Association:** Clustering groups **records/objects**; Association finds relationships **between items** (like market-basket rules).
- ❌ Calling fraud detection "classification" — the slide files it under **Anomaly Detection**.

### 9. Memory Trick
- **Class**ification → **Class**es (categories).
- **Regr**ession → **R**eal numbers.
- **Clu**stering → **Clu**mps with no labels ("clueless" about labels → unsupervised).
- **Anomaly** → **A**bnormal / outlier.
- **Association** → items that go together ("**A** goes with **B**").

---

## 🔹 SECTION H — Storytelling & Data Visualization (Slide 26) ⭐⭐

### 1. Core Idea
Getting a result isn't enough — you must **communicate** it. **Storytelling with data** turns numbers into a message your audience understands, using the right **visualizations**.

### 2. Key Points — 3 key components of storytelling ⭐
1. **Data** — the evidence / facts.
2. **Visualization** — charts, graphs, dashboards.
3. **Narrative** — the storyline guiding interpretation (*what happened, why, what now?*).
- Reference: *"Storytelling with Data," Cole Nussbaumer Knaflic.*
- **Common visualization types** shown: simple text, **scatterplot**, **vertical/horizontal bar**, **table**, **line**, **stacked bar**, **heatmap**, **slopegraph**, **waterfall**, **square area**.

### 3. Definitions
- **Storytelling with data:** combining **data + visualization + narrative** to communicate insights clearly and persuasively.
- **Narrative:** the explanatory storyline answering *what happened, why, and what to do now.*

### 4. Why It Matters
It's the **Communicate results** stage of the lifecycle. Insights that no one understands are useless. Storytelling is listed as a *crucial career skill*.

### 5. Easy Example
Instead of a raw table of 12 monthly sales numbers, you show a **line chart** with a note: *"Sales dropped 30% in July (why) after the price rise; we recommend reverting it (what now)."* = data + viz + narrative.

### 6. Common Mistakes
- Thinking a chart alone is storytelling. ✅ You also need the **narrative** (the "so what").

### 9. Memory Trick
Storytelling = **"D-V-N"**: **D**ata, **V**isualization, **N**arrative.

---

## 🔹 SECTION I — Types of Data (Slides 27–29) ⭐⭐⭐ VERY HIGH YIELD
*(Two separate classifications here — don't mix them up! Both are heavily tested.)*

### Classification #1 — By measurement level (Slide 28)

### 1. Core Idea
Data splits into two big families: **Qualitative (Categorical)** and **Quantitative (Numerical)**. Each splits again into two.

### 2. Key Points — the tree ⭐
```
DATA
├── Qualitative (Categorical)
│   ├── Nominal   → categories with NO order   (Gender, Hair color, Ethnicity)
│   └── Ordinal   → categories WITH order       (Reviews, Grades, Status)
└── Quantitative (Numerical)
    ├── Discrete   → countable whole numbers      (No. of students, Population, No. of houses)
    └── Continuous → measurable, any value in range (Height, Speed of cars, Weight)
```

### 3. Definitions (exam-ready)
- **Qualitative / Categorical data:** describes **qualities or categories** (non-numeric meaning).
- **Nominal:** categories with **no natural order** (e.g., gender, hair color, ethnicity).
- **Ordinal:** categories **with a meaningful order/rank** but unequal/unknown gaps (e.g., grades A>B>C, ratings, status).
- **Quantitative / Numerical data:** expressed as **numbers** you can do math on.
- **Discrete:** **countable** values, usually whole numbers (e.g., number of houses).
- **Continuous:** **measurable** values that can take *any* value in a range, including decimals (e.g., height, weight, speed).

### 6. Common Mistakes ⭐
- ❌ **Nominal vs Ordinal:** Nominal = **no order** (colors); Ordinal = **ordered** (grades, small/medium/large).
- ❌ **Discrete vs Continuous:** Discrete = **count** (5 children); Continuous = **measure** (5.3 kg). You can't have 2.5 students (discrete) but you can have 2.5 kg (continuous).
- ❌ Numbers ≠ always quantitative: a **ZIP code** or **jersey number** is numeric-looking but **nominal** (no math meaning).

### 9. Memory Trick
- **N**ominal = **N**o order (**N**ames/labels).
- **Or**dinal = **Or**der matters (**rank**).
- **Dis**crete = **counted dots** (whole).
- **Continuous** = **continuous measuring tape** (any value).

---

### Classification #2 — By structure (Slide 29)

### 1. Core Idea
Based on *how organized* the data is, it's **Structured**, **Unstructured**, or **Semi-structured**.

### 2. Key Points — the 3 structures ⭐

| Type | Definition | Examples |
|---|---|---|
| **Structured** | Organized data that **follows a predefined schema**, stored in **relational databases** | Customer info tables, transaction records |
| **Unstructured** | Data with **no predefined format**; harder to collect/process/analyze | Emails, videos, social media posts |
| **Semi-structured** | Not in a relational DB, but has **some organizational tags/markers** | **JSON, XML**, graph data |

### 3. Definitions
- **Structured data:** adheres to a **predefined schema**; relational-database friendly.
- **Unstructured data:** **no predefined format** (text, images, video) — the hardest to analyze.
- **Semi-structured data:** has **tags/markers** giving *partial* structure but not a full relational schema (JSON, XML).

### 6. Common Mistakes ⭐
- ❌ Calling JSON/XML "structured." ✅ They're **semi-structured** (tags, but no fixed relational schema).
- ❌ Calling emails "structured." ✅ Emails are **unstructured**.
- ❌ Confusing this (**structure**) classification with the (**measurement-level**) one. They're two *different* ways to classify data.

### 9. Memory Trick
- **Structured** = **S**chema / **S**preadsheet / **S**QL table.
- **Semi** = **tags** (JSON/XML **markers**).
- **Unstructured** = **wild** (emails, videos) — no rules.

---

## 🔹 SECTION J — Tools & Frameworks (Slide 31) ⭐⭐

### 1. Core Idea
Data miners work mostly in **Python**. Each job (data handling, ML, big data, LLMs, deep learning, tuning) has go-to libraries.

### 2. Key Points — the tool table ⭐

| Category | Python tools |
|---|---|
| **Essential packages** (data handling/plotting) | **NumPy, Pandas, Matplotlib, Seaborn** |
| **Machine Learning** | **Scikit-learn, XGBoost, RAPIDS** |
| **Big Data Analytics** | **Apache Spark (PySpark)** |
| **Large Language Models** | **Hugging Face Transformers, OpenAI GPT** |
| **Deep Learning** | **TensorFlow, PyTorch** |
| **Hyperparameter Optimization** | **Optuna, Hyperopt, GridSearchCV** |
| **Non-Python alternatives** | **MATLAB, RapidMiner, R** |

### 3. Definitions (quick)
- **NumPy** = numerical arrays; **Pandas** = tables/dataframes; **Matplotlib/Seaborn** = plotting.
- **Scikit-learn** = classic ML; **XGBoost** = gradient-boosted trees.
- **PySpark** = big-data processing; **TensorFlow/PyTorch** = deep learning.
- **Hyperparameter optimization** = automatically finding the best model settings (Optuna, GridSearchCV).

### 6. Common Mistakes
- ❌ TensorFlow/PyTorch for *classic* ML. ✅ They're for **deep learning**; classic ML = **Scikit-learn**.
- ❌ Thinking Python is the *only* option. ✅ Alternatives: **MATLAB, RapidMiner, R**.

### 9. Memory Trick
Map the job to the tool: **Plot → Matplotlib/Seaborn; Table → Pandas; ML → Scikit-learn; Deep Learning → PyTorch/TensorFlow; Big Data → Spark; Tuning → Optuna.**

---

## 🔹 SECTION K — Career & Course Roadmap (Slides 33–35)
*(Low-medium exam value; a professor might ask "list the sprints" or "career options.")*

### 1. Core Idea
A successful AI career = (1) **fundamentals** (CS, math, programming) → (2) **build a foundation** in specialized fields → (3) **deepen** in a passion area & follow trends.

### 2. Key Points
- **Foundation fields (the wheel):** Machine & Deep Learning, **NLP**, Cloud Computing, Data Engineering, **ML DevOps (MLOps)**, Ethical & Responsible AI, Data Analysis & Visualization, **Storytelling skills**.
- **Career options:** AI Engineer, Data Analyst, Data Scientist, Robotics/Automation Expert, **ML DevOps (ML Engineer)**, Data Engineer, Software Engineer.
- **Course structure — 12 Sprints:**
  1. Intro to Data Mining · 2. Data Mining Workflow · 3. Processing & Dimensionality Reduction · 4. Prototyping & Deployment · 5. **Classification** · 6. **Clustering** · 7. **Regression** · 8. **Neural Networks** · 9. **GenAI / RAG** · 10. Explainability · 11. Ethical & Responsible DM · 12. Wrap-up.

### 6. Common Mistakes
- Mixing the order of the 3 career steps: **Fundamentals → Foundation → Deepen.**

### 9. Memory Trick
Career path = **"1-2-3": 1** Fundamentals, **2** Foundation, **3** Deepen.

---

# PART 2 — COMPARISON TABLES (Exam Gold) ⭐⭐⭐

### Table 1 — Classification vs Regression vs Clustering (MOST TESTED)
| | Classification | Regression | Clustering |
|---|---|---|---|
| **Output** | Category / label | Continuous number | Groups (no labels) |
| **Labels needed?** | Yes (supervised) | Yes (supervised) | **No (unsupervised)** |
| **Example** | Spam / not spam | Predict price 340.5 | Segment customers |
| **Question it answers** | "Which class?" | "How much / how many?" | "Which items are similar?" |

### Table 2 — Qualitative vs Quantitative Data
| | Qualitative (Categorical) | Quantitative (Numerical) |
|---|---|---|
| **Meaning** | Categories/qualities | Numbers you can compute |
| **Sub-types** | Nominal (no order), Ordinal (ordered) | Discrete (count), Continuous (measure) |
| **Examples** | Gender, grades | No. of houses, weight |

### Table 3 — Nominal vs Ordinal
| Nominal | Ordinal |
|---|---|
| No order between categories | Meaningful order/rank |
| Gender, hair color, ethnicity | Grades, reviews, status |

### Table 4 — Discrete vs Continuous
| Discrete | Continuous |
|---|---|
| Countable, whole numbers | Measurable, any value (decimals) |
| No. of students, population | Height, weight, speed |

### Table 5 — Structured vs Unstructured vs Semi-structured
| Structured | Unstructured | Semi-structured |
|---|---|---|
| Predefined schema, relational DB | No predefined format | Tags/markers, not relational |
| Tables, transactions | Emails, videos, social posts | JSON, XML, graphs |

### Table 6 — TF vs IDF
| TF (Term Frequency) | IDF (Inverse Document Frequency) |
|---|---|
| How often a word appears **in one document** | How **rare** a word is **across all documents (corpus)** |
| High for repeated words | High for rare words, low for common words |
| **TF-IDF = TF × IDF** → highlights important, distinctive words | |

---

# PART 3 — ONE-PAGE CHEAT SHEET ⭐ (print this)

**DATA MINING** = *interdisciplinary field using scientific **methods, processes, algorithms, systems** to extract **knowledge & insights** from **structured & unstructured** data across **application domains**.*

**Data Science Venn** = Computer Science + Math/Statistics + Domain Knowledge.
*Josh Wills:* better at stats than any engineer, better at engineering than any statistician.

**LIFECYCLE (7 stages, in order):** Problem understanding → Explore & process → Modelling & Analysis → Communicate results → Deploy & operationalize → Test & Monitoring → End. *(iterative; "all or some"; data in center)*
Mnemonic: **P**urple **E**lephants **M**ake **C**ute **D**ancing **T**igers.

**DOMAIN KNOWLEDGE** does 3 things: guides **Preprocessing**, influences **Modeling**, gives **Context** to explain results.

**DATA REPRESENTATION** = make data **numerical** (vectors).
- Text → Word Embedding / **TF-IDF** / Word2Vec / GloVe
- Images → pixels · Audio → **spectrogram** · Molecule → graph (atom=node, bond=edge) · Graph → Node2Vec
- **TF-IDF:** TF (within doc) × IDF (rareness across corpus). *Rare word = important.*
- **Embeddings:** similar words → close vectors.

**11 TASKS:** Classification (labels), Regression (numbers), Clustering (no labels), Association (item relations), Optimization, Recommendation, Anomaly detection (outliers), Personalization, Decision support, Simulation, Text mining.

**STORYTELLING = Data + Visualization + Narrative** (what happened, why, what now?).

**DATA TYPES (measurement):** Qualitative→{Nominal=no order, Ordinal=ordered}; Quantitative→{Discrete=count, Continuous=measure}.
**DATA TYPES (structure):** Structured (schema/DB) · Unstructured (emails/video) · Semi-structured (JSON/XML).

**TOOLS:** Pandas/NumPy (data), Matplotlib/Seaborn (plots), Scikit-learn/XGBoost (ML), PySpark (big data), TensorFlow/PyTorch (deep learning), Hugging Face/GPT (LLMs), Optuna/GridSearchCV (tuning). Non-Python: MATLAB, RapidMiner, R.

**RULES:** 80% exercise participation mandatory · no group switching · English + Python.

---

# PART 4 — 30 MOST IMPORTANT EXAM QUESTIONS (with answers)

1. **Define data mining.** — An interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data across many application domains.
2. **Why is data mining called "interdisciplinary"?** — Because it combines computer science, mathematics/statistics, and domain knowledge.
3. **Name the three fields overlapping in the Data Science Venn diagram.** — Computer Science, Math & Statistics, Domain Knowledge.
4. **List the 7 stages of the data mining lifecycle in order.** — Problem understanding → Explore & process → Modelling & Analysis → Communicate results → Deploy & operationalize → Test & Monitoring → End.
5. **Is the lifecycle linear or iterative?** — Iterative; a project may involve all or some stages and can loop back to the data.
6. **What are the 3 roles of domain knowledge?** — Guides preprocessing, influences modeling, provides context to explain results.
7. **Why must data be represented numerically?** — Because models/computers can only manipulate numerical (vector) representations.
8. **How is text data represented?** — Via word embeddings (Word2Vec, GloVe) or TF-IDF vectors.
9. **What is TF-IDF?** — A method converting text to numeric vectors reflecting word importance; TF = frequency within a document, IDF = rareness across the corpus; score = TF × IDF.
10. **In TF-IDF, do common words score high or low? Why?** — Low, because IDF is small for words appearing in many documents.
11. **What is a word embedding?** — A representation mapping words to real-valued vectors where semantically similar words lie close together.
12. **How is a molecule represented for data mining?** — As a graph: atoms = nodes, bonds = edges.
13. **How is audio represented?** — As a spectrogram (frequency spectrum over time).
14. **What is Node2Vec used for?** — Learning node embeddings (vector features) that capture a graph's structural information.
15. **Define classification and give an example.** — Sorting data into predefined categories; e.g., spam detection.
16. **Define regression and give an example.** — Predicting continuous numeric values; e.g., stock prices or recovery time.
17. **Define clustering and how it differs from classification.** — Grouping by similarity without predefined labels (unsupervised); classification uses predefined labels (supervised).
18. **Define anomaly detection with an example.** — Identifying unusual patterns/outliers; e.g., fraud detection.
19. **What is association analysis?** — Identifying relationships between items in large datasets (e.g., market-basket "bread → butter").
20. **What are the 3 components of data storytelling?** — Data, Visualization, Narrative.
21. **Name the two main measurement-level categories of data.** — Qualitative (categorical) and Quantitative (numerical).
22. **Difference between nominal and ordinal data?** — Nominal has no order (gender); ordinal has a meaningful order (grades).
23. **Difference between discrete and continuous data?** — Discrete = countable whole numbers (no. of houses); continuous = measurable any value (weight).
24. **Define structured, unstructured, and semi-structured data.** — Structured: predefined schema/relational DB; Unstructured: no predefined format (emails, video); Semi-structured: tags/markers, not relational (JSON, XML).
25. **Which data type is JSON/XML?** — Semi-structured.
26. **Name the main Python libraries for classic machine learning.** — Scikit-learn, XGBoost, RAPIDS.
27. **Which libraries are used for deep learning?** — TensorFlow and PyTorch.
28. **What framework is used for big-data analytics?** — Apache Spark (PySpark).
29. **List 3 application domains of data mining.** — E.g., Healthcare (disease prediction), Finance (fraud detection), E-commerce (recommendation).
30. **What is the mandatory participation requirement for exam eligibility?** — 80% active participation in exercises.

---

# PART 5 — 30 MOST IMPORTANT MCQs (with answers)

1. Data mining is best described as a(n) ___ field.
   A) purely statistical B) purely computer-science C) **interdisciplinary** D) purely mathematical — **C**
2. Which is NOT in the Data Science Venn diagram center?
   A) Computer Science B) Math & Statistics C) Domain Knowledge D) **Marketing** — **D**
3. The FIRST stage of the data mining lifecycle is:
   A) Modelling B) **Problem understanding** C) Deploy D) Explore & process — **B**
4. Which stage comes immediately AFTER "Modelling & Analysis"?
   A) Deploy B) Test C) **Communicate results** D) Problem understanding — **C**
5. The data mining lifecycle is:
   A) strictly linear, never repeated B) **iterative and may use all or some stages** C) only 3 fixed stages D) done without data — **B**
6. Domain knowledge helps with all EXCEPT:
   A) preprocessing B) modeling C) explaining results D) **replacing the need for data** — **D**
7. Converting text into importance-weighted numeric vectors is:
   A) **TF-IDF** B) Node2Vec C) Spectrogram D) Pixel encoding — **A**
8. In TF-IDF, a word appearing in almost every document gets:
   A) high IDF B) **low IDF** C) infinite TF D) negative TF — **B**
9. Word embeddings place semantically similar words:
   A) far apart B) randomly C) **close together** D) at the origin — **C**
10. A molecule is typically represented as a graph where nodes are:
    A) bonds B) **atoms** C) electrons D) pixels — **B**
11. A spectrogram represents:
    A) text importance B) graph structure C) **audio frequency over time** D) pixel intensity — **C**
12. Node2Vec produces:
    A) pixel maps B) **node embeddings for graphs** C) spectrograms D) TF-IDF scores — **B**
13. Predicting whether an email is spam is:
    A) Regression B) Clustering C) **Classification** D) Simulation — **C**
14. Predicting a house's price (a number) is:
    A) Classification B) **Regression** C) Clustering D) Association — **B**
15. Grouping customers with NO predefined labels is:
    A) Classification B) **Clustering** C) Regression D) Recommendation — **B**
16. Fraud detection is listed under which task?
    A) Regression B) Clustering C) **Anomaly Detection** D) Simulation — **C**
17. "Customers who buy X also buy Y" is:
    A) **Association analysis** B) Clustering C) Regression D) Personalization — **A**
18. Netflix suggesting shows is which task?
    A) Classification B) **Recommendation** C) Clustering D) Simulation — **B**
19. The 3 components of storytelling with data are:
    A) Data, Model, Code B) **Data, Visualization, Narrative** C) Text, Image, Audio D) TF, IDF, Vector — **B**
20. Gender and hair color are examples of ___ data.
    A) Ordinal B) Discrete C) **Nominal** D) Continuous — **C**
21. Grades (A > B > C) are an example of ___ data.
    A) Nominal B) **Ordinal** C) Continuous D) Structured — **B**
22. "Number of houses" is:
    A) Continuous B) **Discrete** C) Ordinal D) Nominal — **B**
23. The weight of a person is:
    A) Discrete B) Nominal C) **Continuous** D) Ordinal — **C**
24. JSON and XML files are:
    A) Structured B) Unstructured C) **Semi-structured** D) Continuous — **C**
25. Emails and videos are:
    A) Structured B) **Unstructured** C) Semi-structured D) Nominal — **B**
26. Data stored in relational databases with a schema is:
    A) **Structured** B) Unstructured C) Semi-structured D) Ordinal — **A**
27. Which library is for classic machine learning?
    A) TensorFlow B) **Scikit-learn** C) Matplotlib D) PySpark — **B**
28. Which pair is used for deep learning?
    A) Pandas & NumPy B) **TensorFlow & PyTorch** C) Optuna & Hyperopt D) Spark & RAPIDS — **B**
29. Big-data analytics in Python typically uses:
    A) Seaborn B) **Apache Spark (PySpark)** C) GridSearchCV D) GloVe — **B**
30. Mandatory participation for exam eligibility is:
    A) 50% B) 60% C) 70% D) **80%** — **D**

---

# PART 6 — CONCEPTS PROFESSORS FREQUENTLY ASK ⭐

1. **The definition of data mining** (word-for-word feel) + why it's *interdisciplinary*.
2. **The 7-stage lifecycle in correct order** (repeated 5× in slides = almost certain).
3. **Classification vs Regression vs Clustering** (the #1 trap: labels vs numbers vs no-labels).
4. **Data types — measurement level** (Nominal/Ordinal/Discrete/Continuous, with a "classify this example" question).
5. **Data types — structure** (Structured/Unstructured/Semi-structured, especially "what is JSON?").
6. **TF-IDF** (what TF and IDF each mean; why rare words score high).
7. **Data representation per data type** (text→embedding, image→pixels, molecule→graph, audio→spectrogram).
8. **The 11 task types** — match task ↔ definition ↔ example.
9. **Domain knowledge's 3 roles.**
10. **Storytelling = Data + Visualization + Narrative.**
11. **Tool → job matching** (Scikit-learn=ML, PyTorch=DL, Pandas=data, Spark=big data).

---

# PART 7 — "LAST 10 MINUTES BEFORE EXAM" 🚨

Read ONLY this if time is short:

- **Data mining** = interdisciplinary (CS + Stats + Domain) → extract **knowledge/insights** from **structured & unstructured** data.
- **Lifecycle (in order):** Problem → Explore/Process → Model → **Communicate** → Deploy → Monitor → End. *Iterative, all-or-some.* → *"Purple Elephants Make Cute Dancing Tigers."*
- **Domain knowledge** → helps Preprocessing, Modeling, Context.
- **Representation:** text→**embeddings/TF-IDF**, image→pixels, molecule→**graph (atom=node, bond=edge)**, audio→**spectrogram**, graph→**Node2Vec**.
- **TF-IDF:** TF (in one doc) × IDF (rare across corpus). **Rare = important.**
- **Tasks:** Classification=**labels**, Regression=**numbers**, Clustering=**no labels**, Anomaly=**outliers**, Association=**item pairs**.
- **Storytelling = Data + Visualization + Narrative.**
- **Data types (measurement):** Qualitative {Nominal=no order, Ordinal=ordered}; Quantitative {Discrete=count, Continuous=measure}.
- **Data types (structure):** Structured=schema/DB; Unstructured=emails/video; Semi=**JSON/XML**.
- **Tools:** Pandas/NumPy=data, Scikit-learn=ML, PyTorch/TensorFlow=DL, PySpark=big data, Optuna=tuning.
- **Rules:** **80%** participation mandatory; English + Python.

**Common traps to avoid:**
1. Regression predicts **numbers**, not categories.
2. Clustering has **NO labels**; classification does.
3. JSON/XML = **semi-structured** (not structured).
4. Common words get **LOW** TF-IDF.
5. Molecule graph: **atom = node**, bond = edge (not reversed).
