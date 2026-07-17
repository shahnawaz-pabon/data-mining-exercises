# 📖 Lecture: Sprint1_Intro_2_DataMining_Msc_HIS_SS26

**Course:** Introduction to Data Mining — Prof. Dr. Matteo Marouf (Frankfurt University of Applied Sciences)
**Sprint:** 1 — Introduction to Data Mining

---

# 1. Lecture Overview

## Main Topics

1. **What is Data Mining** — definition and interdisciplinary nature
2. **Data Science, Data Mining & Big Data Analytics** — how the terms relate
3. **Applications of Data Mining** — 9 application domains with examples
4. **Data Mining Project Life Cycle** — the 7-stage pipeline (repeated **4 times** in the slides → very exam-relevant!)
5. **Domain Knowledge** — its crucial role in data mining projects
6. **Data Representation** — turning raw data into informative numerical form (Word Embeddings, Pixel values, Graphs, Spectrograms, Node2Vec, TF-IDF, Word2Vec/GloVe)
7. **Main Task Types in the Field** — Classification, Regression, Clustering, Association Analysis, Anomaly Detection, etc.
8. **Storytelling** — communicating results (Data + Visualization + Narrative)
9. **Types of Data** — Qualitative (Nominal/Ordinal) vs Quantitative (Discrete/Continuous); Structured vs Unstructured vs Semi-Structured
10. **Tools & Frameworks** — Python ecosystem for data mining
11. **Career & Course Outline** — AI career paths, the 12 sprints

## Learning Objectives

- Define data mining and explain why it is **interdisciplinary**
- Name application areas of data mining across industries
- Recall and **order** the stages of the data mining project life cycle
- Explain why **domain knowledge** and **data representation** are fundamental
- Distinguish the main analysis task types (classification, clustering, regression, …)
- Classify data by **measurement type** and by **structure**
- Know the three components of data storytelling

## Important Takeaways

- Data mining = extracting **knowledge and insights** from **structured and unstructured** data using scientific methods, processes, algorithms, and systems.
- The project life cycle was shown **four times** — memorize its order!
- **Key takeaway slide (verbatim message):** "Always dive into the domain behind the data — it's the key to engineering meaningful features, focusing your analysis, and telling a story your audience understands."
- To train a model, data must be converted into an **informative and numerical representation**.

---

# 2. Topic-wise Notes

## Topic 1: What is Data Mining?

### Definition
Data mining is an **interdisciplinary field** that uses scientific methods, processes, algorithms, and systems to **extract knowledge and insights** from **structured and unstructured data** across many application domains.

### Simple Example
A supermarket analyzes millions of receipts to discover that customers who buy diapers often also buy beer — knowledge it uses to redesign shelf placement.

### Why is it Important?
It turns huge amounts of raw data into actionable knowledge that drives decisions in business, science, healthcare, and more.

### Key Points
- Uses **scientific methods, processes, algorithms and systems**
- Works on **both structured and unstructured** data
- Applied across a **broad range of application domains**
- Closely related to **Data Science** and **Big Data Analytics**
- It is **interdisciplinary** — it combines statistics, computer science, machine learning, databases, visualization, and **domain knowledge**

### Common Mistakes
- Thinking data mining works only on structured/tabular data — it explicitly includes **unstructured** data (text, video, images).
- Confusing data mining (extracting knowledge) with simply storing or querying data (databases).
- Believing data mining is a single discipline — it is a **combination** of several fields.

### Comparison

| Aspect | Data Mining | Data Science | Big Data Analytics |
|---|---|---|---|
| Focus | Extracting knowledge/patterns from data | Broad umbrella: scientific insight from data | Analyzing very large data volumes |
| Relation | Core technique inside data science | Interdisciplinary parent field | Emphasis on scale/volume |
| In this lecture | Used almost interchangeably — "data mining (data science)" | Same interdisciplinary definition | Shown alongside as related terms |

### Important Keywords
- **Interdisciplinary** — combines multiple fields (statistics, CS, ML, domain expertise).
- **Knowledge extraction** — the goal of data mining.
- **Structured / Unstructured data** — both are inputs to data mining.

### Memory Tips
Think **"MINE = Methods, Insights, Numerous domains, Extraction"** — data mining uses **M**ethods to get **I**nsights across **N**umerous domains by **E**xtracting knowledge.

---

## Topic 2: Applications of Data Mining

### Definition
Data mining is applied wherever large amounts of data can be turned into useful predictions or insights — from factories to hospitals to social media.

### Simple Example
Netflix suggesting your next movie = a **recommendation system**, one of the e-commerce/retail applications.

### Why is it Important?
Exam questions often ask you to **match an application to its domain** or to name examples per industry.

### Key Points — The 9 Domains (memorize the table!)

| Domain | Applications |
|---|---|
| **Manufacturing & Industry** | Predictive maintenance, Quality control, Supply chain optimization |
| **E-commerce & Retail** | Recommendation systems, Dynamic pricing, Inventory management |
| **Social Media & Internet** | Spam filtering, Fake news detection, Influencer identification |
| **Cybersecurity** | Intrusion detection, Malware detection, Identity theft prevention |
| **Science & Research** | Genomics & bioinformatics, Climate change analysis, Astronomy |
| **Business & Marketing** | Process optimization, Customer segmentation, Churn prediction, Sentiment analysis |
| **Finance & Banking** | Fraud detection, Risk management, Algorithmic trading |
| **Healthcare & Medicine** | Disease prediction & diagnosis, Drug discovery, Patient monitoring |
| **Smart Cities & Transportation** | Traffic prediction, Route optimization, Accident analysis |

### Common Mistakes
- Placing **fraud detection** under cybersecurity — in the lecture it belongs to **Finance & Banking** (cybersecurity has *intrusion/malware detection*).
- Placing **churn prediction** under retail — it is listed under **Business & Marketing**.
- Confusing **spam filtering** (Social Media & Internet) with **malware detection** (Cybersecurity).

### Important Keywords
- **Predictive maintenance** — predicting machine failure before it happens.
- **Churn prediction** — predicting which customers will leave.
- **Customer segmentation** — grouping customers by behavior (a clustering task).
- **Fraud detection** — spotting suspicious financial transactions (an anomaly detection task).
- **Recommendation system** — suggesting items based on user preferences.
- **Sentiment analysis** — determining opinion/emotion in text.

### Memory Tips
Group by "**M**achines, **S**hops, **S**ocial, **S**ecurity, **S**cience, **B**usiness, **B**anks, **B**odies, **C**ities" → *M-4S-2B+B-C*. Or remember: **each domain has exactly 3 examples** (Business & Marketing has 4).

---

## Topic 3: Data Mining Project Life Cycle ⭐ (shown 4×!)

### Definition
The project life cycle is the ordered sequence of stages a data mining project goes through, from understanding the problem to monitoring the deployed solution. A project **may involve all or some** of these stages.

### The Pipeline (memorize the exact order!)

```
Start → 1. Problem understanding
      → 2. Explore and process
      → 3. Modelling & Analysis
      → 4. Communicate results
      → 5. Deploy & operationalize
      → 6. Test and Monitoring
      → End
```

### Simple Example
A hospital wants to predict patient readmission: first they define the problem, then explore/clean patient records, build a model, present results to doctors, deploy it in the hospital IT system, and continuously monitor its accuracy.

### Why is it Important?
This diagram appeared on **four separate slides** — it is almost certainly an exam question (ordering, naming stages, or identifying which stage an activity belongs to).

### Key Points
- The stages are drawn as **ascending steps** from "start" (bottom-left) to "End" (top-right).
- The slide title stresses: *"may involve all or some of these stages"* — not every project uses all stages.
- Communication comes **before** deployment; testing/monitoring comes **after** deployment.

### Common Mistakes
- Putting "Modelling & Analysis" before "Explore and process."
- Forgetting "Communicate results" comes **before** "Deploy & operationalize."
- Assuming all stages are always mandatory — the lecture says a project may use only **some**.
- Thinking the cycle ends at deployment — **Test and Monitoring** is the last stage.

### Diagram Explanation
- **What it represents:** a staircase diagram of 6 stages rising from *start* to *End*.
- **Why important:** repeated 4 times; the backbone of the whole course (Sprint 2 = Data Mining Workflow).
- **Labels:** Problem understanding → Explore and process → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring.
- **Possible exam questions:** "Order the stages"; "Which stage follows modelling?"; "In which stage do you clean data?"
- **Common student mistake:** swapping communicate/deploy order.

### Important Keywords
- **Problem understanding** — defining goals with stakeholders (first stage).
- **Explore and process** — data exploration, cleaning, preprocessing.
- **Modelling & Analysis** — building and analyzing models.
- **Communicate results** — storytelling/visualization of findings.
- **Deploy & operationalize** — putting the model into production.
- **Test and Monitoring** — checking performance after deployment (last stage).

### Memory Tips
Mnemonic: **"P**roblems **E**xplored, **M**odels **C**ommunicated, **D**eployed & **T**ested" → **PEM-CDT**.
Or a sentence: "**P**rofessors **E**xplain **M**odels, **C**lever **D**ata-miners **T**est."

---

## Topic 4: Domain Knowledge

### Definition
Domain knowledge is expertise about the field the data comes from (e.g., medicine, finance). It shapes how data is prepared, modeled, and interpreted.

### Simple Example
A doctor's knowledge tells the data scientist that a heart-rate of 300 bpm in the data must be a sensor error — pure statistics alone might keep it.

### Why is it Important?
The lecture dedicates a "Key takeaway" box to it — a strong exam signal.

### Key Points (exactly as taught)
Domain knowledge:
- **Guides the successful preprocessing**
- **Influences the modeling**
- **Provides the context to explain the results**

**Key takeaway (learn the spirit of this sentence):** *"Always dive into the domain behind the data — it's the key to engineering meaningful features, focusing your analysis, and telling a story your audience understands."*

### Common Mistakes
- Believing algorithms alone are enough — the lecture stresses domain context at **every** stage (preprocessing, modeling, explaining).
- Confusing domain knowledge with data knowledge — it is knowledge about the **field/application**, not the dataset format.

### Important Keywords
- **Feature engineering** — creating meaningful features, guided by domain knowledge.
- **Context** — needed to explain results to the audience.

### Memory Tips
**"P-M-E"**: Domain knowledge helps **P**reprocessing, **M**odeling, **E**xplaining — the same three letters as "**P**lease **M**ind **E**xperts."

---

## Topic 5: Data Representation ⭐

### Definition
Data representation means converting raw data (text, images, audio, graphs) into an **informative and numerical** form that machines/models can process. It is called a **fundamental step in data mining**.

### Simple Example
Words like "king" and "queen" are converted into number vectors (word embeddings) so a computer can measure that they are semantically close.

### Why is it Important?
"To train a model, we must create an informative and numerical representation of the data that can be manipulated by electronic devices." — a direct slide quote, ideal for a definition question.

### Key Points — Data formats shown
Data exist in different formats: **Social media, Imaging, EHR (Electronic Health Records), Genomics, Knowledge graphs, Bioactive molecules, Textual data, Vital signals**.

### Key Points — Representation examples (memorize all!)

| Data type | Representation |
|---|---|
| **Textual data** | **Word Embeddings** (Word2Vec, GloVe) — real-valued vectors; **TF-IDF** — numerical vectors reflecting word importance |
| **Images** | **Pixel values** |
| **Molecular structure** | **Mathematical graph** — each **atom = node**, each **bond = edge** |
| **Audio signal** | **Spectrogram** — frequency spectrum over time (speech & music analysis) |
| **Graph data** | **Node Embeddings (Node2Vec)** — continuous feature vectors for nodes capturing structural information |

### TF-IDF (concept, no formula derived in lecture)
- **TF (Term Frequency):** frequency of a word **within a document**.
- **IDF (Inverse Document Frequency):** relates to frequency of a word **across the corpus**.
- TF-IDF = TF × IDF (shown as a "×" between two pictures) → converts text into numerical vectors reflecting **importance of words in documents**.
- Intuition: a word that is frequent in one document but rare across the corpus is **important** for that document.
- Common mistake: thinking IDF rewards common words — it **down-weights** words that appear in many documents.

### Word Embeddings (Word2Vec, GloVe)
- Map words into **continuous vector spaces**.
- **Semantically similar words are positioned closely.**
- Capture **contextual relationships**; can be **pre-trained on large corpora**.

### Node Embeddings (Node2Vec)
- Learn **continuous feature representations for nodes in a graph**.
- Capture the **network's structural information**.

### Comparison

| Aspect | TF-IDF | Word Embeddings (Word2Vec/GloVe) |
|---|---|---|
| Output | Sparse numerical vectors | Dense continuous vectors |
| Captures | Word **importance** in documents | Word **meaning/context** (similar words close together) |
| Pre-training | Not needed | Can be pre-trained on large corpora |

| Aspect | Word Embeddings | Node Embeddings (Node2Vec) |
|---|---|---|
| Input | Words / text | Nodes in a graph |
| Captures | Semantic/contextual similarity | Network structural information |

### Diagram / Image Explanation
1. **Spectrogram slide:** top image = spectrogram (Frequency [Hz] vs Time [s], color = intensity); bottom = raw waveform (Amplitude vs Time). Shows how one audio signal has two representations; the spectrogram is the informative one for speech/music analysis. *Exam question:* "How can audio be represented for data mining?" *Mistake:* confusing waveform with spectrogram.
2. **Node2Vec slide:** left = a graph with colored node clusters (green/purple/blue); right = the same nodes plotted as points in a 2D vector space, with same-colored nodes near each other. Shows graph → vector space mapping preserving structure/communities.
3. **TF-IDF slide:** one document (TF) **×** many documents/corpus (IDF). Labels: "Frequency of a word within a document" and "Frequency of a word across the corpus."
4. **Word embedding scatter plot:** words plotted in 2D; related words cluster (e.g., bathroom words together, tools together, appliances together — kitchen/sink/bathtub near each other; drill/saw/bosch/dewalt near each other). Demonstrates "semantically similar words positioned closely."

### Common Mistakes
- Saying images are represented by embeddings — the lecture says **pixel values**.
- Mixing up node (atom) and edge (bond) in molecular graphs: **atom = node, bond = edge**.
- Forgetting representation must be **numerical AND informative**.

### Important Keywords
- **Word Embedding** — real-valued vector representation of words for text analysis.
- **Pixel values** — numerical representation of images.
- **Graph representation** — atoms as nodes, bonds as edges (molecules).
- **Spectrogram** — audio's frequency spectrum over time.
- **Node2Vec** — node embedding method for graphs.
- **TF-IDF** — word-importance vectors for documents.
- **Word2Vec / GloVe** — pre-trainable word embedding methods.
- **Corpus** — the whole collection of documents.

### Memory Tips
"**T**ext→Vectors, **I**mages→Pixels, **M**olecules→Graphs, **A**udio→Spectrograms, **N**etworks→Node2Vec" → **TIMAN** ("Tim can" represent anything).

---

## Topic 6: Main Task Types in Data Mining ⭐

### Definition
These are the recurring problem types you meet in the field — each pairs a goal (predict, group, detect) with a data situation (labeled or not).

### Simple Example
Spam detection = classification; Netflix suggestions = recommendation; grouping customers = clustering.

### Why is it Important?
"Identify the analysis type for this scenario" is a classic exam question format.

### Key Points — All 11 tasks (with lecture examples)

| Task | Definition (lecture wording) | Lecture Example |
|---|---|---|
| **Classification** | Sorting data into **predefined categories** | Spam detection |
| **Association Analysis** | Identifying **relationships between items** in large datasets | (market-basket style) |
| **Regression** | Predicting **continuous values** | Stock prices, recovery time for hospitalized patients |
| **Clustering** | Grouping data by **similarity without predefined labels** | Customer segmentation |
| **Optimization** | Finding **best solutions within constraints** | Route planning, resource allocation |
| **Recommendation** | Suggesting items/content based on **user preferences** | Netflix, Amazon engines |
| **Anomaly Detection** | Identifying **unusual patterns or outliers** | Fraud detection, fault prediction |
| **Personalization** | Tailoring user experiences/interactions | Personalized marketing, adaptive learning systems |
| **Decision Support** | Informing decision-making by analyzing large datasets & providing insights | Medical diagnostics, business intelligence |
| **Simulation** | Modeling complex systems to **predict behavior and outcomes** | Weather modeling, financial market |
| **Text Mining** | Understanding, mining, and **generating human language** | Sentiment analysis, language translation |

### Common Mistakes
- **Classification vs Clustering:** classification has **predefined categories (labels)**; clustering has **no labels**.
- **Classification vs Regression:** classification predicts **categories**; regression predicts **continuous values**.
- Calling fraud detection "classification" when the lecture lists it under **anomaly detection**.
- Confusing **Recommendation** (suggesting items) with **Personalization** (tailoring the whole experience).

### Comparison

| Aspect | Classification | Clustering |
|---|---|---|
| Labels | Predefined categories | No predefined labels |
| Learning style | Supervised | Unsupervised |
| Example | Spam detection | Customer segmentation |

| Aspect | Classification | Regression |
|---|---|---|
| Output | Category / class | Continuous value |
| Example | Spam vs not-spam | Stock price, recovery time |

| Aspect | Anomaly Detection | Classification |
|---|---|---|
| Goal | Find unusual/outlier patterns | Assign to known categories |
| Example | Fraud detection, fault prediction | Spam detection |

### Important Keywords
- **Predefined categories** — the trigger word for classification.
- **Continuous values** — the trigger word for regression.
- **Similarity without labels** — the trigger phrase for clustering.
- **Outliers** — the trigger word for anomaly detection.
- **Constraints** — the trigger word for optimization.

### Memory Tips
"**CARCO RAPDST**" is hard — instead use trigger words: *categories→Classification, continuous→Regression, similarity/no labels→Clustering, items-relations→Association, outliers→Anomaly, constraints→Optimization, preferences→Recommendation, tailoring→Personalization, insights-for-decisions→Decision Support, model-systems→Simulation, language→Text Mining.*

---

## Topic 7: Storytelling with Data

### Definition
Storytelling is the skill of presenting data mining results so the audience understands what happened, why, and what to do next.

### Simple Example
Instead of dumping a spreadsheet, an analyst shows one clear chart ("churn rose 15% after the price change") and a narrative recommending a fix.

### Why is it Important?
It is called a **crucial skill** when presenting and communicating data mining results — and maps to the "Communicate results" life-cycle stage.

### Key Points — The 3 Key Components
1. **Data** — the evidence or facts
2. **Visualization** — charts, graphs, dashboards, etc.
3. **Narrative** — the storyline guiding interpretation (**what happened, why, what now?**)

- Reference cited: *"Storytelling with Data," Cole Nussbaumer Knaflic*.
- The slide shows **common types of visualizations used in business presentations** (e.g., a big number "67%", charts).

### Common Mistakes
- Thinking storytelling = only visualization; it needs all **three** components.
- Forgetting the narrative answers three questions: **what happened, why, what now**.

### Important Keywords
- **Narrative** — the storyline of interpretation.
- **Dashboard** — a visualization form for business presentations.

### Memory Tips
**D-V-N** = "**D**on't **V**omit **N**umbers" — use Data, Visualization, Narrative instead.

---

## Topic 8: Types of Data (by measurement) ⭐

### Definition
Data attributes are split into **Qualitative (Categorical)** and **Quantitative (Numerical)**; each splits again into two subtypes.

### The Tree (memorize!)

```
Types of Data
├── Qualitative (Categorical)
│   ├── Nominal  → Gender, Hair color, Ethnicity
│   └── Ordinal  → Reviews, Grades, Status
└── Quantitative (Numerical)
    ├── Discrete    → No. of students, Population, No. of houses
    └── Continuous  → Height of children, Speed of cars, Weight
```

### Simple Example
"Blood type A/B/O" = nominal; "Hotel rating 1–5 stars" = ordinal; "Number of visits" = discrete; "Body temperature" = continuous.

### Why is it Important?
Choosing preprocessing, visualization, and models depends on the data type — and "classify this attribute" is a favorite exam question.

### Key Points
- **Nominal:** categories with **no order** (gender, hair color, ethnicity).
- **Ordinal:** categories **with a meaningful order** (reviews, grades, status).
- **Discrete:** countable numbers (students, population, houses).
- **Continuous:** measurable values on a continuum (height, speed, weight).

### Common Mistakes
- Treating **grades** as numeric — in the lecture they are **ordinal (qualitative)**.
- Calling **population** continuous — it is **discrete** (countable).
- Confusing nominal vs ordinal: only ordinal has an **order**.

### Comparison

| Aspect | Qualitative (Categorical) | Quantitative (Numerical) |
|---|---|---|
| Nature | Describes categories | Describes amounts/numbers |
| Subtypes | Nominal, Ordinal | Discrete, Continuous |
| Examples | Gender, Grades | Population, Weight |

| Aspect | Nominal | Ordinal |
|---|---|---|
| Order? | No | Yes |
| Examples | Gender, hair color, ethnicity | Reviews, grades, status |

| Aspect | Discrete | Continuous |
|---|---|---|
| Values | Countable integers | Any value in a range |
| Examples | No. of students, population, no. of houses | Height, speed, weight |

### Important Keywords
- **Nominal** — unordered categories.
- **Ordinal** — ordered categories.
- **Discrete** — countable numeric.
- **Continuous** — measurable numeric.

### Memory Tips
"**NO**minal = **NO** order; **ORD**inal = **ORD**er." "Dis**crete** = dis**tinct counts**; Continuous = can take **any** value (like a **continuous** line)."

---

## Topic 9: Types of Data (by structure) ⭐

### Definition
Data is also classified by how it is organized: **Structured**, **Unstructured**, and **Semi-Structured**.

### Key Points (lecture wording)

| Type | Definition | Examples |
|---|---|---|
| **Structured** | Organized data adhering to a **predefined schema**, typically stored in **relational databases** | Customer information tables, transaction records |
| **Unstructured** | Data **without a predefined format** — more challenging to collect, process, and analyze | Emails, videos, social media posts |
| **Semi-Structured** | Not in a relational database but has **some organizational properties** such as **tags or markers** | JSON files, XML documents, graph data |

### Simple Example
Your bank statement table = structured; your Instagram video = unstructured; a JSON API response = semi-structured.

### Why is it Important?
Determines storage, tooling, and how much preprocessing is needed; a classic 3-way matching exam question.

### Common Mistakes
- Calling **emails structured** because they have fields — the lecture lists emails as **unstructured**.
- Calling **JSON/XML unstructured** — they are **semi-structured** (they have tags/markers).
- Forgetting **graph data** is listed as semi-structured.

### Comparison

| Aspect | Structured | Semi-Structured | Unstructured |
|---|---|---|---|
| Schema | Predefined | Partial (tags/markers) | None |
| Storage | Relational DB | Non-relational (JSON/XML) | Files, media |
| Difficulty | Easiest | Medium | Hardest |
| Examples | Tables, transactions | JSON, XML, graphs | Emails, videos, posts |

### Important Keywords
- **Schema** — the predefined structure of data.
- **Relational database** — storage for structured data.
- **Tags/markers** — the organizational hints in semi-structured data.

### Memory Tips
"**S**tructured = **S**chema; **S**emi = **S**ome tags; **U**nstructured = **U**npredictable." Examples: "**Table–Tag–Tape**" (table=structured, tagged JSON=semi, video tape=unstructured).

---

## Topic 10: Tools & Development Frameworks

### Definition
The ecosystem of (mostly Python-based) libraries used to analyze and mine data.

### Key Points (the framework table — Python-based)

| Category | Python-Based Tools |
|---|---|
| Other essential packages | **NumPy, Pandas, Matplotlib, Seaborn** |
| Machine Learning | **Scikit-learn, XGBoost, RAPIDS** |
| Big Data Analytics | **Apache Spark (PySpark)** |
| Large Language Models | **Hugging Face Transformers, OpenAI GPT** |
| Deep Learning | **TensorFlow, PyTorch** |
| Hyperparameter Optimization | **Optuna, Hyperopt, GridSearchCV** |

- **Non-Python alternatives:** MATLAB, RapidMiner, R.
- The module itself uses **Python** and common statistical Python frameworks.

### Simple Example
You clean data with Pandas, visualize with Matplotlib/Seaborn, train with Scikit-learn, and tune hyperparameters with Optuna.

### Why is it Important?
Exam may ask "which library belongs to which category" (theory only — no coding).

### Common Mistakes
- Placing Scikit-learn under deep learning — it's **machine learning**; TensorFlow/PyTorch are **deep learning**.
- Forgetting PySpark = **Big Data Analytics**.
- Forgetting GridSearchCV is listed under **hyperparameter optimization**.

### Important Keywords
- **Pandas/NumPy** — data handling & numerics.
- **Matplotlib/Seaborn** — visualization.
- **Scikit-learn** — classic machine learning.
- **PySpark** — big data analytics.
- **TensorFlow/PyTorch** — deep learning.
- **Optuna/Hyperopt/GridSearchCV** — hyperparameter optimization.

### Memory Tips
"**Pa**ndas prepares, **S**cikit learns, **S**park scales, **T**orch deepens, **O**ptuna optimizes."

---

## Topic 11: Course Logistics, Prerequisites, Career & Outline

### Definition
Administrative and orientation content: how to pass the module, what to pre-read, what careers exist, and the 12-sprint plan.

### Key Points — Classroom Agreements & Logistics
- **Interactive teaching** — solve small tasks / answer questions in class (uses **Slido** app).
- Blocks start at **10:00** — arrive on time; mute smartphones.
- Group assignment by end of week; **switching groups is not allowed**; priority to those who haven't cleared the prerequisite.
- **80% active participation in exercises is mandatory for exam eligibility.**
- Module uses **Python**, conducted in **English**; attending all blocks strongly recommended.

### Key Points — Pre-read (prerequisites)
- **Mathematical notations:** Vectors & Matrices (matrix multiplication, vector norms, **eigenvalues**); Derivatives & Gradients (gradient vectors, **Jacobian matrix**); Exponential & Logarithmic functions; Basic Statistics (**Bayes' Theorem**, mean, variance, probability distribution); **Covariance Matrix**; **Dimensionality Reduction with PCA**.
- **Key ML terminology:** Model, Backpropagation, preprocessing, Feature, Target, training, performance evaluation, **Overfitting/Underfitting**.
- Python introduction if unfamiliar; practice on **Kaggle**.

### Key Points — Successful AI Career (3-step slide)
1. **Fundamentals:** computer science, math, and programming subjects.
2. **Build a solid foundation in one or more fields:** Machine and Deep Learning, NLP, Cloud Computing, Data Engineering, Machine Learning DevOps, Ethical and responsible AI use, Data Analysis and visualization, Storytelling skills.
3. **Deepen knowledge** in an area you are passionate about, follow emerging technologies, watch job-market trends.

### Key Points — Career Options
AI Engineer, Data Analyst, Data Scientist, Robotics and Automation expert, Machine Learning DevOps (ML Engineer), Data Engineer, Software Engineer — "Whatever!"

### Key Points — The 12 Sprints (course outline)
1. Introduction to Data Mining
2. Data Mining Workflow
3. Processing and Dimensionality Reduction
4. Prototyping and Deployment
5. Classification Analysis
6. Clustering Analysis
7. Regression Analysis
8. Neural Networks (math to practice)
9. GenAI for Data Mining: Retrieval Augmented Generation (RAG)
10. Explainability Analysis
11. Ethical and Responsible Data Mining
12. Wrap-up & discussion

### Common Mistakes
- Forgetting the **80% participation** eligibility rule.
- Thinking group switching is possible — it is explicitly **not allowed**.

### Memory Tips
Sprints: "**I**ntro, **W**orkflow, **P**rocessing, **P**rototyping, **C**lassify, **C**luster, **R**egress, **N**eural, **G**enAI, **E**xplain, **E**thics, **W**rap-up" → "I Will Practice Patiently; Clever Coders Rarely Need Guessing; Everything Ends Well."

---

# 3. Topic-wise Exam Practice

## Data Mining Definition

1. **MCQ:** Data mining extracts knowledge from: a) only structured data b) only unstructured data c) **both structured and unstructured data** ✔ d) only databases
2. **True/False:** Data mining is a single-discipline field. → **False** (interdisciplinary)
3. **Fill in the blank:** Data mining uses scientific methods, processes, ______ and systems. → **algorithms**
4. **Definition question:** Define data mining. → Interdisciplinary field using scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data across many domains.
5. **One-word:** What adjective describes data mining as combining multiple fields? → **Interdisciplinary**

## Applications

6. **Matching:** Match: Fraud detection / Churn prediction / Spam filtering / Drug discovery → Finance & Banking / Business & Marketing / Social Media & Internet / Healthcare & Medicine
7. **MCQ:** Predictive maintenance belongs to: a) Healthcare b) **Manufacturing & Industry** ✔ c) Finance d) Smart Cities
8. **Scenario:** A telecom wants to know which customers will cancel next month. Which application? → **Churn prediction** (Business & Marketing)
9. **Two-word answer:** Netflix's engine suggesting films is a ______ ______. → **Recommendation system**
10. **True/False:** Intrusion detection is a cybersecurity application. → **True**

## Project Life Cycle

11. **Pipeline ordering:** Order: Deploy & operationalize / Problem understanding / Communicate results / Explore and process / Test and Monitoring / Modelling & Analysis → **Problem understanding → Explore and process → Modelling & Analysis → Communicate results → Deploy & operationalize → Test and Monitoring**
12. **MCQ:** Which stage comes immediately after "Modelling & Analysis"? a) Deploy b) **Communicate results** ✔ c) Testing d) Explore
13. **True/False:** Every data mining project must include all life-cycle stages. → **False** ("may involve all or **some**")
14. **Short answer:** What is the final stage of the life cycle? → **Test and Monitoring**
15. **Scenario:** You are cleaning missing values and looking at distributions. Which stage? → **Explore and process**

## Domain Knowledge

16. **Short answer:** Name the three ways domain knowledge helps a project. → Guides preprocessing, influences modeling, provides context to explain results.
17. **True/False:** Domain knowledge is only needed when interpreting results. → **False** (also preprocessing and modeling)
18. **Fill in the blank:** "Always dive into the ______ behind the data." → **domain**

## Data Representation

19. **MCQ:** Images are typically represented by: a) embeddings b) **pixel values** ✔ c) graphs d) spectrograms
20. **Two-word answer:** Audio can be represented as its frequency spectrum over time, called a ______. → **Spectrogram** (one word) — used for **speech and music** analysis (two words).
21. **Matching:** Text→? Molecules→? Graph nodes→? → Word embeddings/TF-IDF; mathematical graph (atom=node, bond=edge); Node2Vec.
22. **True/False:** In a molecular graph, each bond is a node. → **False** (atom = node, bond = **edge**)
23. **Formula interpretation:** In TF-IDF, what does IDF capture? → How rare/frequent a word is **across the corpus** (down-weights common words).
24. **MCQ:** Word2Vec places semantically similar words: a) far apart b) **close together** ✔ c) randomly d) alphabetically
25. **Short answer:** Why must data be represented numerically? → So it can be manipulated by electronic devices / used to train models.
26. **One-word:** Which method learns continuous representations for graph nodes? → **Node2Vec**

## Task Types

27. **Analysis type identification:** Predicting a patient's recovery time (in days) → **Regression**
28. **Analysis type identification:** Grouping customers with no labels → **Clustering**
29. **Analysis type identification:** Flagging an unusual credit-card transaction → **Anomaly Detection**
30. **Analysis type identification:** Finding items frequently bought together → **Association Analysis**
31. **MCQ:** Which task uses predefined categories? a) Clustering b) **Classification** ✔ c) Association d) Simulation
32. **True/False:** Regression predicts categories. → **False** (continuous values)
33. **Scenario:** Planning delivery routes under constraints → **Optimization**
34. **Two-word answer:** Sentiment analysis is an example of ______ ______. → **Text mining**

## Storytelling

35. **Short answer:** Name the three key components of data storytelling. → **Data, Visualization, Narrative**
36. **Fill in the blank:** The narrative answers: what happened, ______, what now? → **why**
37. **Visualization selection:** For a business presentation of one key metric change, what would you use? → A clear chart / big-number visual (common business visualizations) supported by a narrative.

## Data Types

38. **MCQ:** Hair color is: a) ordinal b) **nominal** ✔ c) discrete d) continuous
39. **MCQ:** Grades are: a) nominal b) **ordinal** ✔ c) continuous d) discrete
40. **MCQ:** Number of houses is: a) continuous b) **discrete** ✔ c) ordinal d) nominal
41. **MCQ:** Speed of cars is: a) discrete b) **continuous** ✔ c) nominal d) ordinal
42. **Matching:** JSON files / Videos / Transaction records → Semi-structured / Unstructured / Structured
43. **True/False:** Emails are structured data. → **False** (unstructured)
44. **Fill in the blank:** Semi-structured data has organizational properties such as ______ or markers. → **tags**
45. **Definition question:** Define structured data. → Data organized under a predefined schema, typically in relational databases.

## Tools

46. **Matching:** Scikit-learn / PySpark / PyTorch / Optuna → Machine Learning / Big Data Analytics / Deep Learning / Hyperparameter Optimization
47. **One-word:** A non-Python data mining alternative starting with "R"? → **R** (also RapidMiner)
48. **True/False:** TensorFlow is listed under Machine Learning. → **False** (Deep Learning)

## Logistics (possible checkpoint questions)

49. **Fill in the blank:** ______% active participation in exercises is mandatory for exam eligibility. → **80**
50. **True/False:** Group switching is allowed on request. → **False**

---

# 4. Flashcards

- **What is data mining?** → Interdisciplinary field using scientific methods, processes, algorithms & systems to extract knowledge/insights from structured & unstructured data.
- **Is data mining interdisciplinary?** → Yes — combines statistics, CS, ML, domain knowledge.
- **6 life-cycle stages in order?** → Problem understanding → Explore & process → Modelling & Analysis → Communicate results → Deploy & operationalize → Test & Monitoring.
- **Must all stages be used?** → No — "all or some."
- **3 roles of domain knowledge?** → Guides preprocessing, influences modeling, provides context for results.
- **Key takeaway about domain?** → Dive into the domain — key to meaningful features, focused analysis, understandable story.
- **How are images represented?** → Pixel values.
- **How is text represented?** → Word embeddings (real-valued vectors) or TF-IDF.
- **How are molecules represented?** → Graph: atom = node, bond = edge.
- **What is a spectrogram?** → Audio signal's frequency spectrum over time (speech/music analysis).
- **What is Node2Vec?** → Learns continuous feature representations for graph nodes, capturing structural information.
- **TF?** → Frequency of a word within a document.
- **IDF?** → Frequency of a word across the corpus (inverse weighting).
- **What does TF-IDF produce?** → Numerical vectors reflecting word importance in documents.
- **Word2Vec/GloVe property?** → Semantically similar words are close in vector space; pre-trainable on large corpora.
- **Classification?** → Sorting data into predefined categories (spam detection).
- **Regression?** → Predicting continuous values (stock prices, recovery time).
- **Clustering?** → Grouping by similarity without predefined labels (customer segmentation).
- **Association analysis?** → Identifying relationships between items in large datasets.
- **Anomaly detection?** → Identifying unusual patterns/outliers (fraud detection, fault prediction).
- **Optimization?** → Best solutions within constraints (route planning).
- **Recommendation?** → Suggesting items based on user preferences (Netflix/Amazon).
- **Personalization?** → Tailoring user experiences (personalized marketing).
- **Decision support?** → Insights for decision-making (medical diagnostics, BI).
- **Simulation?** → Modeling complex systems to predict outcomes (weather, financial market).
- **Text mining?** → Understanding/mining/generating human language (sentiment analysis, translation).
- **3 storytelling components?** → Data (evidence), Visualization (charts), Narrative (what happened, why, what now).
- **Storytelling book referenced?** → "Storytelling with Data" — Cole Nussbaumer Knaflic.
- **Nominal data?** → Unordered categories (gender, hair color, ethnicity).
- **Ordinal data?** → Ordered categories (reviews, grades, status).
- **Discrete data?** → Countable numbers (students, population, houses).
- **Continuous data?** → Measurable values (height, speed, weight).
- **Structured data?** → Predefined schema, relational databases (tables, transactions).
- **Unstructured data?** → No predefined format (emails, videos, social posts).
- **Semi-structured data?** → Tags/markers, no relational DB (JSON, XML, graph data).
- **ML library?** → Scikit-learn (also XGBoost, RAPIDS).
- **Deep learning libraries?** → TensorFlow, PyTorch.
- **Big data tool?** → Apache Spark (PySpark).
- **Hyperparameter optimization tools?** → Optuna, Hyperopt, GridSearchCV.
- **Non-Python alternatives?** → MATLAB, RapidMiner, R.
- **Exam eligibility rule?** → 80% active participation in exercises.
- **Career step 1?** → Fundamentals: CS, math, programming.

---

# 5. Rapid Revision

- **Data mining** = interdisciplinary knowledge extraction from **structured + unstructured** data.
- **Life cycle (4× repeated!):** Problem understanding → Explore & process → Modelling & Analysis → Communicate results → Deploy & operationalize → Test & Monitoring. May use **all or some** stages.
- **Domain knowledge:** guides **preprocessing**, influences **modeling**, explains **results**.
- **Representation must be numerical + informative:** text→embeddings/TF-IDF, images→pixels, molecules→graphs (atom=node, bond=edge), audio→spectrograms, graph nodes→Node2Vec.
- **TF** = word frequency in a document; **IDF** = across the corpus; product = importance.
- **Task triggers:** categories→classification; continuous→regression; no labels→clustering; item relations→association; outliers→anomaly; constraints→optimization; preferences→recommendation; language→text mining.
- **Storytelling = Data + Visualization + Narrative** (what happened, why, what now).
- **Measurement types:** Qualitative (Nominal-no order / Ordinal-order) vs Quantitative (Discrete-count / Continuous-measure).
- **Structure types:** Structured (schema, relational DB) / Semi-structured (tags: JSON, XML, graphs) / Unstructured (emails, videos, posts).
- **Tools:** Pandas/NumPy/Matplotlib/Seaborn; Scikit-learn/XGBoost/RAPIDS; PySpark; HF Transformers/GPT; TensorFlow/PyTorch; Optuna/Hyperopt/GridSearchCV. Non-Python: MATLAB, RapidMiner, R.
- **Logistics:** 80% exercise participation for exam eligibility; no group switching; English; Python.

---

# 6. High Probability Exam Questions

★★★★★ **Very High**
1. Order the stages of the data mining project life cycle. (Shown 4×!)
2. Define data mining (interdisciplinary, structured + unstructured).
3. Classification vs Clustering vs Regression — define and give examples.
4. Classify attributes as nominal/ordinal/discrete/continuous (gender, grades, population, weight…).
5. Structured vs Unstructured vs Semi-structured — definitions + examples (JSON, emails, tables).

★★★★ **High**
6. Three roles of domain knowledge in a data mining project.
7. How are text / images / molecules / audio / graph data represented? (embeddings, pixels, atom-node/bond-edge graphs, spectrograms, Node2Vec)
8. What do TF and IDF stand for and what does TF-IDF reflect?
9. Three components of data storytelling (Data, Visualization, Narrative — what/why/what now).
10. Match applications to domains (fraud→finance, churn→marketing, spam→social media, predictive maintenance→manufacturing).
11. Identify the analysis type for a scenario (fraud=anomaly, segmentation=clustering, recovery-time=regression).

★★★ **Medium**
12. Property of Word2Vec/GloVe embeddings (similar words close; pre-trainable).
13. What does Node2Vec capture? (structural information of a network)
14. Match libraries to categories (Scikit-learn→ML, PyTorch→DL, PySpark→Big Data, Optuna→hyperparameter optimization).
15. Why must data representation be numerical? (so electronic devices/models can process it)
16. "May involve all or some stages" — true/false trap.
17. Name non-Python data mining alternatives (MATLAB, RapidMiner, R).

---

# 7. Lecture Cheat Sheet

## Definitions
- **Data mining:** interdisciplinary field extracting knowledge/insights from structured & unstructured data via scientific methods, processes, algorithms, systems.
- **Data representation:** informative **numerical** encoding of data for model training.
- **Storytelling:** Data + Visualization + Narrative.

## The Pipeline
`Problem understanding → Explore & process → Modelling & Analysis → Communicate results → Deploy & operationalize → Test & Monitoring` *(all or some)*

## Task Trigger Words
| Trigger | Task |
|---|---|
| predefined categories | Classification |
| continuous values | Regression |
| similarity, no labels | Clustering |
| item relationships | Association Analysis |
| outliers/unusual | Anomaly Detection |
| constraints | Optimization |
| user preferences | Recommendation |
| human language | Text Mining |

## Representations
| Data | Representation |
|---|---|
| Text | Word embeddings, TF-IDF |
| Images | Pixel values |
| Molecules | Graph (atom=node, bond=edge) |
| Audio | Spectrogram |
| Graph nodes | Node2Vec |

## Data Types
- **Qualitative:** Nominal (no order: gender) | Ordinal (order: grades)
- **Quantitative:** Discrete (count: population) | Continuous (measure: weight)
- **Structure:** Structured (schema/tables) | Semi (JSON/XML/graphs — tags) | Unstructured (emails/videos)

## TF-IDF
TF = word freq **within** a document; IDF = based on freq **across** the corpus; TF×IDF = word importance.

## Tools
Pandas/NumPy (data) • Matplotlib/Seaborn (viz) • Scikit-learn/XGBoost/RAPIDS (ML) • PySpark (big data) • TensorFlow/PyTorch (DL) • HF Transformers/GPT (LLM) • Optuna/Hyperopt/GridSearchCV (HPO) • MATLAB/RapidMiner/R (non-Python)

## Frequently Confused
- Classification (labels) ≠ Clustering (no labels)
- Classification (categories) ≠ Regression (continuous)
- Nominal (no order) ≠ Ordinal (order)
- Discrete (count) ≠ Continuous (measure)
- Semi-structured (JSON/XML) ≠ Unstructured (emails/video)
- Atom = **node**, Bond = **edge**
- Fraud detection = **Finance** domain, but **anomaly detection** task type

---

# 8. Lecture Summary

## Top 50 Most Probable Exam Questions (condensed)
1. Define data mining. 2. Why is it interdisciplinary? 3. Order the 6 life-cycle stages. 4. Which stage follows modelling? 5. Which is the last stage? 6. Must all stages be used? 7. Three roles of domain knowledge. 8. State the key takeaway about domain knowledge. 9. Why must representations be numerical? 10. How is text represented? 11. How are images represented? 12. How are molecules represented? 13. What is a node/edge in a molecular graph? 14. What is a spectrogram? 15. What is Node2Vec? 16. What is TF? 17. What is IDF? 18. What does TF-IDF reflect? 19. Property of Word2Vec/GloVe. 20. Define classification + example. 21. Define regression + example. 22. Define clustering + example. 23. Define association analysis. 24. Define anomaly detection + example. 25. Define optimization + example. 26. Define recommendation + example. 27. Define personalization. 28. Define decision support. 29. Define simulation. 30. Define text mining + example. 31. Classification vs clustering. 32. Classification vs regression. 33. Three storytelling components. 34. What does the narrative answer? 35. Nominal vs ordinal. 36. Discrete vs continuous. 37. Classify: gender, grades, population, weight. 38. Structured vs unstructured vs semi-structured. 39. Examples of semi-structured data. 40. Are emails structured? 41. Which domain: fraud detection? 42. Which domain: churn prediction? 43. Which domain: predictive maintenance? 44. Which domain: spam filtering? 45. Scikit-learn category? 46. TensorFlow/PyTorch category? 47. PySpark category? 48. Optuna category? 49. Non-Python alternatives? 50. Scenario→task-type identification.

## Top 30 Definitions
Data mining • Interdisciplinary • Problem understanding • Explore and process • Modelling & Analysis • Communicate results • Deploy & operationalize • Test and Monitoring • Domain knowledge • Data representation • Word embedding • Pixel values • Molecular graph • Spectrogram • Node embedding (Node2Vec) • TF • IDF • TF-IDF • Word2Vec/GloVe • Classification • Regression • Clustering • Association analysis • Anomaly detection • Optimization • Recommendation • Storytelling (Data/Visualization/Narrative) • Nominal/Ordinal • Discrete/Continuous • Structured/Semi-structured/Unstructured

## Top 30 Keywords
interdisciplinary • knowledge extraction • structured data • unstructured data • semi-structured • schema • relational database • tags/markers • life cycle • preprocessing • modeling • deployment • monitoring • domain knowledge • feature engineering • numerical representation • embedding • pixel • node • edge • corpus • spectrogram • predefined categories • continuous values • similarity • outliers • constraints • narrative • nominal • ordinal

## Top 20 Comparisons
1. Classification vs Clustering 2. Classification vs Regression 3. Clustering vs Anomaly Detection 4. Nominal vs Ordinal 5. Discrete vs Continuous 6. Qualitative vs Quantitative 7. Structured vs Unstructured 8. Structured vs Semi-structured 9. Semi-structured vs Unstructured 10. TF vs IDF 11. TF-IDF vs Word Embeddings 12. Word Embeddings vs Node Embeddings 13. Waveform vs Spectrogram 14. Data Mining vs Data Science 15. Data Mining vs Big Data Analytics 16. Recommendation vs Personalization 17. ML libraries vs DL libraries 18. Python vs non-Python tools 19. Communicate results vs Deploy (order!) 20. Fraud detection (domain: finance) vs (task: anomaly detection)

## Top 20 Formulas / Formula-like Concepts
(This intro lecture is light on formulas; these are the quantitative concepts named:)
1. TF-IDF = TF × IDF (conceptual product) 2. TF = word frequency within a document 3. IDF = inverse of frequency across corpus 4. Vector representation of words (real-valued vectors) 5. Pixel value matrices for images 6. Graph G = nodes (atoms) + edges (bonds) 7. Frequency spectrum over time (spectrogram) 8–14. Pre-read math: matrix multiplication, vector norms, eigenvalues, derivatives/gradients, Jacobian matrix, Bayes' theorem, mean & variance 15. Probability distribution 16. Covariance matrix 17. PCA (dimensionality reduction) 18. Exponential functions 19. Logarithmic functions 20. Vector spaces for embeddings

## Top 20 Diagrams / Images
1. Life-cycle staircase (×4 — most important!) 2. Data-formats collage (social media, imaging, EHR, genomics, knowledge graphs, molecules, text, vital signals) 3. Spectrogram vs waveform 4. Graph → 2D embedding (Node2Vec) 5. TF × IDF document pictures 6. Word-embedding scatter plot (semantic clusters) 7. Data-types tree (qualitative/quantitative) 8. Structured/unstructured/semi-structured 3-column table 9. Applications 9-domain grid 10. Storytelling visualization examples ("67%") 11. Domain-knowledge takeaway box 12. Interdisciplinary Venn-style slide 13. Data Science & Big Data example slide 14. Framework table 15. AI-career 3-step diagram (fundamentals → foundation fields → deepen) 16. 12-sprint course outline 17. Career options list 18. Pre-read topics slide 19. Classroom agreements slide 20. Questions/Answers closing slide

## 10-Minute Revision Sheet
1. Data mining = interdisciplinary extraction of knowledge from structured + unstructured data.
2. Pipeline: **P**roblem → **E**xplore → **M**odel → **C**ommunicate → **D**eploy → **T**est/Monitor ("all or some").
3. Domain knowledge: preprocessing + modeling + explaining.
4. Representation: numerical & informative — text→vectors, image→pixels, molecule→graph, audio→spectrogram, node→Node2Vec.
5. TF-IDF: in-document freq × across-corpus rarity = importance.
6. 11 task types; know trigger words (categories/continuous/no-labels/outliers/constraints).
7. Storytelling = Data + Visualization + Narrative (what, why, what now).
8. Data types: Nominal/Ordinal | Discrete/Continuous; Structured/Semi/Unstructured.
9. Tools: sklearn=ML, TF/PyTorch=DL, PySpark=Big Data, Optuna=HPO.
10. 80% participation → exam eligibility.

---

# Examples Repository

| # | Example | Topic | Original Lecture Context | Simple Explanation | Why Important | Possible Exam Questions |
|---|---|---|---|---|---|---|
| 1 | **Spam detection** | Classification | Listed as example of sorting into predefined categories | Emails sorted into spam/not-spam | Canonical classification example | "Give an example of classification." |
| 2 | **Stock prices; recovery time for hospitalized patients** | Regression | Examples of predicting continuous values | Output is a number, not a class | Shows regression targets are continuous | "Is predicting recovery time classification or regression?" |
| 3 | **Customer segmentation** | Clustering | Example of grouping without predefined labels | Similar customers grouped automatically | Canonical clustering example | "Which task groups customers without labels?" |
| 4 | **Route planning, resource allocation** | Optimization | Best solutions within constraints | Finding shortest/cheapest option | Distinguishes optimization from prediction | "Delivery routing is which task type?" |
| 5 | **Netflix / Amazon engines** | Recommendation | Suggesting items based on preferences | Suggest what you'll likely enjoy | Real-world household example | "Name a real recommendation system." |
| 6 | **Fraud detection, fault prediction** | Anomaly Detection | Identifying unusual patterns/outliers | Rare, suspicious events flagged | Fraud is anomaly detection, not classification (here) | "Fraud detection is an example of ___?" |
| 7 | **Personalized marketing, adaptive learning systems** | Personalization | Tailoring user experiences | Content adapts to the individual | Distinguish from recommendation | "Give an example of personalization." |
| 8 | **Medical diagnostics, business intelligence** | Decision Support | Insights for decision-making | Data informs human decisions | Shows humans stay in the loop | "BI dashboards support which task type?" |
| 9 | **Weather modeling, financial market** | Simulation | Modeling complex systems to predict outcomes | Simulate to forecast behavior | Simulation ≠ prediction from historical labels only | "Weather modeling is an example of ___?" |
| 10 | **Sentiment analysis, language translation** | Text Mining | Understanding/mining/generating language | Machines process human language | Links to NLP | "Two examples of text mining?" |
| 11 | **Word embedding for text** | Data Representation | "Representation of words … typically in real valued vectors" | Words become number vectors | Core representation concept | "How is text represented numerically?" |
| 12 | **Pixel values for images** | Data Representation | Listed under representation examples | Each pixel is a number | Simplest image representation | "How are images represented?" |
| 13 | **Molecule as a graph (atom=node, bond=edge)** | Data Representation | Molecular structure representation | Chemistry becomes a network | Tests node/edge understanding | "In a molecular graph, what is an edge?" |
| 14 | **Spectrogram of an audio signal** | Data Representation | Frequency spectrum over time; speech & music analysis | Sound becomes an image-like map | Non-obvious representation | "How can audio be represented?" |
| 15 | **Node2Vec on a network** | Data Representation | Continuous node features capturing structure | Graph nodes → points in vector space | Graph ML foundation | "What does Node2Vec learn?" |
| 16 | **TF-IDF document pictures** | Data Representation | TF (within doc) × IDF (across corpus) | Important words score high | Only quasi-formula of the lecture | "What do TF and IDF measure?" |
| 17 | **Word2Vec scatter (kitchen/tools/appliances clusters)** | Data Representation | Semantically similar words close together | Related words cluster in 2D plot | Visual proof of embeddings | "What property does the word-embedding plot show?" |
| 18 | **Application domains grid** (predictive maintenance, dynamic pricing, fake news detection, genomics, astronomy, algorithmic trading, drug discovery, traffic prediction, …) | Applications | 9-domain slide | Data mining is everywhere | Matching-question goldmine | "Match application to domain." |
| 19 | **Gender, hair color, ethnicity** | Data types | Nominal examples | Categories without order | Classify-the-attribute questions | "Hair color is which data type?" |
| 20 | **Reviews, grades, status** | Data types | Ordinal examples | Ordered categories | Grades ≠ numeric here | "Grades are which data type?" |
| 21 | **No. of students, population, no. of houses** | Data types | Discrete examples | Countable | Population is discrete! | "Population: discrete or continuous?" |
| 22 | **Height of children, speed of cars, weight** | Data types | Continuous examples | Measurable | Standard continuous examples | "Weight is which data type?" |
| 23 | **Customer tables, transaction records** | Structured data | Structured examples | Fit relational schema | Structure classification | "Give examples of structured data." |
| 24 | **Emails, videos, social media posts** | Unstructured data | Unstructured examples | No predefined format | Emails trap-question | "Are emails structured?" |
| 25 | **JSON files, XML documents, graph data** | Semi-structured data | Semi-structured examples | Tags/markers but no relational DB | JSON/XML classic exam item | "JSON is which structure type?" |
| 26 | **"67%" business chart** | Storytelling | Common business visualization types | One number can carry the story | Illustrates visualization component | "Name the 3 storytelling components." |
| 27 | **Kaggle** | Logistics/Practice | Recommended platform for hands-on challenges | Practice arena for DS/AI | Shows expected self-study | "Which platform was recommended for practice?" |
| 28 | **Data formats collage (Social media, Imaging, EHR, Genomics, Knowledge graphs, Bioactive molecules, Textual data, Vital signals)** | Data Representation | "Data exist in different formats" | Many raw formats → all need numerical representation | Motivates the representation step | "Why is representation a fundamental step?" |

---

# Quiz Repository

The lecture contained interactive **Slido** polls and orientation questions rather than graded quizzes. All are collected here.

| # | Original Question | Correct / Inferred Answer | Explanation | Related Topic | Difficulty |
|---|---|---|---|---|---|
| 1 | *(Slido poll)* "How would you rate your knowledge in topics like AI, Data Science and Data Mining?" | No fixed answer — opinion poll | Icebreaker to gauge class level; **Inferred Answer:** self-assessment on a scale | Course intro | Easy |
| 2 | *(Slido poll)* "What skills or knowledge are you hoping to develop through this data mining course?" | Open-ended | **Inferred Answer:** skills from the course outline — data mining workflow, preprocessing, classification, clustering, regression, neural networks, GenAI/RAG, explainability, ethics | Course outline | Easy |
| 3 | *(Discussion)* "Who is your professor? What should you expect from me? What do I appreciate?" | Prof. Dr. Matteo Marouf; expects interactive participation | **Inferred Answer:** interactive teaching, punctuality (10:00), silence unless questions, no excuse emails | Classroom agreement | Easy |
| 4 | *(Discussion)* "Establishing Classroom agreement: How should we behave, communicate and deal with each other throughout this course?" | The classroom agreements list | Interactive teaching, stay silent unless questions, mute phones, arrive by 10:00, no group switching, avoid excuse emails | Classroom agreement | Easy |
| 5 | *(Section question)* "How do you pass this module successfully?" | **80% active participation** in exercises for exam eligibility; attend all blocks; join your assigned group (no switching); do the pre-reads | Answered by the logistics slides that follow | Logistics | Easy |
| 6 | *(Section question)* "What is data mining and how did it all start?" | Data mining is an interdisciplinary field using scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data across a broad range of application domains | Answered directly on the following definition slide | Data mining definition | Medium |
| 7 | *(Section question)* "Which data types can we analyze with data mining?" | Qualitative (nominal, ordinal) and Quantitative (discrete, continuous); Structured, Unstructured, Semi-structured | Answered by the two "Types of Data" slides | Data types | Medium |
| 8 | *(Section question)* "What will we use to analyze and mine data?" | Python + frameworks: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, RAPIDS, PySpark, HF Transformers, OpenAI GPT, TensorFlow, PyTorch, Optuna, Hyperopt, GridSearchCV; non-Python: MATLAB, RapidMiner, R | Answered by the frameworks slide | Tools | Easy |
| 9 | *(Section question)* "So, what will we address throughout this course?" | The 12 sprints (Intro → Workflow → Processing/Dim. reduction → Prototyping/Deployment → Classification → Clustering → Regression → Neural Networks → GenAI/RAG → Explainability → Ethics → Wrap-up) | Answered by the "What will you learn?" slide | Course outline | Easy |
| 10 | *(Section question)* "I want to commence a career in this field — what are my options afterwards?" | AI Engineer, Data Analyst, Data Scientist, Robotics & Automation expert, ML DevOps (ML Engineer), Data Engineer, Software Engineer | Answered by the career slide | Careers | Easy |
| 11 | *(Section question)* "What is needed for a successful AI-related career as a computer scientist?" | 1) Fundamentals (CS, math, programming); 2) solid foundation in one or more fields (ML/DL, NLP, cloud, data engineering, MLOps, ethics, analysis/visualization, storytelling); 3) deepen a passion area + follow trends | From the 3-step career slide | Careers | Medium |
| 12 | *(Closing slide)* "Questions → Answers" | Open Q&A | End-of-lecture discussion slot | Wrap-up | Easy |

---

## ✅ Final Verification

- [x] All 36 slides covered (intro/polls, definition, applications, life cycle ×4, domain knowledge, representation examples, task types, storytelling, data types ×2, frameworks, career, outline, Q&A)
- [x] The only formula-like content (TF-IDF) explained; pre-read math topics listed
- [x] All important images/diagrams explained (life cycle, spectrogram, Node2Vec, TF-IDF, word-embedding plot, data-type tree, structure table, career diagram)
- [x] All keywords included
- [x] All applicable comparisons created
- [x] All quizzes/polls/section questions collected in Quiz Repository (with inferred answers marked)
- [x] All examples collected in Examples Repository
