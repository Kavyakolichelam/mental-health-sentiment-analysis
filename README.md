# Social Network Sentiment Analysis for Mental Health Awareness

> An end-to-end NLP and exploratory data analysis project examining 38,000+ Reddit posts about mental health. Combines pre-trained sentiment models (VADER, DistilBERT) with K-means thematic clustering and visualization-driven insights to surface patterns in online mental health discourse.

## 📊 Project at a Glance

| Aspect | Detail |
|---|---|
| **Dataset** | 38,033 Reddit posts (350 features) |
| **Source** | Mental health–focused subreddits |
| **Pre-trained sentiment models** | VADER (rule-based) + DistilBERT (transformer) |
| **Clustering approach** | K-means (k=5) on TF-IDF features |
| **Dimensionality reduction** | PCA (2D) for cluster visualization |
| **Sentiment classes** | Positive / Neutral / Negative (VADER); Positive / Negative (DistilBERT) |

---

## 🎯 Problem

Mental health discussions on social media are abundant but unstructured. A single subreddit can contain thousands of posts with overlapping themes, varying emotional valence, and uneven posting cadence. This project asks: **can we use NLP and clustering to surface meaningful patterns in this noisy data — and what themes emerge when we let the data cluster itself?**

The goal is not to build a state-of-the-art sentiment classifier (pre-trained models already do that well). The goal is to demonstrate an **end-to-end exploratory pipeline** that a mental-health-awareness team or researcher could use to understand what people are talking about, how the tone shifts over time, and which subreddits drive the conversation.

## 🧠 Approach

A six-stage exploratory pipeline:

```
Raw Reddit posts (38,033 rows × 350 columns)
        ↓
[1] Data cleaning · text normalization · date parsing · missing value handling
        ↓
[2] TF-IDF vectorization (top 1,000 features, English stop-words removed)
        ↓
[3] Sentiment labeling
    ├─ VADER  (3-class: positive / neutral / negative)
    └─ DistilBERT (binary: positive / negative)
        ↓
[4] K-means clustering (k=5) → thematic groups
        ↓
[5] PCA dimensionality reduction → 2D cluster visualization
        ↓
[6] Visualization & insight extraction
    (sentiment distribution · temporal trends · word clouds · cluster keywords)
```

**Why this design:**
- **Pre-trained models** were used deliberately — building a sentiment classifier from scratch on this dataset would underperform DistilBERT, so the time was better spent on analysis
- **Unsupervised clustering** lets the data reveal natural groupings instead of forcing prior assumptions about what categories should exist
- **Visualization-first** because the audience for this kind of analysis is rarely technical — the deliverable is interpretable charts, not just numbers

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?logoColor=white)
![WordCloud](https://img.shields.io/badge/WordCloud-1A1A1A?logoColor=white)

## 📦 Dataset

A pre-processed Reddit dataset containing posts from mental-health-focused subreddits, with the following structure:

| Field | Type | Description |
|---|---|---|
| `post` | text | Raw post content |
| `subreddit` | categorical | Source subreddit |
| `author` | anonymized | Anonymized user identifier |
| `date` | datetime | Post creation date |
| TF-IDF features | numerical | 256-dim pre-computed vectors |

**Total:** 38,033 posts across 350 columns.

## 📁 Project Structure

```
mental-health-sentiment-analysis/
├── notebook.ipynb               # Full analysis pipeline
├── docs/
│   ├── Final_Project.docx       # Project report
│   └── documentation.docx       # Methodology documentation
└── README.md
```

## 🚀 Reproducing the Results

```bash
# Install dependencies
pip install pandas numpy scikit-learn nltk spacy transformers matplotlib seaborn wordcloud

# Download VADER lexicon
python -c "import nltk; nltk.download('vader_lexicon')"

# Open notebook
jupyter notebook notebook.ipynb
```

The notebook walks through all six pipeline stages with intermediate visualizations and saved outputs.

## 📈 Key Findings

### Thematic Clusters (K-means, k=5)

Top keywords surfaced for each cluster reveal distinct discussion themes:

| Cluster | Top Keywords | Interpretation |
|---|---|---|
| 0 | don, want, just, know, life | Existential / decision-related |
| 1 | im, dont, just, like, feel | First-person emotional expression |
| 2 | just, depression, life, like, help | Depression-specific support seeking |
| 3 | ve, just, like, life, don | Experiential reflection |
| 4 | feel, like, just, don, know | Affective uncertainty |

### Temporal Patterns
- Posting volume shows clear monthly variation
- Negative sentiment shows spikes around specific events and holiday periods

### Sentiment Distribution
- VADER-labeled sentiment is heavily skewed toward negative across mental-health subreddits (as expected given the topic)
- DistilBERT binary labeling shows similar negative-class dominance

### Linguistic Patterns
- Higher word counts correlate with stronger sentiment intensity in either direction
- Frequent terms across negative posts include *feel*, *life*, *depression* — surfaced via both word-cloud and TF-IDF analysis

## 🔍 What This Project Demonstrates

- **End-to-end NLP workflow** from raw text through cleaning, feature engineering, and analysis
- **Integration of pre-trained models** (VADER, DistilBERT via HuggingFace Transformers) rather than rebuilding sentiment classifiers from scratch
- **Unsupervised theme discovery** through K-means clustering on TF-IDF features
- **Visualization-driven analysis** appropriate for non-technical stakeholders
- **Honest scope** — uses the right tool for the job rather than over-engineering

## ⚠️ Limitations and Honest Framing

- **Pre-trained models, not custom-trained** — VADER and DistilBERT were used out-of-the-box; no fine-tuning on the target domain
- **No formal model evaluation against labeled ground truth** — the dataset doesn't include human-annotated sentiment labels, so model accuracy on this specific data is not directly measured
- **K-means cluster count (k=5) chosen heuristically** — not optimized via silhouette/elbow analysis
- **English-only**, single-platform (Reddit), and not real-time
- **Cluster keyword overlap** (many clusters contain *just*, *like*, *feel*, *don*) suggests TF-IDF + K-means is finding texture rather than crisp topic separation — a known limitation when stop-word-adjacent terms dominate the vocabulary

## 🔭 Future Work

- **Domain fine-tuning** of DistilBERT or a similar transformer on mental-health-specific labeled data (e.g., DAIC-WOZ, CLPsych)
- **Topic modeling** (LDA, BERTopic) to replace K-means + TF-IDF with a method better suited to surfacing semantic themes
- **Multi-platform expansion** — Twitter, Instagram, or Facebook posts for cross-platform comparison
- **Real-time streaming** pipeline for live trend monitoring
- **Interactive dashboard** (Streamlit or Plotly Dash) for non-technical stakeholders

## 👥 Authors

This project was completed collaboratively at Lawrence Technological University:

- **Kavya Kolichelam** — kkolichel@ltu.edu
- **Bhavyasri Lakkamraju** — blakkamra@ltu.edu

Department of Mathematics and Computer Science
Lawrence Technological University, Southfield, Michigan, USA

## 📬 Contact

**Kavya Kolichelam**
📧 kavyakolichelam@gmail.com
💼 [LinkedIn](https://www.linkedin.com/in/kavya-kolichelam-54452a317)
💻 [GitHub](https://github.com/Kavyakolichelam)

---

⭐ If you found this project useful, please consider starring the repository.
