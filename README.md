# 🎭 End-to-End Sentiment Classification: From Classical ML to Deep Learning Deployment

## 1. Project Overview
In text analytics, accurately parsing human emotion and sentiment from unstructured textual data is a core engineering capability. This project builds a comprehensive, **End-to-End Sentiment Classification Pipeline** that handles raw, messy text data, evaluates traditional and sequential deep learning architectures, and concludes with an active real-world deployment state.

The key highlight of this project is the strict empirical validation across dynamic data cuts. By benchmarking multiple model architectures, the optimized candidate is extracted, frozen, and serialized to serve live predictions via an interactive web interface and evaluated globally on unseen holdout boundaries.

### Core Objectives:
* **Text Engineering Pipeline:** Design a clean text preprocessing workflow utilizing regular expression masks, emoji text conversions, and semantic deduplication.
* **Algorithmic Benchmarking:** Conduct a rigorous comparison across 6 traditional machine learning models using statistical TF-IDF text representations.
* **Deep Learning Exploration:** Architect and analyze sequential neural networks (**Bidirectional LSTM**) using embedded tokenization strategies.
* **Holdout Validation & Deployment:** Generate and verify production predictions on an anonymous 5,000-record Kaggle holdout dataset and build an interactive **Streamlit web application**.

---

## 2. Environment & Dependency Setup
To support both rapid feature vectorization and GPU-accelerated sequential neural network training, the workspace environment was configured with strict version boundaries to guarantee full execution reproducibility.

* **Platform:** Linux
* **Python Version:** 3.11.0 / 3.12 (GCC Accelerated Layout)
* **Compute Engine:** NVIDIA CUDA GPU Support Natively Enabled

### Core Libraries Frameworks:

| Library | Version / Channel |
| --- | --- |
| `torch` | 2.2.2+cu121 |
| `tensorflow` / `keras` | Production Cloud Stable |
| `ultralytics` | 8.3.27 (Preserved Infrastructure) |
| `xgboost` | Production Baseline |
| `emoji` | Dynamic Demojize Packaging |
| `nltk` | Corpora (WordNet, Stopwords) |

---

## 3. Data Transformation & Text Engineering
The raw dataset consists of a consolidated pool of text frames. Due to the messy nature of social/web textual streams, an atomic cleansing engine was designed within a unified `wrangle` routine:

1. **programmatical Cleaners:** Stripped out residual web structures and formatting using `BeautifulSoup` and targeted regular expressions (`remove_html_tags`, `remove_urls`).
2. **Shorthand Mapping & Demojization:** Implemented a custom internet chat-word dictionary lookup to expand contractions (e.g., `LOL` to `Laughing Out Loud`) and leveraged the `emoji` library to map graphical nodes into string text tokens (e.g., `🟢` to `:green_circle:`).
3. **Linguistic Refactoring:** Enforced lowercase conversion, eliminated numerical data artifacts, and pruned graphical punctuation characters using high-speed string mapping.
4. **Semantic Deduplication:** Performed precise text-subset analysis to drop **353 duplicate rows** post-cleansing, safely mitigating any risks of data leakage or artificial metric inflation before splitting.
5. **Morphological Normalization:** Standardized root structures utilizing the `WordNetLemmatizer` engine to maximize cross-token alignment.

---

## 4. Classical Machine Learning Benchmarking
The processed, balanced textual matrix was split into a **80% Training Set (32,000 instances)** and a **20% Stratified Test Set (8,000 instances)**. A benchmark loop evaluated 6 distinct pipelines backed by `TfidfVectorizer` mapping text features.

### Algorithmic Evaluation Summary:
* **Baseline Majority-Class Reference:** 50.00%
* **Naive Bayes:** Test Accuracy: 86.36% | F1-Score: 0.864
* **Decision Tree:** Test Accuracy: 71.66% | F1-Score: 0.717
* **Logistic Regression:** **Test Accuracy: 89.19% | F1-Score: 0.892 (The Champion Model)**
* **Random Forest:** Test Accuracy: 84.92% | F1-Score: 0.849
* **KNN:** Test Accuracy: 77.59% | F1-Score: 0.775
* **XGBoost:** Test Accuracy: 85.31% | F1-Score: 0.853

<p align="center">
  <img src="assets/Linguistic Algorithmic Performance Benchmarks.png" alt="Traditional ML Models Accuracy Chart" width="700">
</p>

---

## 5. Deep Learning Exploration (Sequential Bidirectional LSTM)
To explore deep sequence learning, an optimized **Bidirectional LSTM** framework was deployed using Keras embedded tokenization across an expanded vocabulary footprint (`vocab_size=10,000` nodes) and post-padding sequences (`max_length=150`).

### Architectural Performance Diagnostics:
* **Hardware Acceleration:** Omitting non-cuDNN recurrent dropouts successfully unlocked native GPU kernels, accelerating runtime to ~200s per epoch.
* **Peak Generalization:** The sequential network located its optimal generalized weights exceptionally early, peaking at a **Validation Accuracy of 87.66% at Epoch 2** (Validation Loss: `0.3016`).
* **Overfitting Trjectory:** Beyond Epoch 3, overfitting occurred as training accuracy expanded to `93.75%` while validation loss rebounded to `0.3388`. The `Early Stopping` callback gracefully terminated execution at Epoch 5, rolling back to preserve the optimized weights from Epoch 2.

### Production Selection Decision (Occam's Razor)
While the custom Bidirectional LSTM achieved a strong score of 87.66%, the classical **TF-IDF + Logistic Regression pipeline** demonstrably out-performed it, securing **89.19% Accuracy**. Following the law of *Occam's Razor*, the Logistic Regression pipeline was chosen as the **Production Champion Model** due to its superior score, lack of overfitting, ultra-fast inference speed, and light memory footprint.

---

## 6. Real-World Holdout Test Set Evaluation (Kaggle Verification)
To conduct an unbiased, external validation, the serialized production pipeline processed an un-labeled 5,000-record Kaggle holdout testing partition (`Test.csv`). Predictions were mapped programmatically and uploaded to the Kaggle evaluation engine.

### Verified Production Scores:
* **Kaggle Public Score:** **0.89100** (~89.10% accuracy)
* **Kaggle Private Score:** **0.89400** (~89.40% accuracy)

The near-identical alignment between the training split tests (**89.19%**) and the external private holdout verification (**89.40%**) mathematically confirms the model's exceptional stability and production-grade generalization capabilities on completely unseen linguistic structures.

<p align="center">
  <img src="assets/final score.png" alt="Kaggle Submission Verification Score" width="750">
</p>

---

## 7. Production Model Serialization & Web Deployment
The entire integrated champion pipeline was frozen and serialized into a deployment asset (`sentiment_pipeline_model.pkl`) using `joblib`. 

An active, lightweight user-facing web dashboard was built using **Streamlit** to serve predictions live. The interface allows users to type custom text arrays, handles feature extraction on the fly, logs inference speed, and outputs visual color-coded sentiment metrics instantly.

### How to Run the Application Locally

To execute and test the interactive Streamlit dashboard on your local machine, follow these standard steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd end-to-end-sentiment-classification
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then install the required production libraries via the bundled package manager:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   Start the local host execution stream by targeting the native user interface script:
   ```bash
   streamlit run app.py
   ```

---

## 8. Conclusion & Future Directions
* **Summary:** This project completed a full-cycle product lifecycle—transitioning from messy raw token handling, comparative machine learning modeling, sequential deep learning diagnostics, and external blind test validation, to an operational web interface.
* **Future Upgrades:** 
    * Scale representation capabilities by replacing statistical TF-IDF blocks with modern transformer models such as **BERT** or fine-tuned **DistilBERT** embeddings.
    * Containerize the Streamlit dashboard structure using **Docker** and deploy the microservice to cloud instances like Hugging Face Spaces or AWS EC2.
