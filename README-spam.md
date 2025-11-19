# **Spam Detection Task**

## **1. Overview**

The spam detection component of this project focuses on building a machine learning system that classifies emails as either spam or ham (non-spam).
This task uses two labeled training datasets and one unlabeled test dataset.

For every step of the spam pipeline, we:

* Load and merge the two training CSV datasets
* Normalize and clean the raw text
* Convert text into numerical features using TF-IDF
* Train and compare four model families suited for text classification
* Select the best model based on 5-fold stratified cross validated **F1-score**
* Retrain the best model on all available data
* Generate predictions for the test dataset

All logic for the spam task is implemented in:

```text
spam/main_spam.py
```

---

## **2. Dataset Description**

The spam dataset consists of three CSV files:

| File            | Purpose                        |
| --------------- | ------------------------------ |
| spam_train1.csv | labeled training messages      |
| spam_train2.csv | additional training data       |
| spam_test.csv   | unlabeled messages to classify |

The two training files are concatenated into one combined dataset.
Each CSV contains a text message column (commonly `v2` or `text`) and a label column (`v1`), though the script includes automatic column detection to ensure compatibility.

### **Label Encoding**

Labels in the original files may be strings such as:

* `"spam"`, `"ham"`
* `"1"`, `"0"`
* `"yes"`, `"no"`
* `"true"`, `"false"`

These are mapped to:

* **1 = spam**
* **0 = ham**

### **Text Cleaning**

Before modeling, every message undergoes a custom cleaning routine:

1. Lowercasing
2. Replacing URLs with the placeholder token `"url"`
3. Replacing phone numbers with the token `"phone"`
4. Removing punctuation and non-alphanumeric characters
5. Collapsing repeated whitespace

This ensures consistent representation of common spam indicators and reduces noise.

---

## **3. Preprocessing & Feature Engineering**

To convert text into meaningful numerical features, we use a combined TF-IDF representation:

### **FeatureUnion Components**

**1. Word-level TF-IDF**

* Analyzer: words
* N-grams: unigrams and bigrams
* Stopwords removed
* `min_df` and `max_df` tuned through GridSearch

**2. Character-level TF-IDF**

* Analyzer: character n-grams
* Ranges: 3–5 characters
* Captures short patterns such as:

  * “win”, “free”, “$$$”, “txt”, “msg”
  * partial words that indicate spam triggers

These two components are combined using:

```python
FeatureUnion(transformer_list=[
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])
```

This hybrid representation is widely used in competitive spam detection and performs strongly with linear models and Naive Bayes.

---

## **4. Models Evaluated**

We compare four model families frequently used in spam detection:

### **1. Logistic Regression**

* Strong baseline for TF-IDF features
* Uses balanced class weights
* Solver: *liblinear* or *saga*, depending on configuration

### **2. Linear Support Vector Machine (LinearSVC)**

* Often the strongest text classifier
* Effective in very high dimensional sparse spaces
* Balanced class weighting improves performance on spam minority classes

### **3. Complement Naive Bayes (CNB)**

* Variant of Naive Bayes designed for imbalanced datasets
* Performs well on text with word/char TF-IDF

### **4. Multinomial Naive Bayes (MNB)**

* Classic probabilistic baseline
* Simple, fast, and competitive on short messages

All models are evaluated under a consistent pipeline:

```
TF-IDF Features  →  Model  →  CV F1 Score
```

---

## **5. Cross Validation & Hyperparameter Tuning**

To determine the strongest model, we use:

* **5-fold stratified cross validation**
* **GridSearchCV** using **F1 score** (appropriate for class imbalance)
* Tunable hyperparameters include:

  * Word TF-IDF `min_df`
  * Char TF-IDF `min_df`
  * Regularization strength `C` for SVM/LogReg
  * Laplace smoothing `alpha` for Naive Bayes

This process ensures each model is evaluated fairly on the same folds.

---

## **6. Results**

After running the full pipeline on both training datasets (combined total of **4,296 messages**), each model was evaluated using 5-fold stratified cross validation with **F1 score** as the selection metric.

The following table summarizes the best hyperparameters and cross validated F1 scores for each model:

| Model       | Best Hyperparameters                                                | Best CV F1 Score |
|-------------|---------------------------------------------------------------------|------------------|
| Logistic Regression | `C = 0.5`, `word_min_df = 2`, `char_min_df = 5`            | **0.9547**       |
| Linear SVM (LinearSVC) | `C = 0.5`, `word_min_df = 5`, `char_min_df = 2`         | **0.9571**       |
| Complement Naive Bayes | `alpha = 0.5`, `word_min_df = 2`, `char_min_df = 2`     | **0.9197**       |
| Multinomial Naive Bayes | `alpha = 0.5`, `word_min_df = 2`, `char_min_df = 5`    | **0.9464**       |

### **Selected Model**

The highest performing model was:

### ⭐ **Linear SVM (LinearSVC)**
with a **cross validated F1 score of 0.9571**.

This model demonstrated strong performance on sparse TF-IDF features and handled the class imbalance effectively using balanced class weights.

The selected model was then retrained on **100% of the combined training data** and used to generate the final predictions for the test set.


---

## **7. Output Files**

After training, predictions for the test set are written to:

```text
spam/results/JamisonSpam.txt
```

Each line contains a single integer label:

* `1` = spam
* `0` = ham

Format example:

```
1
0
0
1
...
```

---

## **8. How to Run**

From the project root:

```bash
cd spam
python main_spam.py
```

This executes the full pipeline:

* Loads and merges training CSVs
* Cleans the text
* Builds word- and character-level TF-IDF features
* Performs model selection with GridSearchCV
* Trains the best model
* Creates the final prediction file in `spam/results/`

---

## **9. Summary**

The spam detection task demonstrates a complete text classification workflow:

- Loading and merging two training datasets
- Cleaning messages by replacing URLs, phone numbers, removing punctuation, and normalizing text
- Building TF-IDF features using both word-level and character-level n-grams
- Evaluating four classic text classification models using 5-fold cross validation
- Selecting the best model based on F1 score

Based on cross validated F1 performance, **LinearSVM** was selected as the final model for spam detection, achieving a CV F1 score of **0.9571**. The final predictions were generated and saved according to the project requirements.
