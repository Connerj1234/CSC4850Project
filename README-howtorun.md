# How to Run the Project

This project contains two separate parts:

1. **Numeric Classification** (4 datasets)
2. **Spam Detection** (text classification)

## 1. Install Dependencies

Make sure you have Python 3 installed.
Then install the required packages:

```bash
pip install numpy pandas scikit-learn scipy
```

---

## 2. Run the Classification Task

From the project root:

```bash
cd classification
python main_classification.py
```

This script will:

* Load the four classification datasets
* Train and evaluate models
* Generate prediction files in `classification/results/`

---

## 3. Run the Spam Detection Task

From the project root:

```bash
cd spam
python main_spam.py
```

This script will:

* Load and clean the spam datasets
* Train and evaluate models
* Generate the final predictions in `spam/results/`

---

## 4. Output Files

After running both scripts, you will find prediction outputs in:

* `classification/results/`
* `spam/results/`
