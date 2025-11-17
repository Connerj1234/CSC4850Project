Classification Task
1. Overview

The Classification portion of this project involves training machine-learning models on four separate datasets, each representing a standalone classification problem.
For each dataset, the objective is to:

Train a model on the provided training data

Predict labels for the corresponding test data

Output a text file with one predicted label per line

All logic for this task is implemented in:

classification/main_classification.py

2. Dataset Description

Each dataset contains three files:

Dataset	Training Data	Training Labels	Test Data
1	TrainData1.txt	TrainLabel1.txt	TestData1.txt
2	TrainData2.txt	TrainLabel2.txt	TestData2.txt
3	TrainData3.txt	TrainLabel3.txt	TestData3.txt
4	TrainData4.txt	TrainLabel4.txt	TestData4.txt

All files are whitespace-separated numeric matrices.

Missing Values

Missing entries are encoded as:

1.00000000000000e+99


These are treated as missing values and imputed during preprocessing.

3. Approach & Methodology
Step 1 — Loading Data

For each dataset k, the script loads:

TrainData{k}.txt  → X_train
TrainLabel{k}.txt → y_train
TestData{k}.txt   → X_test


All files are read using NumPy.

Step 2 — Handling Missing Values

Missing values encoded as 1e99 are replaced using:

SimpleImputer(strategy="mean", missing_values=1e99)


This imputes the mean value of each feature column.

Step 3 — Feature Scaling

Each dataset is standardized using:

StandardScaler()


Standardization ensures that all features operate on comparable scales, which benefits models such as logistic regression.

Step 4 — Model Selection

A pipeline is constructed using:

Mean imputation

Standard scaling

Logistic Regression classifier

Logistic Regression was chosen because it:

Performs well on high-dimensional numeric data

Includes built-in regularization

Trains efficiently across all four datasets

Step 5 — Hyperparameter Tuning

To improve performance and reduce overfitting, the model undergoes 5-fold stratified cross validation through GridSearchCV.

The hyperparameter grid:

C = [0.01, 0.1, 1, 10]


For each dataset, the script prints:

Best C value

Cross-validated accuracy

These results will be included in the final project report.

Step 6 — Final Training & Predictions

After selecting the best hyperparameters, the final model is retrained on all available training data for that dataset.

The script then predicts the labels for the test dataset and saves them to:

classification/results/YourLastNameClassification{k}.txt


Format:

1
3
2
2
...


(One label per line.)

4. How to Run

From the project root:

cd classification
python main_classification.py


This will:

Load and preprocess each dataset

Perform cross validation

Select the best model

Train on full data

Generate prediction files in classification/results/

6. Expected Output

After running the script, the folder:

classification/results/


will contain:

YourLastNameClassification1.txt
YourLastNameClassification2.txt
YourLastNameClassification3.txt
YourLastNameClassification4.txt


Each file contains the predicted class labels for the corresponding test dataset.
