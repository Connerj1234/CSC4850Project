````markdown
## **Classification Task**

### **1. Overview**

The classification part of this project uses four separate numeric datasets.
Each dataset is treated as an independent classification problem.

For every dataset we:

- Load the training features, training labels, and test features
- Impute missing values
- Standardize features where appropriate
- Train and compare four different model families
- Select the best model based on cross validation accuracy
- Retrain the best model on all training data
- Generate a prediction file for the test set

All logic is implemented in:

```text
classification/main_classification.py
````

---

### **2. Dataset Description**

Each dataset consists of three files:

| Dataset | Train Data     | Train Labels    | Test Data     | Train Shape | Test Shape  |
| ------- | -------------- | --------------- | ------------- | ----------- | ----------- |
| 1       | TrainData1.txt | TrainLabel1.txt | TestData1.txt | (150, 3312) | (53, 3312)  |
| 2       | TrainData2.txt | TrainLabel2.txt | TestData2.txt | (100, 9182) | (74, 9182)  |
| 3       | TrainData3.txt | TrainLabel3.txt | TestData3.txt | (2547, 112) | (1092, 112) |
| 4       | TrainData4.txt | TrainLabel4.txt | TestData4.txt | (1119, 11)  | (480, 11)   |

All files are whitespace separated numeric matrices.

#### **Missing Values**

Missing values in the feature matrices are encoded as:

```text
1.00000000000000e+99
```

These sentinel values are treated as missing and imputed using the mean of each feature column.

---

### **3. Preprocessing Pipeline**

For each model, preprocessing is handled through a scikit learn `Pipeline`:

1. **Imputation**

   ```python
   SimpleImputer(missing_values=1e99, strategy="mean")
   ```

   Replaces the sentinel value with the mean of the corresponding feature.

2. **Standardization**
   For linear models and neural networks we apply:

   ```python
   StandardScaler()
   ```

   This transforms each feature to zero mean and unit variance.

3. **No scaling for trees**
   Random Forest models use imputation but do not require feature scaling.

---

### **4. Models Evaluated**

For each dataset we evaluate four model families:

1. **Logistic Regression**

   * Linear classifier with L2 regularization
   * Strong baseline for high dimensional numeric data

2. **Linear SVM (LinearSVC)**

   * Linear support vector machine
   * Often performs very well on high dimensional spaces

3. **Random Forest Classifier**

   * Ensemble of decision trees
   * Captures nonlinear relationships and feature interactions
   * Well suited for lower dimensional datasets with enough samples

4. **MLP Neural Network (MLPClassifier)**

   * Fully connected feedforward neural network
   * Tested with different regularization strengths
   * Most effective for medium sized datasets with moderate feature counts

Each model family is wrapped in a `Pipeline` together with preprocessing so that cross validation treats the entire workflow consistently.

---

### **5. Cross Validation and Hyperparameter Tuning**

To compare models fairly we use:

* **Stratified K fold cross validation with 5 folds**

  ```python
  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  ```
* **GridSearchCV** with accuracy as the scoring metric

Hyperparameter grids (per model family):

* **Logistic Regression**

  * Parameter: `C`
  * Grid: `[0.01, 0.1, 1.0, 10.0]`

* **Linear SVM (LinearSVC)**

  * Parameter: `C`
  * Grid: `[0.01, 0.1, 1.0, 10.0]`

* **Random Forest**

  * Parameters: `max_depth`, `min_samples_split`
  * Grid examples (dataset specific defaults):

    * `max_depth`: values such as `None`, `5`, `7`, `10`, `20`
    * `min_samples_split`: `[2, 5]`

* **MLP Neural Network**

  * Parameter: `alpha` (L2 regularization term)
  * Grid: `[0.0005, 0.001, 0.01]`
  * Hidden layer sizes differ for high dimensional and lower dimensional datasets.

For each dataset and model, `GridSearchCV` reports the best hyperparameters and the mean cross validated accuracy.
The overall best model per dataset is then selected based on this accuracy.

Some warnings appear during training:

* **UserWarning** for Dataset 1 due to a very small minority class when using 5 folds
* **ConvergenceWarning** for some SVM and MLP configurations that did not fully converge
* **FutureWarning** regarding the default `dual` parameter in `LinearSVC`

These warnings do not affect the final selected models and are noted but not critical for the workflow.

---

### **6. Results per Dataset**

#### **Summary Table**

| Dataset | Best Model          | Best Hyperparameters                        | Best CV Accuracy |
| ------- | ------------------- | ------------------------------------------- | ---------------- |
| 1       | Logistic Regression | `C = 1.0`                                   | **0.9800**       |
| 2       | Linear SVM          | `C = 0.01`                                  | **0.9600**       |
| 3       | Random Forest       | `max_depth = None`, `min_samples_split = 2` | **0.9627**       |
| 4       | Random Forest       | `max_depth = 20`, `min_samples_split = 2`   | **0.6827**       |

#### **Dataset 1**

* **Train shape**: (150, 3312)
* **Test shape**: (53, 3312)

Models and best cross validated accuracies:

* Logistic Regression

  * Best `C`: 1.0
  * Best CV accuracy: 0.9800
* Linear SVM

  * Best `C`: 0.01
  * Best CV accuracy: 0.8667
* Random Forest

  * Best `max_depth`: 5
  * Best `min_samples_split`: 2
  * Best CV accuracy: 0.9000
* MLP

  * Best `alpha`: 0.0005
  * Best CV accuracy: 0.9333

**Selected model**: Logistic Regression with `C = 1.0`
**Reason**: Highest cross validated accuracy among all models.

---

#### **Dataset 2**

* **Train shape**: (100, 9182)
* **Test shape**: (74, 9182)

Models and best cross validated accuracies:

* Logistic Regression

  * Best `C`: 0.1
  * Best CV accuracy: 0.9100
* Linear SVM

  * Best `C`: 0.01
  * Best CV accuracy: 0.9600
* Random Forest

  * Best `max_depth`: 7
  * Best `min_samples_split`: 5
  * Best CV accuracy: 0.8800
* MLP

  * Best `alpha`: 0.0005
  * Best CV accuracy: 0.8700

**Selected model**: Linear SVM with `C = 0.01`
**Reason**: Strong performance on extremely high dimensional data and highest CV accuracy.

---

#### **Dataset 3**

* **Train shape**: (2547, 112)
* **Test shape**: (1092, 112)

Models and best cross validated accuracies:

* Logistic Regression

  * Best `C`: 1.0
  * Best CV accuracy: 0.8547
* Linear SVM

  * Best `C`: 0.1
  * Best CV accuracy: 0.8516
* Random Forest

  * Best `max_depth`: None
  * Best `min_samples_split`: 2
  * Best CV accuracy: 0.9627
* MLP

  * Best `alpha`: 0.0005
  * Best CV accuracy: 0.9160

**Selected model**: Random Forest with `max_depth = None`, `min_samples_split = 2`
**Reason**: Significantly higher CV accuracy than the linear models and the MLP, indicating that nonlinear relationships are important in this dataset.

---

#### **Dataset 4**

* **Train shape**: (1119, 11)
* **Test shape**: (480, 11)

Models and best cross validated accuracies:

* Logistic Regression

  * Best `C`: 1.0
  * Best CV accuracy: 0.5996
* Linear SVM

  * Best `C`: 10.0
  * Best CV accuracy: 0.5800
* Random Forest

  * Best `max_depth`: 20
  * Best `min_samples_split`: 2
  * Best CV accuracy: 0.6827
* MLP

  * Best `alpha`: 0.0005
  * Best CV accuracy: 0.6185

**Selected model**: Random Forest with `max_depth = 20`, `min_samples_split = 2`
**Reason**: Best cross validated accuracy among all tested models on this low dimensional dataset.

---

### **7. Output Files**

For each dataset the final selected model is retrained on the full training set and used to predict labels for the test set.
Predictions are saved as plain text files with one integer label per line:

```text
classification/results/JamisonClassification1.txt
classification/results/JamisonClassification2.txt
classification/results/JamisonClassification3.txt
classification/results/JamisonClassification4.txt
```

The filenames follow the required naming convention:

```text
<LastName>Classification<k>.txt
```

---

### **8. How to Run**

From the project root:

```bash
cd classification
python main_classification.py
```

This command will:

* Load and preprocess each of the four datasets
* Train and tune the four model families per dataset using stratified cross validation
* Select the best model for each dataset based on CV accuracy
* Retrain the best model on all training data
* Generate the prediction files in `classification/results/`

These cross validated accuracies and selected models are then used in the written report to justify model choices for each dataset.

```
```
