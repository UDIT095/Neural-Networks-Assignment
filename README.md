# 🔠 Artificial Neural Networks for Alphabet Classification with Hyperparameter Tuning

## 📌 Project Title  
**Classification Using Artificial Neural Networks with Hyperparameter Tuning on Alphabets Data**

---

## 📖 Overview  
This project builds a classification model using **Artificial Neural Networks (ANNs)** to recognize alphabets from numerical character features. The project emphasizes the role of **hyperparameter tuning** in enhancing model performance using **Keras Tuner**.

---

## 📂 Dataset  
**File:** `Alphabets_data.csv`  
- 20,000 samples  
- 16 numerical features  
- 1 target column `letter` (A–Z)  

### 🧾 Columns Description  
- **letter**: Target label (A-Z)  
- **xbox, ybox, width, height**: Bounding box and dimensions  
- **onpix**: Number of 'on' pixels  
- **xbar, ybar, x2bar, y2bar**: Statistical position metrics  
- **xybar, x2ybar, xy2bar**: Position interactions  
- **xedge, xedgey, yedge, yedgex**: Edge crossing stats

---

## 📁 Files in this Repository
- `Alphabets_data.csv` – Dataset  
- `Neural Networks.ipynb` – Code for ANN, preprocessing, and hyperparameter tuning  
- `Neural Networks.docx` – Assignment overview, dataset details, and evaluation criteria  

---

## 🛠️ Tools and Libraries  
- `pandas`, `numpy` – Data handling  
- `scikit-learn` – Splitting, scaling, label encoding, metrics  
- `tensorflow.keras` – ANN model building  
- `keras-tuner` – Hyperparameter tuning  
- `Jupyter Notebook` – Development environment  

---

## 🔄 Project Structure and Workflow

### 1. 🧹 Data Exploration and Preprocessing
- **Load dataset** using `pandas`
- **Explore structure**: Shape, class distribution, missing values  
- **Scale numerical features** using `StandardScaler`  
- **Label Encode** the `letter` column to numeric values (0–25)  
- **Split data** (80% training, 20% test)

### 2. 🤖 Baseline ANN Model
- **Sequential model architecture**:
  - Input layer (16 features)  
  - 2 hidden layers (`Dense + ReLU`)  
  - Output layer (`Softmax`, 26 classes)  
- **Compilation**:
  - Optimizer: `adam`  
  - Loss: `sparse_categorical_crossentropy`  
  - Metric: `accuracy`  
- **Training**:
  - 50 epochs  
  - Validation split: 20%

### 3. ⚙️ Hyperparameter Tuning (Keras Tuner)
- **Tunable parameters**:
  - Hidden units (32–256)  
  - Dropout rate (0.0–0.5)  
  - Learning rate (1e-2, 1e-3, 1e-4)  
- **Tuner setup**: `kt.Hyperband`  
  - Objective: `val_accuracy`  
  - Max epochs: 50  
  - Early stopping to prevent overfitting  
- **Execution**:
  - `tuner.search()`  
  - Retrieve best model and retrain with optimal parameters  

### 4. ✅ Evaluation
- **Test Accuracy**: ~96.8% (tuned model)  
- **Classification Report**:  
  - Precision, recall, F1-score per letter  
  - Consistent high scores across all 26 classes  
- **Comparison**:
  - Tuned model outperforms baseline (e.g., 92% → 96.8%)  
  - Validation/test scores are closely matched — good generalization

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn tensorflow keras-tuner
```

### 3. Run the Notebook
```bash
jupyter notebook
```

- Open `Neural Networks.ipynb`
- Run all cells to preprocess data, train baseline model, tune hyperparameters, and evaluate performance

---

## 📊 Key Takeaways
- ANN with basic architecture can achieve good accuracy
- Keras Tuner significantly improves performance by automating hyperparameter selection
- Data scaling and label encoding are critical preprocessing steps
- Final model achieves **96.8% accuracy**, demonstrating robust performance across all alphabet classes
