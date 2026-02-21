# 🧠 Understanding Support Vector Machines, One Margin at a Time  
*(End-to-End SVM Implementation using scikit-learn)*

---

## 🌟 Project Overview

This project demonstrates a **complete, end-to-end implementation of Support Vector Machines (SVM)** using **scikit-learn**, with a strong focus on **conceptual clarity**.

It goes beyond `.fit()` and `.predict()` to show **how SVM actually works**, including:

- Margin construction  
- Support vector identification  
- Non-linear decision boundaries using the **RBF kernel**

---

## 🎯 What This Project Aims to Do

- Build **Linear and RBF SVM classifiers**
- Show why **feature scaling is mandatory**
- Visualize:
  - Decision boundary
  - Margin
  - Support vectors
- Explain the effect of hyperparameters (`C`, `gamma`)
- Help learners **see what SVM is doing internally**

---

## 🧩 Key Concepts Covered

- 📐 Maximum-margin hyperplane  
- 🎯 Support vectors (the only points that matter)  
- 🔁 Kernel trick for non-linear separation  
- ⚖️ Bias–variance tradeoff using `C`  
- 🌊 Boundary smoothness controlled by `gamma`

---

## 🛠️ Tech Stack

- 🐍 Python 3.8+  
- 📦 scikit-learn  
- 📊 NumPy  
- 📉 Matplotlib  

---

## 📂 Project Structure

```text
.
├── svm_sklearn.py     # Complete SVM pipeline (training + visualization)
├── README.md          # Project documentation
