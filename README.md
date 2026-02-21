🧠 Understanding Support Vector Machines, One Margin at a Time

(End-to-End SVM Implementation using scikit-learn)

🌟 Project Overview

This project demonstrates a complete, end-to-end implementation of Support Vector Machines (SVM) using scikit-learn, with a strong focus on conceptual clarity.

It goes beyond .fit() and .predict() to show how SVM actually works, including:

Margin construction

Support vector identification

Non-linear decision boundaries using the RBF kernel

🎯 What This Project Aims to Do

Build Linear and RBF SVM classifiers

Show why feature scaling is mandatory

Visualize:

Decision boundary

Margin

Support vectors

Explain the effect of hyperparameters (C, gamma)

Help learners see what SVM is doing internally

🧩 Key Concepts Covered

📐 Maximum-margin hyperplane

🎯 Support vectors (the only points that matter)

🔁 Kernel trick for non-linear separation

⚖️ Bias–variance tradeoff using C

🌊 Boundary smoothness controlled by gamma

🛠️ Tech Stack

🐍 Python 3.8+

📦 scikit-learn

📊 NumPy

📉 Matplotlib

📂 Project Structure
.
├── svm_sklearn.py     # Complete SVM pipeline (training + visualization)
├── README.md          # Project documentation
▶️ How to Run the Project
1️⃣ Install dependencies
pip install numpy matplotlib scikit-learn
2️⃣ Run the script
python svm_sklearn.py
📊 Output You’ll See

🔵🔴 Data points from two classes

➖ Solid curve → Decision boundary

➖➖ Dashed curves → Margin boundaries

⭕ Circled points → Support vectors

📌 The curved boundary appears because the RBF kernel maps data into a higher-dimensional space where separation becomes linear.

⚙️ Hyperparameters Explained
🔧 C – Regularization Strength

High C → smaller margin, fewer errors, overfitting risk

Low C → wider margin, more errors, underfitting risk

🔧 gamma – Kernel Influence

High gamma → very complex, wiggly boundary

Low gamma → smoother, simpler boundary

🚫 Common Mistakes This Project Avoids

❌ Training SVM without scaling features

❌ Blindly using RBF kernel

❌ Ignoring support vectors

❌ Evaluating only on training data

👨‍🎓 Who This Project Is For

Students learning Machine Learning fundamentals

Engineers who want intuition + implementation

Interview preparation (conceptual depth)

Anyone tired of black-box ML
