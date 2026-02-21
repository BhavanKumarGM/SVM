import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =========================
# 1. DATA GENERATION
# =========================
X, y = make_blobs(
    n_samples=200,
    centers=2,
    cluster_std=1.5,
    random_state=42
)

# =========================
# 2. TRAIN–TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =========================
# 3. LINEAR SVM PIPELINE
# =========================
linear_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1))
])

linear_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))

# =========================
# 4. RBF SVM PIPELINE
# =========================
rbf_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1, gamma=0.5))
])

rbf_svm.fit(X_train, y_train)

y_pred_rbf = rbf_svm.predict(X_test)
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))

# =========================
# 5. ACCESS SUPPORT VECTORS
# =========================
svm_model = rbf_svm.named_steps["svm"]

print("Number of support vectors:", svm_model.n_support_)
print("Support vectors shape:", svm_model.support_vectors_.shape)

# =========================
# 6. VISUALIZATION FUNCTION
# =========================
def plot_svm(model, X, y, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 40)
    yy = np.linspace(ylim[0], ylim[1], 40)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)

    # Decision boundary & margins
    ax.contour(
        XX, YY, Z,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        colors="black"
    )

    # Support vectors
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=120,
        linewidth=1.5,
        facecolors="none",
        edgecolors="black"
    )

    plt.title(title)
    plt.show()

# =========================
# 7. PLOT RBF SVM
# =========================
plot_svm(
    svm_model,
    X_train,
    y_train,
    "SVM (RBF Kernel) – Decision Boundary, Margin & Support Vectors"
)