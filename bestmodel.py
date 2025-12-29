import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("precorrect.csv")

# Define features and targets
X = df.drop(columns=["Primary_Issue", "Secondary_Issue", "Tertiary_Issue"])
y = df[["Primary_Issue", "Secondary_Issue", "Tertiary_Issue"]].copy()

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Scale numerical features
X[X.select_dtypes(include="number").columns] = StandardScaler().fit_transform(X.select_dtypes(include="number"))

# Encode each label column
label_encoders = {}
for col in y.columns:
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": MultiOutputClassifier(RandomForestClassifier(random_state=42)),
    "ANN (MLPClassifier)": MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)),
    "KNN": MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
}

# (Optional) XGBoost
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42))
except ImportError:
    print("❌ XGBoost not available — skipping...")

# Train and evaluate each model
for name, model in models.items():
    print(f"\n=== Multi-Label Report: {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    for i, col in enumerate(y.columns):
        acc = accuracy_score(y_test[col], y_pred[:, i])
        print(f"\n-- {col} --")
        print("Accuracy:", round(acc * 100, 2), "%")
        print(classification_report(y_test[col], y_pred[:, i], zero_division=0))
