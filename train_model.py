import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("precorrect.csv")

# Fix 'None' values
columns_to_fix = [
    "Preexisting_Conditions", "Recent_Illnesses", "Family_History",
    "Diet_Type", "Physical_Activity", "Smoking_Status", "Alcohol_Consumption"
]
for col in columns_to_fix:
    df[col] = df[col].fillna("None").replace({None: "None"})

# Define features and target
X = df.drop(columns=["Primary_Issue", "Secondary_Issue", "Tertiary_Issue"])
y = df["Primary_Issue"]

# Encode categorical features
feature_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X[X.select_dtypes(include="number").columns] = scaler.fit_transform(X.select_dtypes(include="number"))

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy:.4f}")

# Save model and encoders
with open("model_topk.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoders_topk.pkl", "wb") as f:
    pickle.dump((feature_encoders, label_encoder, scaler), f)
