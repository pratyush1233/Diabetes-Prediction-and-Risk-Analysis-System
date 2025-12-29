import pickle
import pandas as pd
import numpy as np

# Load model and encoders
with open("model_topk.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders_topk.pkl", "rb") as f:
    feature_encoders, label_encoder, scaler = pickle.load(f)

# Load dietary recommendations
diet_df = pd.read_csv("Final_diet.csv")

# Sample input
sample = {
    'Age': 30,
    'Gender': 'Female',
    'Height_cm': 171.4,
    'Weight_kg': 75.4,
    'BMI': 27.73,
    'Heart_Rate': 92,
    'Blood_Sugar': 121,
    'Cholesterol': 183,
    'Oxygen_Saturation': 96,
    'Body_Temperature_C': 37.3,
    'Sleep_Hours': 8.9,
    'Preexisting_Conditions': 'None',
    'Family_History': 'No',
    'Recent_Illnesses': 'Cold',
    'Physical_Activity': 'Sedentary',
    'Diet_Type': 'Vegetarian',
    'Alcohol_Consumption': 'No',
    'Smoking_Status': 'Yes',
    'Stress_Level': 4,
    'Fatigue': 0,
    'Chest_Pain': 0,
    'Shortness_of_Breath': 0,
    'Dizziness': 0,
    'Persistent_Cough': 0,
    'Systolic_BP': 120,
    'Diastolic_BP': 80
}

# Prepare dataframe
input_df = pd.DataFrame([sample])

# Encode categorical features
for col, le in feature_encoders.items():
    val = input_df.at[0, col]
    if val not in le.classes_:
        print(f"\n❌ ERROR: Column '{col}' — value '{val}' not in encoder classes:")
        print(f"   Known classes: {le.classes_}")
        raise ValueError(f"Unseen label '{val}' in column '{col}'")
    input_df[col] = le.transform([val])

# Scale numerical features
num_cols = input_df.select_dtypes(include=["int64", "float64"]).columns
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict
proba = model.predict_proba(input_df)[0]
top3_idx = np.argsort(proba)[-3:][::-1]
top3_labels = label_encoder.inverse_transform(top3_idx)
top3_probs = proba[top3_idx]

# Display results
print("\n=== Top 3 Predicted Issues ===")
for i in range(3):
    print(f"{i+1}. {top3_labels[i]} ({top3_probs[i]*100:.2f}%)")

# Show dietary recommendation
top_disease = top3_labels[0]
diet_row = diet_df[diet_df['Disease'] == top_disease]

if not diet_row.empty:
    row = diet_row.iloc[0]
    print(f"\n=== Dietary Recommendation for: {top_disease} ===")
    print(f"\n✅ Foods to Eat:\n- " + "\n- ".join(row['Eat'].split(', ')))
    print(f"\n❌ Foods to Avoid:\n- " + "\n- ".join(row['Avoid'].split(', ')))
    print(f"\n🔬 Macronutrient Breakdown:")
    print(f"- Calories: {row['Calories']} kcal")
    print(f"- Protein: {row['Protein_g']} g")
    print(f"- Carbs:   {row['Carbs_g']} g")
    print(f"- Fats:    {row['Fats_g']} g")
else:
    print(f"\n(No dietary data found for {top_disease})")
