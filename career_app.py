import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("career_dataset.csv")

# Features & Target
X = df.drop("Career", axis=1)
y = df["Career"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Career Suggestion App")
st.write("Get career suggestions based on your interests and skills.")

user_data = {}
for col in X.columns:
    user_data[col] = st.slider(f"Rate your skill/interest in {col} (0-10)", 0, 10, 5)

if st.button("Suggest Career"):
    input_df = pd.DataFrame([user_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Suggested Career: {prediction}")

