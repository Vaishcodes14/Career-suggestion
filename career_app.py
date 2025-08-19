# career_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# --------------------------
# Load dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("career_suggestion_dataset.csv")
    return df

df = load_data()
st.title("🎓 Career Suggestion System")
st.write("This app suggests careers based on **Age + Interest + Aptitude** using Decision Trees & Logistic Regression.")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------
# Encode categorical data
# --------------------------
le_interest = LabelEncoder()
le_aptitude = LabelEncoder()
le_career = LabelEncoder()

df["Interest_enc"] = le_interest.fit_transform(df["Interest"])
df["Aptitude_enc"] = le_aptitude.fit_transform(df["Aptitude"])
df["Career_enc"] = le_career.fit_transform(df["Career"])

X = df[["Age", "Interest_enc", "Aptitude_enc"]]
y = df["Career_enc"]

# --------------------------
# Train models
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# --------------------------
# User Input
# --------------------------
st.sidebar.header("🔍 Input Your Details")
age = st.sidebar.slider("Select Age", 15, 30, 20)
interest = st.sidebar.selectbox("Select Interest", le_interest.classes_)
aptitude = st.sidebar.selectbox("Select Aptitude", le_aptitude.classes_)

# Encode input
user_data = [[
    age,
    le_interest.transform([interest])[0],
    le_aptitude.transform([aptitude])[0]
]]

# --------------------------
# Predictions
# --------------------------
dt_prediction = le_career.inverse_transform(dt_model.predict(user_data))[0]
lr_prediction = le_career.inverse_transform(lr_model.predict(user_data))[0]

st.subheader("💡 Career Suggestions")
st.write(f"**Decision Tree Suggestion:** {dt_prediction}")
st.write(f"**Logistic Regression Suggestion:** {lr_prediction}")

# --------------------------
# Visualize Decision Tree
# --------------------------
st.subheader("🌳 Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(dt_model, 
          feature_names=["Age", "Interest", "Aptitude"],
          class_names=le_career.classes_,
          filled=True, fontsize=8)
st.pyplot(fig)
