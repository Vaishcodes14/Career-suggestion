import streamlit as st
import pandas as pd
df = pd.read_csv("career.csv")  # CSV in same repo folder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# -----------------------------
# Title & Description
# -----------------------------
st.title("Career Suggestion Model 🎯")
st.write("""
This app suggests career paths based on your **Age** and **Interest**.
The model uses a Decision Tree trained on a dataset of career preferences.
""")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("large_career_dataset.csv")  # Change filename if needed
    return df

df = load_data()
st.write("### Dataset Preview", df.head())

# -----------------------------
# Encode categorical features
# -----------------------------
df_encoded = pd.get_dummies(df, columns=["Interest"])
X = df_encoded.drop(columns=["Career"])
y = df_encoded["Career"]

# -----------------------------
# Train the Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -----------------------------
# User Input Form
# -----------------------------
st.sidebar.header("Enter Your Details")
age = st.sidebar.slider("Age", int(df["Age"].min()), int(df["Age"].max()))

# For interest dropdown
interests_list = sorted(df["Interest"].unique())
interest = st.sidebar.selectbox("Interest", interests_list)

# Create input dataframe
user_input = pd.DataFrame([[age, interest]], columns=["Age", "Interest"])
user_input_encoded = pd.get_dummies(user_input)
user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Suggest Career"):
    prediction = model.predict(user_input_encoded)
    st.success(f"💡 Suggested Career: **{prediction[0]}**")

# -----------------------------
# Decision Tree Visualization
# -----------------------------
st.write("### Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
st.pyplot(fig)
