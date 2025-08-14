import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("large_career_dataset.csv")  # Updated file name

df = load_data()

st.title("🎯 Career Suggestion Model")
st.write("Suggesting careers based on Age and Interests using Decision Tree")

# Encode categorical data
le_interest = LabelEncoder()
le_career = LabelEncoder()

df['Interest_encoded'] = le_interest.fit_transform(df['Interest'])
df['Career_encoded'] = le_career.fit_transform(df['Career'])

# Features & Target
X = df[['Age', 'Interest_encoded']]
y = df['Career_encoded']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# User input
st.subheader("Enter your details:")
age = st.number_input("Age", min_value=10, max_value=60, step=1)
interest = st.selectbox("Select Interest", le_interest.classes_)

if st.button("Suggest Career"):
    interest_code = le_interest.transform([interest])[0]
    prediction_code = model.predict([[age, interest_code]])[0]
    predicted_career = le_career.inverse_transform([prediction_code])[0]
    st.success(f"Recommended Career: **{predicted_career}**")

# Show decision tree
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(model, feature_names=['Age', 'Interest_encoded'], 
          class_names=le_career.classes_, filled=True, ax=ax)
st.pyplot(fig)

# Show dataset preview
if st.checkbox("Show dataset"):
    st.dataframe(df)
