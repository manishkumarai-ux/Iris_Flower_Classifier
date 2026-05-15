import streamlit as st
import pickle
from sklearn.datasets import load_iris
from PIL import Image

# Load trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load iris dataset
iris = load_iris()

# App title
st.title("🌸 Iris Flower Classification App")

st.write("Enter flower measurements to predict flower type.")

# User inputs
sl = st.number_input("Enter Sepal Length (4cm-8cm):", min_value=0.0)
sw = st.number_input("Enter Sepal Width (2cm-5cm):", min_value=0.0)
pl = st.number_input("Enter Petal Length (0.10cm-2.50cm):", min_value=0.0)
pw = st.number_input("Enter Petal Width (0.1cm-2cm):", min_value=0.0)

# Predict button
if st.button("Predict Flower"):

    # Prepare input
    sample = [[sl, sw, pl, pw]]

    # Prediction
    prediction = model.predict(sample)

    # Flower name
    flower_name = iris.target_names[prediction[0]]

    # Show result
    st.success(f"Predicted Flower: {flower_name.upper()}")

    # Show probability
    probabilities = model.predict_proba(sample)

    st.subheader("Prediction Confidence")

    for i, name in enumerate(iris.target_names):
        st.write(f"{name}: {probabilities[0][i]*100:.2f}%")

    # Image path
    image_path = f"images/{flower_name}.jpg"

    # Display image
    image = Image.open(image_path)

    st.image(image, caption=flower_name.upper(), width=300)