from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully")