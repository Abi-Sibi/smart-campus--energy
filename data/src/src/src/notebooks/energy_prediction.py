from src.preprocess import load_data, preprocess
from src.train_model import train_linear, train_random_forest
from src.evaluate import evaluate

# Load data
data = load_data("../data/energy_data.csv")

# Preprocess
X_train, X_test, y_train, y_test = preprocess(data)

# Train models
lr_model = train_linear(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate
print("Linear Regression:", evaluate(lr_model, X_test, y_test))
print("Random Forest:", evaluate(rf_model, X_test, y_test))
