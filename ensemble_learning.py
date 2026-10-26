import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv("student_data.csv")
print("First 5 records:")
print(data.head())

# 2. Separate input and output
X = data[
    [
        'Study_Hours',
        'Attendance',
        'Assignment_Score',
        'Previous_Score'
    ]
]
y = data['Result']

# 3. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. BAGGING - Random Forest
bagging_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
bagging_model.fit(X_train, y_train)
bagging_prediction = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(
    y_test,
    bagging_prediction
)

# 5. BOOSTING - AdaBoost
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
boosting_model.fit(X_train, y_train)
boosting_prediction = boosting_model.predict(X_test)
boosting_accuracy = accuracy_score(
    y_test,
    boosting_prediction
)

# 6. Display results
print("\n------------------------------")
print("ENSEMBLE LEARNING RESULTS")
print("------------------------------")
print(
    "Bagging Accuracy:",
    round(bagging_accuracy * 100, 2),
    "%"
)
print(
    "Boosting Accuracy:",
    round(boosting_accuracy * 100, 2),
    "%"
)

# 7. Compare Bagging and Boosting
methods = ['Bagging', 'Boosting']
accuracies = [
    bagging_accuracy * 100,
    boosting_accuracy * 100
]
plt.bar(methods, accuracies)
plt.xlabel("Ensemble Method")
plt.ylabel("Accuracy (%)")
plt.title("Bagging vs Boosting")
plt.ylim(0, 100)
plt.show()
