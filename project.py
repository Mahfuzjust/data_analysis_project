# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
path = "E:/Academic/data_analysis/Social_Network_Ads.csv"
try:
    data = pd.read_csv(path)
    print("The imported data is:\n", data.head())
except FileNotFoundError:
    print("The specified file path is invalid. Please check the path.")
    exit()

# Insert some blank data and incorrect data types
data.loc[0:2, 'Age'] = np.nan  # Insert some blank data
if 'Gender' not in data.columns:
    data['Gender'] = [25, "Male", "Female", 30]  # Insert incorrect data types

# Show the dataset with inserted anomalies
print("Dataset with inserted blank and incorrect data:\n", data.head())

# Select features and target
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# Handle missing data using SimpleImputer
imputer = SimpleImputer(strategy='mean')
x[:, 0:1] = imputer.fit_transform(x[:, 0:1])
print("Processed features after handling missing data:\n", x[:5])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Apply MinMax scaling
scale1 = MinMaxScaler()
x_train = scale1.fit_transform(x_train)
x_test = scale1.transform(x_test)

# Show scaled training and testing datasets
print("Scaled training data:\n", x_train[:5])
print("Scaled testing data:\n", x_test[:5])

# Train SVM model with linear kernel
cl1 = SVC(kernel='linear', random_state=0)
cl1.fit(x_train, y_train)

# Make predictions
y_predict = cl1.predict(x_test)

# Evaluate model performance
cm = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)

print("SVM Confusion Matrix:")
print(cm)
print(f"SVM Accuracy: {acc:.2f}")
print("SVM Classification Report:")
print(classification_report(y_test, y_predict))

# Train Random Forest model
cl2 = RandomForestClassifier(n_estimators=10, random_state=0)
cl2.fit(x_train, y_train)

# Make predictions
y_predict_rf = cl2.predict(x_test)

# Evaluate model performance
cm_rf = confusion_matrix(y_test, y_predict_rf)
acc_rf = accuracy_score(y_test, y_predict_rf)

print("Random Forest Confusion Matrix:")
print(cm_rf)
print(f"Random Forest Accuracy: {acc_rf:.2f}")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_predict_rf))

# Visualize SVM decision boundary
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                      np.arange(y_min, y_max, 0.01))
z = cl1.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, z, alpha=0.8, cmap='autumn')
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='winter', edgecolor='k')
plt.title('SVM Decision Boundary Visualization')
plt.show()