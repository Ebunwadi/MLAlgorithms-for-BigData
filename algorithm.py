import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
data = pd.read_csv('daily_weather.csv')

# Data Visualization
sns.pairplot(data[['air_temp_9am', 'air_pressure_9am', 'avg_wind_speed_9am', 'relative_humidity_9am', 'relative_humidity_3pm']])
plt.suptitle('Pairwise Relationships between Features', y=1.02)
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['relative_humidity_3pm'], kde=True, bins=30)
plt.title('Distribution of Relative Humidity at 3 PM')
plt.xlabel('Relative Humidity at 3 PM (%)')
plt.ylabel('Frequency')
plt.show()

# Preprocessing
data.fillna(data.median(), inplace=True)
data['humidity_class'] = data['relative_humidity_3pm'].apply(lambda x: 1 if x < 25 else 0)
X_core = data[['air_temp_9am', 'air_pressure_9am', 'avg_wind_speed_9am', 'relative_humidity_9am']]
X_all = data.drop(['relative_humidity_3pm', 'humidity_class'], axis=1)
y = data['humidity_class']

# Scaling
scaler = MinMaxScaler()
X_core_scaled = scaler.fit_transform(X_core)
X_all_scaled = scaler.fit_transform(X_all)

# Train-test split
X_train_core, X_test_core, y_train, y_test = train_test_split(X_core_scaled, y, test_size=0.2, random_state=42)
X_train_all, X_test_all, y_train, y_test = train_test_split(X_all_scaled, y, test_size=0.2, random_state=42)

# KNN Model
knn_core_model = KNeighborsClassifier(n_neighbors=5)
knn_core_model.fit(X_train_core, y_train)
knn_core_preds = knn_core_model.predict(X_test_core)

knn_all_model = KNeighborsClassifier(n_neighbors=5)
knn_all_model.fit(X_train_all, y_train)
knn_all_preds = knn_all_model.predict(X_test_all)

# Decision Tree Model
dt_core_model = DecisionTreeClassifier()
dt_core_model.fit(X_train_core, y_train)
dt_core_preds = dt_core_model.predict(X_test_core)

dt_all_model = DecisionTreeClassifier()
dt_all_model.fit(X_train_all, y_train)
dt_all_preds = dt_all_model.predict(X_test_all)

# Naive Bayes Model
nb_core_model = GaussianNB()
nb_core_model.fit(X_train_core, y_train)
nb_core_preds = nb_core_model.predict(X_test_core)

nb_all_model = GaussianNB()
nb_all_model.fit(X_train_all, y_train)
nb_all_preds = nb_all_model.predict(X_test_all)

# Random Forest Model
rf_core_model = RandomForestClassifier(n_estimators=100)
rf_core_model.fit(X_train_core, y_train)
rf_core_preds = rf_core_model.predict(X_test_core)

rf_all_model = RandomForestClassifier(n_estimators=100)
rf_all_model.fit(X_train_all, y_train)
rf_all_preds = rf_all_model.predict(X_test_all)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the models and their predictions
models_core = {'KNN': knn_core_preds, 'Decision Tree': dt_core_preds, 'Naive Bayes': nb_core_preds, 'Random Forest': rf_core_preds}
models_all = {'KNN': knn_all_preds, 'Decision Tree': dt_all_preds, 'Naive Bayes': nb_all_preds, 'Random Forest': rf_all_preds}

# Core features results
results_core = {}
print("Results for Core Features:")
for name, preds in models_core.items():
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results_core[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    print(f"\n{name} - Core Features:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# All features results
results_all = {}
print("\nResults for All Features:")
for name, preds in models_all.items():
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results_all[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    print(f"\n{name} - All Features:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
# print confusion matrix for all features and core features
for name, preds in models_core.items():
    cm = confusion_matrix(y_test, preds)
    print(f'Confusion Matrix for {name} (Core Features):\n {cm}')

for name, preds in models_all.items():
    cm = confusion_matrix(y_test, preds)
    print(f'Confusion Matrix for {name} (All Features):\n {cm}')
