
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Simulierte Transaktionsdaten erstellen
def generate_transaction_data(n=10000):
    np.random.seed(42)
    data = pd.DataFrame({
        'amount': np.random.uniform(1, 10000, n),  # Transaktionsbetrag
        'time': np.random.randint(0, 86400, n),  # Zeitpunkt in Sekunden des Tages
        'location': np.random.choice(['DE', 'US', 'UK', 'FR', 'IN'], n),  # Land
        'card_present': np.random.choice([0, 1], n),  # 1 = Physische Karte, 0 = Online
        'is_fraud': np.random.choice([0, 1], n, p=[0.98, 0.02])  # 2% Betrug
    })
    return data

# Daten generieren
data = generate_transaction_data()

# 2️⃣ Datenvorverarbeitung
# Kategorische Variablen in numerische Werte umwandeln
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Features & Labels definieren
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Modelltraining mit Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ Modellbewertung
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# 5️⃣ Visualisierung der Feature-Wichtigkeit
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance")
plt.show()

# 6️⃣ Beispiel-Vorhersage
def predict_fraud(amount, time, location, card_present):
    location_cols = ['location_DE', 'location_FR', 'location_IN', 'location_UK', 'location_US']
    location_data = {col: 0 for col in location_cols}
    if f'location_{location}' in location_data:
        location_data[f'location_{location}'] = 1
    
    input_data = pd.DataFrame([{**{'amount': amount, 'time': time, 'card_present': card_present}, **location_data}])
    prediction = model.predict(input_data)[0]
    return "Fraudulent" if prediction == 1 else "Legitimate"

# Test-Vorhersage
print(predict_fraud(5000, 30000, 'US', 0))
