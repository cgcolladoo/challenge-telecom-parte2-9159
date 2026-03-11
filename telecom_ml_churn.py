import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=== TELECOM CHURN ML - TASK 91590 PARTE 2 ===")

# 1. ETL: LOAD + CLEAN
df = pd.read_csv('telecom_clientes.csv')  # Tu dataset
print(f"Shape inicial: {df.shape}")

# Limpieza
df = df.dropna()
df['Churn'] = df['Churn'].astype(int)
print(f"Shape limpio: {df.shape}")

# 2. FEATURE ENGINEERING
df['Valor_total'] = df['Valor'] + df['Impuesto'] - df['Descuento']
df['Riesgo_alto'] = ((df['Area'] > 80) | (df['Valor'] < 4000)).astype(int)

# 3. FEATURES TOP (correlación churn)
features = ['Suites', 'Area', 'Valor', 'Valor_total', 'Riesgo_alto']
X = df[features]
y = df['Churn']

# 4. SPLIT TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODELO RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. PREDICCIONES + EVALUACIÓN
y_pred = model.predict(X_test)
print("\n📊 REPORT ML:")
print(classification_report(y_test, y_pred))

# 7. FEATURE IMPORTANCE
importancias = pd.DataFrame({
    'feature': features,
    'importancia': model.feature_importances_
}).sort_values('importancia', ascending=False)

print("\n🔍 TOP FEATURES:")
print(importancias)

# 8. VISUALIZACIÓN
plt.figure(figsize=(10,6))
sns.barplot(data=importancias, x='importancia', y='feature')
plt.title('Importancia Features Churn')
plt.savefig('feature_importance.png')
plt.show()

# 9. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

print("\n✅ ML PIPELINE COMPLETA - PARTE 2")
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
