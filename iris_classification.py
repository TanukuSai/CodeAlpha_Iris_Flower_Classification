import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('Iris.csv')

print(df.head())
print(df.info())
print(df['Species'].value_counts())

X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', style='Species', s=70)
plt.title('Sepal Length vs Sepal Width')
plt.savefig('iris_sepal_plot.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', style='Species', s=70)
plt.title('Petal Length vs Petal Width')
plt.savefig('iris_petal_plot.png')
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_plot.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Species', y='PetalLengthCm', data=df, hue='Species', palette='Set2')
plt.title('Petal Length by Species')
plt.tight_layout()
plt.savefig('iris_boxplots.png')
plt.close()

print("Execution completed successfully.")
