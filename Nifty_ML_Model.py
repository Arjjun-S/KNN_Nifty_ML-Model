import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("NFLX_minor.csv")
print(df.head())
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
targets = []
for i in range(len(df) - 1):
    if df.loc[i + 1, 'Close'] > df.loc[i, 'Close']:
        targets.append(1)
    else:
        targets.append(0)

df = df.iloc[:-1]
df['Target'] = targets

X = df.drop(columns=['Date', 'Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {acc * 100:.2f}%")
