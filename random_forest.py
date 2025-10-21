import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

with open("soccerNet/annotations_train.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)


df['label'] = df['answer'].apply(lambda x: 1 if 'yes' in x.lower() else 0)

df['number_of_games'] = pd.to_numeric(df['number_of_games'], errors='coerce').fillna(0).astype(int)

df['league'] = df['league'].fillna('unknown')
le = LabelEncoder()
df['league_encoded'] = le.fit_transform(df['league'])

X = df[['answer', 'number_of_games', 'league_encoded']]
y = df['label']

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=1000), "answer"),
        ("num", StandardScaler(), ["number_of_games"]),
        ("cat", "passthrough", ["league_encoded"]),
    ]
)

pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=100, random_state=42)
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Report：")
print(classification_report(y_test, y_pred))
