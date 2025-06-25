import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Prepare dataset
penguins = pd.read_csv("penguins_cleaned.csv")

df = penguins.copy()
target = "species"
encode = ["sex", "island"]

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {
    "Adelie": 0,
    "Chinstrap": 1,
    "Gentoo": 2
}
def target_encode(val):
    return target_mapper[val]
df["species"] = df["species"].apply(target_encode)

# Separate dataset into X and Y for training
X = df.drop(["species"], axis=1)
y = df["species"]

# Build random forest model
classifier = RandomForestClassifier()
classifier.fit(X, y)

# Saving model
pickle.dump(classifier, open("penguins_classifier.pkl", "wb"))
