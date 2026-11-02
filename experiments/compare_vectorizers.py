"""Compare three TF-IDF feature variants on the weakly labelled corpus."""

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

NOTE_LEXICON = {
    "красные ягоды": ["красн.*ягод", "вишн", "клубник", "малин", "земляник", "брусник", "клюкв"],
    "черные ягоды": ["черн.*ягод", "смородин", "ежевик", "черник"],
    "цитрусы": ["цитрус", "лимон", "лайм", "грейпфрут", "апельсин", "мандарин"],
    "тропические фрукты": ["тропичес", "ананас", "манго", "маракуй", "банан"],
    "косточковые фрукты": ["косточков", "персик", "абрикос", "слив", "нектарин"],
    "яблоко и груша": ["яблок", "груш"],
    "цветы": ["цветоч", "цветущ", "жасмин", "фиалк", "роза", "боярышник"],
    "пряности": ["пряност", "специ", "перец", "кориц", "гвоздик"],
    "древесина": ["древес", "дуб", "кедр", "бочк"],
    "минералы": ["минерал", "кремни", "мел", "слан"],
    "сухофрукты": ["сухофрукт", "чернослив", "изюм", "кураг"],
    "ваниль": ["ванил"],
    "шоколад": ["шоколад", "какао"],
    "табак": ["табак"],
    "травы": ["трав", "папорот", "эвкалипт", "мят"],
    "мед": ["медов", "\\bмед\\b"],
}


def clean_text(value):
    value = "" if pd.isna(value) else str(value)
    value = value.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def section(text, name):
    match = re.search(rf"{name}\s*:\s*(.*?)(?=\s(?:цвет|аромат|вкус)\s*:|$)", text, re.I | re.S)
    return match.group(1) if match else text


def notes(text):
    return sorted(label for label, patterns in NOTE_LEXICON.items() if any(re.search(p, text, re.I) for p in patterns))


def make_labels(df):
    color = df.get("color", pd.Series("", index=df.index)).fillna("").map(clean_text)
    text = df["model_text"]
    df["acidity_level"] = [
        "высокая" if re.search(r"хрустящ.*кисл|ярк.*кисл|высок.*кисл|свеж.*кисл", t)
        else "низкая" if re.search(r"мягк.*кисл|низк.*кисл", t)
        else "высокая" if c in {"белое", "розовое"} else "средняя"
        for t, c in zip(text, color)
    ]
    df["tannin_level"] = [
        "высокая" if re.search(r"мощн.*танин|выраженн.*танин|терпк", t)
        else "средняя" if re.search(r"мягк.*танин|шелковист.*танин|округл.*танин", t) or c == "красное"
        else "низкая"
        for t, c in zip(text, color)
    ]
    df["body_level"] = [
        "полное" if re.search(r"плотн|насыщенн|богат|полнотел|мощн", t)
        else "легкое" if re.search(r"легк|воздуш|невесом", t) else "среднее" for t in text
    ]
    df["intensity_level"] = [
        "высокая" if re.search(r"интенсив|ярк|выразитель|насыщенн|богат", t)
        else "низкая" if re.search(r"деликат|нежн|тонк|легк", t) else "средняя" for t in text
    ]
    df["aroma_notes"] = text.map(lambda t: notes(section(t, "аромат")))
    df["taste_notes"] = text.map(lambda t: notes(section(t, "вкус")))
    return df


def evaluate_variant(name, vectorizer, df, train_idx, test_idx):
    started = time.perf_counter()
    x_train = vectorizer.fit_transform(df.loc[train_idx, "model_text"])
    x_test = vectorizer.transform(df.loc[test_idx, "model_text"])
    category_f1, category_accuracy = [], []
    for target in ["acidity_level", "tannin_level", "body_level", "intensity_level"]:
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(x_train, df.loc[train_idx, target])
        prediction = model.predict(x_test)
        category_f1.append(f1_score(df.loc[test_idx, target], prediction, average="macro", zero_division=0))
        category_accuracy.append(accuracy_score(df.loc[test_idx, target], prediction))

    row = {
        "variant": name,
        "features": x_train.shape[1],
        "categorical_macro_f1_mean": np.mean(category_f1),
        "categorical_accuracy_mean": np.mean(category_accuracy),
    }
    for target in ["aroma_notes", "taste_notes"]:
        counts = pd.Series([item for values in df[target] for item in values]).value_counts()
        labels = counts[(counts >= 2) & (counts <= len(df) - 2)].index.tolist()
        mlb = MultiLabelBinarizer(classes=labels)
        y = mlb.fit_transform(df[target].map(lambda xs: [x for x in xs if x in labels]))
        probabilities = np.zeros((len(test_idx), len(labels)))
        for column in range(len(labels)):
            model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
            model.fit(x_train, y[train_idx, column])
            probabilities[:, column] = model.predict_proba(x_test)[:, 1]
        prediction = (probabilities >= 0.35).astype(int)
        row[f"{target}_micro_f1"] = f1_score(y[test_idx], prediction, average="micro", zero_division=0)
        row[f"{target}_macro_f1"] = f1_score(y[test_idx], prediction, average="macro", zero_division=0)
    row["fit_eval_seconds"] = time.perf_counter() - started
    return row


def main():
    df = pd.read_csv(ROOT / "data" / "wines.csv")
    columns = [c for c in ["name", "description", "consumption_recommendation"] if c in df]
    df["model_text"] = df[columns].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)
    df = df[df["model_text"].str.len() >= 10].drop_duplicates("model_text").reset_index(drop=True)
    df = make_labels(df)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.25, random_state=RANDOM_STATE)

    variants = {
        "char_3_5": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30_000, sublinear_tf=True),
        "word_1_2": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True),
        "hybrid_char_word": FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15_000, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=15_000, sublinear_tf=True)),
        ]),
    }
    results = [evaluate_variant(name, vectorizer, df, train_idx, test_idx) for name, vectorizer in variants.items()]
    output = pd.DataFrame(results).sort_values("categorical_macro_f1_mean", ascending=False)
    output.to_csv(ROOT / "artifacts" / "experiment_comparison.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
