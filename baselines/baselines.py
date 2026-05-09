"""
Базовые методы детекции AI-текста для сравнения с основным синтаксическим методом.

Два семейства методов (соответствуют Главе 3 ВКР):
  A. Лексические (TF-IDF): используют содержание слов → чувствительны к домену
  B. Поверхностно-стилометрические: используют символьные статистики → быстрые,
     но менее точные, чем синтаксический подход

Каждый метод реализован как класс с унифицированным интерфейсом:
  .fit(X_text, y)
  .predict(X_text) → np.ndarray
  .predict_proba(X_text) → np.ndarray[:, 2]
  .get_metrics(X_text, y_true) → dict
  .save(path) / .load(path)

Это позволяет использовать все методы взаимозаменяемо в экспериментах.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Стоп-слова: доменные маркеры новостного корпуса, которые создают bias
# (явно упомянуты в исходных baseline-ноутбуках как источник утечки данных)
DOMAIN_STOP_WORDS = [
    'риа', 'новости', 'сообщает', 'говорится', 'ранее', 'рф', 'года', '2019',
    'сообщил', 'отметил', 'добавил', 'уточнил', 'подчеркнул',
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, name: str = '') -> dict:
    """Вычисляет стандартный набор метрик классификации."""
    return {
        'name':      name,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 5),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 5),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 5),
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 5),
        'AUC-ROC':   round(roc_auc_score(y_true, y_prob), 5),
        'AP':        round(average_precision_score(y_true, y_prob), 5),
    }


# ---------------------------------------------------------------------------
# A1. TF-IDF + Random Forest
# ---------------------------------------------------------------------------

class TfidfRFDetector:
    """Лексический baseline: TF-IDF (1-2 граммы) + Random Forest.

    Сильная сторона: высокое качество на обучающем домене.
    Слабая сторона: TF-IDF кодирует конкретные слова → при смене домена
    или языка качество резко падает (ожидаемо показывает в эксперименте
    с датасетами AINL-Eval-2025 и H3-WikiCSAI).
    """

    NAME = 'TF-IDF + Random Forest'
    EXP_ID = 'tfidf_rf'

    def __init__(self, max_features: int = 3000, ngram_range=(1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            ngram_range=ngram_range,
            stop_words=DOMAIN_STOP_WORDS,
            sublinear_tf=True,          # log(1+tf) — уменьшает влияние частых слов
        )
        self.clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X_text, y) -> 'TfidfRFDetector':
        X = self.vectorizer.fit_transform(X_text)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X_text) -> np.ndarray:
        return self.clf.predict(self.vectorizer.transform(X_text))

    def predict_proba(self, X_text) -> np.ndarray:
        return self.clf.predict_proba(self.vectorizer.transform(X_text))

    def get_metrics(self, X_text, y_true) -> dict:
        y_pred = self.predict(X_text)
        y_prob = self.predict_proba(X_text)[:, 1]
        return compute_metrics(y_true, y_pred, y_prob, self.NAME)

    def get_top_features(self, n: int = 20) -> dict:
        """Возвращает топ-N признаков по важности (интерпретируемость)."""
        names = self.vectorizer.get_feature_names_out()
        imp   = self.clf.feature_importances_
        idx   = np.argsort(imp)[-n:][::-1]
        return {names[i]: float(imp[i]) for i in idx}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'TfidfRFDetector':
        return joblib.load(path)


# ---------------------------------------------------------------------------
# A2. TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

class TfidfLogRegDetector:
    """Лексический baseline: TF-IDF + Logistic Regression.

    LogReg с L2-регуляризацией предоставляет линейно интерпретируемые веса:
    положительные веса → признаки, указывающие на AI-текст.
    Также чувствителен к смене домена как и TfidfRF.
    """

    NAME = 'TF-IDF + Logistic Regression'
    EXP_ID = 'tfidf_logreg'

    def __init__(self, max_features: int = 3000, C: float = 1.0) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            ngram_range=(1, 2),
            stop_words=DOMAIN_STOP_WORDS,
            sublinear_tf=True,
        )
        self.clf = LogisticRegression(
            C=C,
            max_iter=2000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
        )
        self._fitted = False

    def fit(self, X_text, y) -> 'TfidfLogRegDetector':
        X = self.vectorizer.fit_transform(X_text)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X_text) -> np.ndarray:
        return self.clf.predict(self.vectorizer.transform(X_text))

    def predict_proba(self, X_text) -> np.ndarray:
        return self.clf.predict_proba(self.vectorizer.transform(X_text))

    def get_metrics(self, X_text, y_true) -> dict:
        y_pred = self.predict(X_text)
        y_prob = self.predict_proba(X_text)[:, 1]
        return compute_metrics(y_true, y_pred, y_prob, self.NAME)

    def get_top_features(self, n: int = 10) -> dict:
        """Возвращает топ-N токенов с наибольшим положительным весом (→ AI)."""
        names   = self.vectorizer.get_feature_names_out()
        weights = self.clf.coef_[0]
        idx_ai  = np.argsort(weights)[-n:][::-1]
        return {names[i]: float(weights[i]) for i in idx_ai}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'TfidfLogRegDetector':
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Функции поверхностной стилометрии (общие для B1 и B2)
# ---------------------------------------------------------------------------

def extract_surface_stylometric_features(text: str) -> list:
    """Извлекает 10 поверхностных (символьных) стилометрических признаков.

    В отличие от оригинальных baseline-ноутбуков (7 признаков), здесь
    добавлены 3 признака для лучшего соответствия гипотезам Главы 2:
      - exclamation_density  (Г2: пунктуационная вариативность)
      - digit_ratio          (стилистический маркер)
      - unique_word_ratio    (Г3: приближение к TTR без NLP-парсинга)

    Признаки вычисляются без NLP-библиотек → язык-независимы.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return [0.0] * 10

    char_count = len(text)
    words      = text.split()
    word_count = len(words) or 1
    sentences  = re.findall(r'[.!?]+', text)
    sent_count = max(len(sentences), 1)

    # --- Пунктуация (Г2) ---
    comma_density  = text.count(',')  / char_count * 1000
    dash_density   = (text.count('-') + text.count('—')) / char_count * 1000
    quote_density  = (text.count('"') + text.count('«') + text.count('»')) / char_count * 1000
    excl_density   = text.count('!') / char_count * 1000
    punct_intensity = sum(1 for c in text if c in '.,!?;:-—()') / char_count * 1000

    # --- Лексика/структура (Г3, Г4) ---
    avg_word_len   = sum(len(w) for w in words) / word_count
    avg_sent_len   = word_count / sent_count
    caps_ratio     = sum(1 for c in text if c.isupper()) / char_count
    digit_ratio    = sum(1 for c in text if c.isdigit()) / char_count
    unique_word_ratio = len(set(w.lower() for w in words)) / word_count  # ~TTR

    return [
        comma_density,       # 0
        dash_density,        # 1
        quote_density,       # 2
        excl_density,        # 3
        punct_intensity,     # 4
        avg_word_len,        # 5
        avg_sent_len,        # 6
        caps_ratio,          # 7
        digit_ratio,         # 8
        unique_word_ratio,   # 9
    ]


SURFACE_FEATURE_NAMES = [
    'Comma_Density', 'Dash_Density', 'Quote_Density', 'Excl_Density',
    'Punct_Intensity', 'Avg_Word_Len', 'Avg_Sent_Len',
    'Caps_Ratio', 'Digit_Ratio', 'Unique_Word_Ratio',
]


# ---------------------------------------------------------------------------
# B1. Surface Stylometry + Random Forest
# ---------------------------------------------------------------------------

class StyloRFDetector:
    """Поверхностно-стилометрический baseline: 10 символьных признаков + RF.

    Не требует NLP-парсинга → работает на любом языке.
    Гипотеза: поверхностные признаки менее информативны, чем синтаксические
    (ожидаемо уступает основному методу по AUC-ROC).
    """

    NAME = 'Stylometry + Random Forest'
    EXP_ID = 'stylo_rf'

    def __init__(self) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names = SURFACE_FEATURE_NAMES
        self._fitted = False

    def _transform(self, X_text) -> np.ndarray:
        return np.array(
            [extract_surface_stylometric_features(t) for t in X_text],
            dtype=np.float32,
        )

    def fit(self, X_text, y) -> 'StyloRFDetector':
        self.clf.fit(self._transform(X_text), y)
        self._fitted = True
        return self

    def predict(self, X_text) -> np.ndarray:
        return self.clf.predict(self._transform(X_text))

    def predict_proba(self, X_text) -> np.ndarray:
        return self.clf.predict_proba(self._transform(X_text))

    def get_metrics(self, X_text, y_true) -> dict:
        y_pred = self.predict(X_text)
        y_prob = self.predict_proba(X_text)[:, 1]
        return compute_metrics(y_true, y_pred, y_prob, self.NAME)

    def get_feature_importances(self) -> dict:
        return dict(zip(self.feature_names, self.clf.feature_importances_))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'StyloRFDetector':
        return joblib.load(path)


# ---------------------------------------------------------------------------
# B2. Surface Stylometry + Logistic Regression
# ---------------------------------------------------------------------------

class StyloLogRegDetector:
    """Поверхностно-стилометрический baseline: 10 символьных признаков + LogReg.

    StandardScaler обязателен для LogReg (признаки разных масштабов).
    Линейные коэффициенты показывают направление влияния каждого признака.
    """

    NAME = 'Stylometry + Logistic Regression'
    EXP_ID = 'stylo_logreg'

    def __init__(self, C: float = 1.0) -> None:
        self.scaler = StandardScaler()
        self.clf    = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
        )
        self.feature_names = SURFACE_FEATURE_NAMES
        self._fitted = False

    def _transform_raw(self, X_text) -> np.ndarray:
        return np.array(
            [extract_surface_stylometric_features(t) for t in X_text],
            dtype=np.float32,
        )

    def fit(self, X_text, y) -> 'StyloLogRegDetector':
        X_raw = self._transform_raw(X_text)
        X = self.scaler.fit_transform(X_raw)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X_text) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(self._transform_raw(X_text)))

    def predict_proba(self, X_text) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(self._transform_raw(X_text)))

    def get_metrics(self, X_text, y_true) -> dict:
        y_pred = self.predict(X_text)
        y_prob = self.predict_proba(X_text)[:, 1]
        return compute_metrics(y_true, y_pred, y_prob, self.NAME)

    def get_coefficients(self) -> dict:
        return dict(zip(self.feature_names, self.clf.coef_[0]))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'StyloLogRegDetector':
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Реестр всех методов (для итерации в экспериментах)
# ---------------------------------------------------------------------------

ALL_BASELINES = [
    TfidfRFDetector,
    TfidfLogRegDetector,
    StyloRFDetector,
    StyloLogRegDetector,
]