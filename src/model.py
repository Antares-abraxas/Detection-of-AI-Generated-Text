"""
model.py — Классификационная модель детектора AI-текста.

Реализует интерпретируемый гибридный метод на основе Random Forest,
обученного на лингвистическом векторе из 15 дескрипторов (см. features.py).

Ключевые решения:
  - Random Forest: интерпретируем через feature importance и SHAP,
    не требует масштабирования, устойчив к выбросам [André et al., 2023].
  - StandardScaler: нормализация для совместимости с будущими baseline-моделями.
  - Пороговые значения в predict_with_explanation основаны на теоретическом
    анализе гипотез и ожидаемых диапазонах метрик.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Пороговые значения для интерпретируемых объяснений
# (обоснованы в Главе 2 ВКР, раздел 2.1)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    # Гипотеза 1: синтаксическая шаблонность
    "cr_pos_high":            0.62,   # CR-POS > порога → шаблонность
    "unique_trigrams_low":    0.25,   # уникальных триграмм < порога → шаблонность
    "pos_entropy_low":        2.0,    # энтропия POS < порога → монотонность

    # Гипотеза 2: пунктуационная однородность
    "burstiness_low":         4.0,    # std длин предложений < порога → монотонность
    "punct_entropy_low":      1.5,    # энтропия пунктуации < порога → предсказуемость
    "comma_cv_low":           0.3,    # CV запятых < порога → механический ритм

    # Гипотеза 3: стилистическая «гладкость»
    "ttr_low":                0.55,   # TTR < порога → лексическое однообразие
    "hlr_low":                0.35,   # HLR < порога → мало уникальных слов
    "complexity_low":         0.04,   # мало причастий/деепричастий → упрощённость

    # Гипотеза 4: семантико-структурное несоответствие
    "dep_depth_low":          3.5,    # глубина дерева < порога → плоские конструкции
    "nv_ratio_high":          3.0,    # много существительных vs. глаголов
}

# Описания причин для пользовательского объяснения
_REASON_TEMPLATES = {
    "cr_pos_high": (
        "[Г1] Высокий CR-POS ({val:.3f} > {thr}): повышенная повторяемость "
        "POS-последовательностей — признак синтаксической шаблонности."
    ),
    "unique_trigrams_low": (
        "[Г1] Низкое разнообразие POS-триграмм ({val:.3f} < {thr}): "
        "текст использует ограниченный набор синтаксических конструкций."
    ),
    "pos_entropy_low": (
        "[Г1] Низкая энтропия POS-распределения ({val:.3f} < {thr}): "
        "монотонное распределение частей речи."
    ),
    "burstiness_low": (
        "[Г2] Пониженный Burstiness ({val:.3f} < {thr}): "
        "однородная длина предложений — отсутствие естественного ритма."
    ),
    "punct_entropy_low": (
        "[Г2] Низкая энтропия пунктуации ({val:.3f} < {thr}): "
        "предсказуемые и монотонные паттерны знаков препинания."
    ),
    "comma_cv_low": (
        "[Г2] Низкий CV запятых ({val:.3f} < {thr}): "
        "равномерные интервалы между запятыми — механическое пунктуирование."
    ),
    "ttr_low": (
        "[Г3] Низкий TTR ({val:.3f} < {thr}): "
        "ограниченное лексическое разнообразие."
    ),
    "hlr_low": (
        "[Г3] Низкий HLR ({val:.3f} < {thr}): "
        "мало уникальных слов (hapax legomena) — бедный словарный запас."
    ),
    "complexity_low": (
        "[Г3] Мало причастий/деепричастий ({val:.3f} < {thr}): "
        "упрощённые синтаксические конструкции."
    ),
    "dep_depth_low": (
        "[Г4] Малая глубина дерева зависимостей ({val:.3f} < {thr}): "
        "плоские синтаксические структуры при возможной поверхностной связности."
    ),
    "nv_ratio_high": (
        "[Г4] Высокий Noun/Verb Ratio ({val:.3f} > {thr}): "
        "номинальный стиль — преобладание именных структур над глагольными."
    ),
}


class AICoreDetector:
    """Детектор AI-генерированного текста на основе Random Forest.

    Обучается на матрице лингвистических признаков X ∈ R^(n, 15) и метках
    y ∈ {0=Human, 1=AI}. Реализует интерпретируемое предсказание с
    привязкой каждого сигнала к конкретной гипотезе из Главы 2 ВКР.
    """

    def __init__(self) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",   # устойчивость к дисбалансу классов
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = FEATURE_NAMES
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обучает классификатор.

        Args:
            X: Матрица признаков формы (n_samples, 15).
            y: Вектор меток (0=Human, 1=AI).
        """
        self.scaler.fit(X)
        self.clf.fit(X, y)   # RF не требует масштабирования, scaler для baseline
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Предсказание и интерпретация
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности классов для матрицы X."""
        self._check_fitted()
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает бинарные метки для матрицы X."""
        self._check_fitted()
        return self.clf.predict(X)

    def predict_with_explanation(
        self,
        text: str,
        features: list | np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """Интерпретируемое предсказание для одного текста.

        Возвращает вердикт, вероятность и список причин, привязанных
        к конкретным гипотезам из Главы 2 ВКР.

        Args:
            text: Исходный текст (для отображения в UI/отчёте).
            features: Вектор признаков длиной 15.
            threshold: Порог для класса AI (по умолчанию 0.5).

        Returns:
            Словарь с полями: verdict, probability, reasons, feature_values,
            hypothesis_signals.
        """
        self._check_fitted()
        features = np.array(features, dtype=float)
        prob_ai = float(self.clf.predict_proba([features])[0][1])
        verdict = "AI-Generated" if prob_ai >= threshold else "Human-Written"

        feat_dict = dict(zip(self.feature_names, features))
        reasons = self._build_reasons(feat_dict)

        # Сводка по гипотезам
        hyp_signals = {
            "Г1 (синтаксическая шаблонность)": sum(
                1 for k in ("cr_pos_high", "unique_trigrams_low", "pos_entropy_low")
                if k in [r.split("]")[0][1:] + "]" for r in reasons]  # упрощённо
            ),
        }

        return {
            "verdict": verdict,
            "probability": round(prob_ai, 4),
            "confidence_level": _confidence_label(prob_ai),
            "reasons": reasons if reasons else [
                "Текст демонстрирует естественные лингвистические флуктуации; "
                "явные маркеры AI-генерации не обнаружены."
            ],
            "feature_values": {k: round(float(v), 5) for k, v in feat_dict.items()},
            "n_triggered_signals": len(reasons),
        }

    def get_feature_importances(self) -> dict:
        """Возвращает словарь {имя_признака: важность} из RF."""
        self._check_fitted()
        return dict(zip(self.feature_names, self.clf.feature_importances_))

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save(self, path: str = "data/syntactic_detector.pkl") -> None:
        """Сохраняет весь объект детектора (clf + scaler + метаданные)."""
        self._check_fitted()
        joblib.dump(self, path)
        print(f"[AICoreDetector] Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str = "data/syntactic_detector.pkl") -> "AICoreDetector":
        """Загружает детектор из файла."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Файл {path} не содержит объект AICoreDetector.")
        return obj

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Детектор не обучен. Вызовите метод train() перед использованием."
            )

    def _build_reasons(self, feat: dict) -> list[str]:
        """Формирует список текстовых объяснений на основе пороговых значений."""
        reasons = []

        # --- Гипотеза 1 ---
        if feat["cr_pos"] > THRESHOLDS["cr_pos_high"]:
            reasons.append(_REASON_TEMPLATES["cr_pos_high"].format(
                val=feat["cr_pos"], thr=THRESHOLDS["cr_pos_high"]))

        if feat["unique_trigrams_per_tok"] < THRESHOLDS["unique_trigrams_low"]:
            reasons.append(_REASON_TEMPLATES["unique_trigrams_low"].format(
                val=feat["unique_trigrams_per_tok"], thr=THRESHOLDS["unique_trigrams_low"]))

        if feat["pos_entropy"] < THRESHOLDS["pos_entropy_low"]:
            reasons.append(_REASON_TEMPLATES["pos_entropy_low"].format(
                val=feat["pos_entropy"], thr=THRESHOLDS["pos_entropy_low"]))

        # --- Гипотеза 2 ---
        if feat["burstiness"] < THRESHOLDS["burstiness_low"]:
            reasons.append(_REASON_TEMPLATES["burstiness_low"].format(
                val=feat["burstiness"], thr=THRESHOLDS["burstiness_low"]))

        if feat["punct_entropy"] < THRESHOLDS["punct_entropy_low"]:
            reasons.append(_REASON_TEMPLATES["punct_entropy_low"].format(
                val=feat["punct_entropy"], thr=THRESHOLDS["punct_entropy_low"]))

        if feat["comma_cv"] < THRESHOLDS["comma_cv_low"]:
            reasons.append(_REASON_TEMPLATES["comma_cv_low"].format(
                val=feat["comma_cv"], thr=THRESHOLDS["comma_cv_low"]))

        # --- Гипотеза 3 ---
        if feat["ttr"] < THRESHOLDS["ttr_low"]:
            reasons.append(_REASON_TEMPLATES["ttr_low"].format(
                val=feat["ttr"], thr=THRESHOLDS["ttr_low"]))

        if feat["hlr"] < THRESHOLDS["hlr_low"]:
            reasons.append(_REASON_TEMPLATES["hlr_low"].format(
                val=feat["hlr"], thr=THRESHOLDS["hlr_low"]))

        if feat["complexity_ratio"] < THRESHOLDS["complexity_low"]:
            reasons.append(_REASON_TEMPLATES["complexity_low"].format(
                val=feat["complexity_ratio"], thr=THRESHOLDS["complexity_low"]))

        # --- Гипотеза 4 ---
        if feat["avg_dep_depth"] < THRESHOLDS["dep_depth_low"]:
            reasons.append(_REASON_TEMPLATES["dep_depth_low"].format(
                val=feat["avg_dep_depth"], thr=THRESHOLDS["dep_depth_low"]))

        if feat["nv_ratio"] > THRESHOLDS["nv_ratio_high"]:
            reasons.append(_REASON_TEMPLATES["nv_ratio_high"].format(
                val=feat["nv_ratio"], thr=THRESHOLDS["nv_ratio_high"]))

        return reasons


def _confidence_label(prob: float) -> str:
    """Возвращает текстовую метку уверенности модели."""
    if prob > 0.85 or prob < 0.15:
        return "Высокая"
    if prob > 0.70 or prob < 0.30:
        return "Средняя"
    return "Низкая (пограничный случай)"