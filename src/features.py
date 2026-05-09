"""
features.py — Модуль извлечения лингвистических признаков для детекции AI-текста.

Реализует набор синтаксических и пунктуационных дескрипторов согласно:
  - Гипотезе №1 (синтаксическая шаблонность): CR-POS, уникальные POS-шаблоны [Shaib et al., 2024]
  - Гипотезе №2 (пунктуационная однородность): энтропия пунктуации, burstiness, CV запятых [Rujeedawa et al., 2025; Park et al., 2024]
  - Гипотезе №3 (стилистическая «гладкость»): TTR, доля причастий/деепричастий [André et al., 2023]
  - Гипотезе №4 (семантико-структурное несоответствие): глубина дерева зависимостей [Zamaraeva et al., 2025]
"""

import math
import zlib
from collections import Counter

import numpy as np
import spacy

# Загружаем модель один раз на уровне модуля для производительности
try:
    nlp = spacy.load("ru_core_news_lg")
except OSError:
    raise OSError(
        "Модель spaCy 'ru_core_news_lg' не найдена. "
        "Установите её командой: python -m spacy download ru_core_news_lg"
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _shannon_entropy(labels: list) -> float:
    """Вычисляет энтропию Шеннона (бит) для последовательности меток.
    Используется для оценки предсказуемости пунктуации и POS-тегов.
    """
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _cr_pos(pos_sequence: list) -> float:
    """Коэффициент сжатия POS-последовательности (CR-POS).

    Применяет алгоритм gzip к строке POS-тегов. Высокое значение CR-POS
    (близкое к 1) соответствует высокой повторяемости — признаку AI-текста.
    Метрика введена в работе Shaib et al. (2024) [источник 14 в статье].
    """
    if not pos_sequence:
        return 0.0
    encoded = " ".join(pos_sequence).encode("utf-8")
    compressed = zlib.compress(encoded, level=9)
    return len(compressed) / len(encoded)


def _unique_trigrams_per_token(pos_sequence: list) -> float:
    """Количество уникальных POS-триграмм на токен.

    Низкое значение означает, что текст использует ограниченный набор
    синтаксических конструкций — характерный признак шаблонного AI-письма.
    """
    if len(pos_sequence) < 3:
        return 0.0
    trigrams = [
        (pos_sequence[i], pos_sequence[i + 1], pos_sequence[i + 2])
        for i in range(len(pos_sequence) - 2)
    ]
    return len(set(trigrams)) / len(pos_sequence)


def _mean_dep_tree_depth(doc) -> float:
    """Средняя глубина синтаксического дерева зависимостей по предложениям.

    Глубина — максимальное расстояние от корня до листа в предложении.
    Низкая глубина указывает на упрощённые синтаксические конструкции
    (Гипотеза №4, Zamaraeva et al., 2025).
    """
    depths = []
    for sent in doc.sents:
        sent_depths = [len(list(token.ancestors)) for token in sent]
        if sent_depths:
            depths.append(max(sent_depths))
    return float(np.mean(depths)) if depths else 0.0


def _comma_variation_coefficient(tokens: list) -> float:
    """Коэффициент вариации (CV) интервалов между запятыми.

    CV = std / mean. Низкое значение означает равномерное расположение
    запятых — признак механического, а не интонационного пунктуирования.
    Метрика соответствует анализу пунктуационного ритма из Park et al. (2024).
    """
    comma_positions = [i for i, t in enumerate(tokens) if t.text == ","]
    if len(comma_positions) < 2:
        return 0.0
    intervals = np.diff(comma_positions).astype(float)
    mean_iv = np.mean(intervals)
    return float(np.std(intervals) / mean_iv) if mean_iv > 1e-9 else 0.0


def _hapax_legomenon_rate(lemmas: list) -> float:
    """Hapax Legomenon Rate — доля слов, встречающихся ровно один раз.

    Высокий HLR характерен для богатого, разнообразного словаря
    (человеческий текст). Упоминается в Главе 2 ВКР в разделе метрик
    лексического разнообразия.
    """
    if not lemmas:
        return 0.0
    freq = Counter(lemmas)
    hapax_count = sum(1 for c in freq.values() if c == 1)
    return hapax_count / len(lemmas)


def _passive_voice_ratio(doc) -> float:
    """Доля пассивных конструкций относительно общего числа глагольных токенов.

    Вычисляется через наличие зависимостей 'nsubjpass' или морфологического
    признака Voice=Pass в spaCy (ru_core_news_lg поддерживает).
    Соответствует метрике «доля сложных синтаксических конструкций» из Главы 2.
    """
    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    if not verbs:
        return 0.0
    passive = [
        t for t in verbs
        if "Pass" in t.morph.get("Voice") or t.dep_ in ("nsubjpass", "auxpass")
    ]
    return len(passive) / len(verbs)


# ---------------------------------------------------------------------------
# Основная функция извлечения признаков
# ---------------------------------------------------------------------------

# Имена признаков в том же порядке, что и возвращаемый вектор
FEATURE_NAMES = [
    "avg_sent_len",          # F01: Средняя длина предложения (токены)
    "burstiness",            # F02: Burstiness — стд. откл. длины предложений
    "punct_entropy",         # F03: Энтропия типов знаков препинания
    "punct_density",         # F04: Плотность пунктуации
    "avg_dep_depth",         # F05: Средняя глубина дерева зависимостей
    "nv_ratio",              # F06: Noun/Verb Ratio — POS-дистрибуция
    "ttr",                   # F07: Type-Token Ratio (лексическое разнообразие)
    "hlr",                   # F08: Hapax Legomenon Rate
    "complexity_ratio",      # F09: Доля причастий и деепричастий
    "passive_ratio",         # F10: Доля пассивных конструкций
    "comma_cv",              # F11: CV интервалов между запятыми
    "cr_pos",                # F12: Коэффициент сжатия POS-последовательности
    "unique_trigrams_per_tok", # F13: Уникальные POS-триграммы / токен
    "pos_entropy",           # F14: Энтропия POS-распределения
    "total_tokens",          # F15: Общее кол-во токенов (контрольный параметр)
]

N_FEATURES = len(FEATURE_NAMES)
_ZERO_VECTOR = [0.0] * N_FEATURES


def get_syntactic_features(text: str) -> list:
    """Извлекает вектор из {N} лингвистических дескрипторов для заданного текста.

    Вектор покрывает все четыре гипотезы, сформулированные в Главе 2 ВКР:
      - Гипотеза 1 (синтаксическая шаблонность): F12, F13, F14
      - Гипотеза 2 (пунктуационная однородность): F03, F04, F11, F02
      - Гипотеза 3 (стилистическая «гладкость»): F07, F08, F09, F10
      - Гипотеза 4 (семантико-структурное несоответствие): F05, F06, F07

    Args:
        text: Исходный текст для анализа.

    Returns:
        Список из {N} вещественных чисел (вектор признаков).
    """.format(N=N_FEATURES)
    if not isinstance(text, str) or len(text.strip()) < 20:
        return _ZERO_VECTOR.copy()

    doc = nlp(text)
    sents = list(doc.sents)
    tokens = [t for t in doc if not t.is_space]

    if not tokens:
        return _ZERO_VECTOR.copy()

    # ---- Базовые коллекции -----------------------------------------------
    punct_tokens = [t.text for t in doc if t.is_punct]
    content_tokens = [t for t in tokens if not t.is_punct]
    lemmas = [t.lemma_.lower() for t in content_tokens]
    pos_tags = [t.pos_ for t in tokens if not t.is_punct]

    # ---- F01, F02: длина предложений (Burstiness) ------------------------
    sent_lens = [len([t for t in s if not t.is_space and not t.is_punct])
                 for s in sents]
    avg_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0
    burstiness = float(np.std(sent_lens)) if sent_lens else 0.0

    # ---- F03, F04: пунктуация -------------------------------------------
    punct_entropy = _shannon_entropy(punct_tokens)
    punct_density = len(punct_tokens) / len(tokens) if tokens else 0.0

    # ---- F05: глубина синтаксического дерева ----------------------------
    avg_dep_depth = _mean_dep_tree_depth(doc)

    # ---- F06: Noun/Verb Ratio -------------------------------------------
    pos_counts = Counter(pos_tags)
    nv_ratio = pos_counts.get("NOUN", 0) / (pos_counts.get("VERB", 0) + 1e-9)

    # ---- F07: TTR (Type-Token Ratio) ------------------------------------
    ttr = len(set(lemmas)) / len(lemmas) if lemmas else 0.0

    # ---- F08: Hapax Legomenon Rate -------------------------------------
    hlr = _hapax_legomenon_rate(lemmas)

    # ---- F09: доля причастий и деепричастий ----------------------------
    complex_forms = [
        t for t in tokens
        if "Conv" in t.morph.get("VerbForm") or "Part" in t.morph.get("VerbForm")
    ]
    complexity_ratio = len(complex_forms) / len(tokens) if tokens else 0.0

    # ---- F10: пассивный залог ------------------------------------------
    passive_ratio = _passive_voice_ratio(doc)

    # ---- F11: CV запятых -----------------------------------------------
    comma_cv = _comma_variation_coefficient(tokens)

    # ---- F12: CR-POS ---------------------------------------------------
    cr_pos = _cr_pos(pos_tags)

    # ---- F13: уникальные POS-триграммы / токен -------------------------
    unique_trigrams = _unique_trigrams_per_tok(pos_tags)

    # ---- F14: энтропия POS-распределения --------------------------------
    pos_entropy = _shannon_entropy(pos_tags)

    # ---- F15: общее кол-во токенов -------------------------------------
    total_tokens = float(len(tokens))

    return [
        avg_sent_len,       # F01
        burstiness,         # F02
        punct_entropy,      # F03
        punct_density,      # F04
        avg_dep_depth,      # F05
        nv_ratio,           # F06
        ttr,                # F07
        hlr,                # F08
        complexity_ratio,   # F09
        passive_ratio,      # F10
        comma_cv,           # F11
        cr_pos,             # F12
        unique_trigrams,    # F13
        pos_entropy,        # F14
        total_tokens,       # F15
    ]


def _unique_trigrams_per_tok(pos_sequence: list) -> float:
    """Алиас для использования внутри модуля."""
    return _unique_trigrams_per_token(pos_sequence)