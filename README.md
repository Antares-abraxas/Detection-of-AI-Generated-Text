# Детектор AI-генерированного текста
### Обнаружение аномалий в синтаксических структурах и паттернах пунктуации

**Автор:** Тен В.Р.

---

## Структура проекта

```
Detection-of-AI-Generated-Text/
├── src/
│   ├── __init__.py
│   ├── features.py            ← Извлечение 15 лингвистических признаков (spaCy)
│   └── model.py               ← Классификатор Random Forest + интерпретация
│
├── baselines/
│   ├── baselines.py           ← 4 базовых метода (TF-IDF RF/LogReg, Stylometry RF/LogReg)
│   ├── 01_lexical_tfidf_rf.ipynb ← Блокноты для наглядности
│   ├── 02_lexical_logreg.ipynb
│   ├── 03_stylo_rf.ipynb
│   ├── 04_stylo_logreg.ipynb
│   ├── 05_final_report.ipynb
│   └── output/                ← CSV с результатами baseline экспериментов
│
├── data/
│   ├── ai_detection_ru_dataset_v4.csv   ← Основной датасет (10 000 записей, новости RU)
│   ├── ai_dataset_humanized.csv         ← Исходный корпус генераций (5 000 записей) после back-translation
│   ├── ai_dataset.csv                   ← Корпус генерация до back-translation
│   ├── RU_abstracts.csv                 ← Академический корпус (domain shift тест)
│   └── wiki_csai.jsonl                  ← H3 wiki_csai (language shift тест, EN)
│
├── cache/                     ← Кэши признаков 
│   ├── features_cache.npy
│   ├── feats_ainl_dev.npy
│   ├── feats_indomain.npy
│   ├── feats_wiki_csai.npy
│   └── labels_cache.npy
│
├── figures/                   ← Все графики экспериментов
│
├── generation/
│   ├── ai_text_generator.py   ← Генерация AI-текстов 
│   └── merge.py               ← Сборка итогового датасета
│
├── solution.ipynb             ← Основной ноутбук: обучение и оценка модели
├── robustness_eval.ipynb      ← Эксперимент: робастность к domain/language shift
├── requirements.txt
└── README.md
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
python -m spacy download ru_core_news_lg
```

### 2. Обучение модели

```bash
jupyter notebook solution.ipynb
```

### 3. Эксперимент по робастности

```bash
jupyter notebook robustness_eval.ipynb
```

---

## Датасеты

### Основной корпус (`ai_detection_ru_dataset_v4.csv`)
Сбалансированный набор для обучения и оценки классификатора.
- **Размер:** 10 000 записей (5 000 Human + 5 000 AI)
- **Домен:** новостные тексты на русском языке
- **Аугментация:** 20% AI-класса (1 000 записей) прошли back-translation `ru → en → ru` для проверки устойчивости к парафразу
- **Модели-генераторы:** Llama-3.1-8B, Qwen-2.5-7B, DeepSeek-R1
- **Поля:** `id`, `text`, `is_ai`, `is_redacted`, `source`, `topic`, `temperature`, `top_p`, `length_words`, `length_chars`

### Академический корпус (`RU_abstracts.csv`)
Используется двояко: часть идёт в **доменное обогащение** train, часть — в **domain shift тест**.
- **Домен:** научные аннотации на русском языке
- **Метки:** `label` ∈ {`human`, `gpt-4-turbo`, `llama-3.3-70b`, `gemma-2-27b`}
- **Разбивка:** 1 000 + 1 000 (test) — по индексам, без пересечений

### Language shift тест (`wiki_csai.jsonl`)
- **Источник:** H3 датасет, Wikipedia CSAI (английский)
- **Поля:** `question`, `human_answers`, `chatgpt_answers`

---

## Этапы `solution.ipynb`

| Этап | Содержание | Выход |
|------|-----------|-------|
| **0** | Загрузка библиотек, чтение CSV | `df`, настройки |
| **1** | Извлечение 15 признаков (spaCy) + кэш | `X`, `y`, `X_acad`, `y_acad` |
| **2** | Mann–Whitney U-test + поправка Бонферрони | Таблица значимости, violin-plots |
| **3** | Смешанный train (in-domain + academic), 5-fold CV, hold-out оценка | Accuracy, F1, AUC-ROC |
| **4** | MDI Feature Importance + SHAP | Графики вклада признаков |
| **5** | Интерпретируемое предсказание | Вердикт + причины по гипотезам |

> **Кэш признаков:** при первом запуске spaCy-парсинг занимает несколько минут.
> Результаты сохраняются в `cache/`. Повторные запуски мгновенны.

---

## Этапы `robustness_eval.ipynb`

| Сценарий | Датасет | Домен | Язык |
|----------|---------|-------|------|
| In-domain | `ai_detection_ru_dataset_v4.csv` (test 20%) | Новости | RU |
| Domain shift | `RU_abstracts.csv` | Академический | RU |
| Language shift | `wiki_csai.jsonl` | Энциклопедический | EN |

---

## Базовые методы (`baselines/baselines.py`)

| Метод | Класс | Описание |
|-------|-------|----------|
| TF-IDF + Random Forest | `TfidfRFDetector` | Лексический, чувствителен к смене домена |
| TF-IDF + Logistic Regression | `TfidfLogRegDetector` | Лексический, линейно интерпретируем |
| Stylometry + Random Forest | `StyloRFDetector` | 10 символьных признаков, язык-независим |
| Stylometry + Logistic Regression | `StyloLogRegDetector` | То же + StandardScaler |

---

## Описание признаков (15 дескрипторов)

| Код | Название | Гипотеза | Описание |
|-----|----------|----------|----------|
| F01 | `avg_sent_len` | Г2 | Средняя длина предложения в токенах |
| F02 | `burstiness` | **Г2** | Стандартное отклонение длин предложений |
| F03 | `punct_entropy` | **Г2** | Энтропия Шеннона типов знаков препинания |
| F04 | `punct_density` | Г2 | Доля знаков препинания от всех токенов |
| F05 | `avg_dep_depth` | **Г4** | Средняя глубина дерева синтаксических зависимостей |
| F06 | `nv_ratio` | Г4 | Noun/Verb Ratio |
| F07 | `ttr` | **Г3** | Type-Token Ratio — лексическое разнообразие |
| F08 | `hlr` | **Г3** | Hapax Legomenon Rate |
| F09 | `complexity_ratio` | Г3 | Доля причастий и деепричастий |
| F10 | `passive_ratio` | Г3 | Доля пассивных конструкций |
| F11 | `comma_cv` | **Г2** | Коэффициент вариации интервалов между запятыми |
| F12 | `cr_pos` | **Г1** | Коэффициент сжатия POS-последовательности (gzip) |
| F13 | `unique_trigrams_per_tok` | **Г1** | Уникальные POS-триграммы / токен |
| F14 | `pos_entropy` | **Г1** | Энтропия распределения частей речи |
| F15 | `total_tokens` | Ctrl | Общее число токенов |

---

## Четыре гипотезы (Глава 2 ВКР)

**Г1 — Синтаксическая шаблонность** *(F12, F13, F14)*  
AI-тексты повторяют ограниченный набор POS-конструкций → высокий CR-POS, мало уникальных триграмм, низкая энтропия POS.

**Г2 — Пунктуационная однородность** *(F02, F03, F04, F11)*  
AI-тексты имеют предсказуемые паттерны пунктуации → низкий burstiness, низкая энтропия пунктуации, равномерный ритм запятых.

**Г3 — Стилистическая «гладкость»** *(F07, F08, F09, F10)*  
AI-тексты демонстрируют сниженное лексическое разнообразие → низкий TTR, низкий HLR, мало причастий/пассивов.

**Г4 — Семантико-структурное несоответствие** *(F05, F06)*  
AI-тексты используют плоские синтаксические деревья при номинальном стиле → малая глубина зависимостей, высокий NV-ratio.

---

## Использование модели в коде

```python
from src.features import get_syntactic_features
from src.model import AICoreDetector

detector = AICoreDetector.load('cache/syntactic_detector.pkl')

text = "Ваш текст для проверки..."
features = get_syntactic_features(text)
result = detector.predict_with_explanation(text, features)

print(result['verdict'])          # 'AI-Generated' или 'Human-Written'
print(result['probability'])      # вероятность P(AI)
print(result['confidence_level']) # 'Высокая' / 'Средняя' / 'Низкая'
print(result['reasons'])          # список причин по гипотезам Г1–Г4
```

---

## Зависимости

| Пакет | Версия | Назначение |
|-------|--------|------------|
| spacy | ≥ 3.7 | NLP-парсинг |
| ru_core_news_lg | ≥ 3.7 | Русская языковая модель |
| scikit-learn | ≥ 1.3 | Random Forest, метрики |
| numpy | ≥ 1.24 | Числовые вычисления |
| pandas | ≥ 2.0 | Работа с данными |
| scipy | ≥ 1.11 | Статистические тесты (Mann–Whitney) |
| shap | ≥ 0.44 | Интерпретация SHAP |
| matplotlib | ≥ 3.7 | Визуализация |
| seaborn | ≥ 0.13 | Статистические графики |
| tqdm | ≥ 4.65 | Прогресс-бар |
| joblib | ≥ 1.3 | Сохранение модели |