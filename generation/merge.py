import pandas as pd
import re

# --- КОНФИГУРАЦИЯ ---
INPUT_AI = 'ai_dataset_humanized.csv'
INPUT_HUMAN = 'news.csv'
OUTPUT_FILE = 'ai_detection_ru_dataset_v4.csv'
TARGET_PER_CLASS = 5000

# 1. Загрузка данных
try:
    df_ai_raw = pd.read_csv(INPUT_AI)
    df_human_raw = pd.read_csv(INPUT_HUMAN)
    print("Файлы успешно загружены")
except Exception as e:
    print(f"Ошибка загрузки: {e}")

# 2. Функция глубокой очистки (убираем обрывки и метаданные)
def clean_full(text):
    if not isinstance(text, str): return ""
    # Убираем лишние пробелы и переносы
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\xa0', ' ')
    
    # Паттерны для очистки человеческих метаданных СМИ
    patterns = [
        r'^[А-ЯЁA-Z\s\-]{2,},\s\d+\s[а-яё]+\s[—–-]\s*(РИА Новости|ТАСС|Интерфакс|Лента\.ру)\.?\s*',
        r'^[А-ЯЁA-Z\s\-]{2,}\s[—–-]\s*(РИА Новости|ТАСС)\.?\s*',
        r'^[А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+\.\.\s*',
        r'^(РИА НОВОСТИ|ТАСС|ЛЕНТА\.РУ)\.?\s*',
        r'^ТАСС,\s+\d+\s+[а-яё]+\.\s*'
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    # Убираем обрывки в начале (всё до первой заглавной буквы)
    match_start = re.search(r'[А-ЯЁA-Z]', text)
    if match_start:
        text = text[match_start.start():]
    

    return text.strip()

# 3. Обработка человеческого корпуса (Human)
print("Подготовка Human-корпуса...")
df_human_pool = df_human_raw.dropna(subset=['text']).drop_duplicates(subset=['text']).copy()
df_human_pool['text'] = df_human_pool['text'].apply(clean_full)

df_human_sampled = df_human_pool.sample(n=min(TARGET_PER_CLASS, len(df_human_pool)), random_state=42)

df_h = pd.DataFrame({
    'text': df_human_sampled['text'],
    'is_ai': False,
    'is_redacted': False,
    'source': 'human_archive',
    'topic': df_human_sampled['rubric'].fillna('General'),
    'temperature': 0.0,
    'top_p': 0.0
})

# 4. Обработка ИИ корпуса (AI)
print("Подготовка AI-корпуса...")
df_ai_pool = df_ai_raw.copy()
df_ai_pool['text'] = df_ai_pool['text'].apply(clean_full)

df_ai_sampled = df_ai_pool.sample(n=min(TARGET_PER_CLASS, len(df_ai_pool)), random_state=42)

df_a = pd.DataFrame({
    'text': df_ai_sampled['text'],
    'is_ai': True,
    'is_redacted': df_ai_sampled['redacted'].astype(bool),
    'source': df_ai_sampled['source'],
    'topic': df_ai_sampled['topic'].fillna('General'),
    'temperature': df_ai_sampled.get('temperature', 0.7), # Если нет в исходнике, ставим среднее
    'top_p': df_ai_sampled.get('top_p', 0.9)
})

# 5. Сборка, перемешивание и расчет метрик
print("Финализация датасета...")
final_df = pd.concat([df_h, df_a], ignore_index=True)

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.index.name = 'id'
final_df.reset_index(inplace=True)

# Считаем длину
final_df['length_words'] = final_df['text'].apply(lambda x: len(str(x).split()))
final_df['length_chars'] = final_df['text'].apply(lambda x: len(str(x)))

# 6. Сохранение
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("\n" + "="*50)
print(f"СБОРКА ЗАВЕРШЕНА!")
print(f"Файл: {OUTPUT_FILE}")
print(f"Колонки: {final_df.columns.tolist()}")
print("-" * 50)
print(f"Всего строк: {len(final_df)}")
print(f"AI текстов: {len(final_df[final_df['is_ai']==True])}")
print(f"Human текстов: {len(final_df[final_df['is_ai']==False])}")
print(f"Redacted (Back-Translated): {final_df['is_redacted'].sum()}")
print(f"Средняя длина (слов): {int(final_df['length_words'].mean())}")
print("="*50)