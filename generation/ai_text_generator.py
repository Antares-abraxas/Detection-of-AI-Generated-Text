import pandas as pd
from openai import OpenAI
import random
import time
import csv
import os
import re  # Добавлено для фильтрации иероглифов
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- КОНФИГУРАЦИЯ ---
load_dotenv()
raw_tokens = os.getenv("HF_TOKENS", "")
TOKENS = [t.strip() for t in raw_tokens.split(",") if t.strip()]
if not TOKENS:
    print("КРИТИЧЕСКАЯ ОШИБКА: Токены не найдены в файле .env или файл отсутствует.")
    exit()
    
INPUT_FILE = "news.csv"
OUTPUT_FILE = "ai_dataset.csv"
TOTAL_TO_GENERATE = 10000

MODELS = [
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "label": "Llama-3.1-8B"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen-2.5-7B"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "label": "DeepSeek-R1-Llama"}
]

# Обновленные стили для жесткой привязки к RU-контексту
STYLES = [
    "в сухом официальном стиле российского информагентства",
    "в стиле аналитической колонки российского СМИ",
    "как краткую сводку для российского телеграм-канала",
    "в стиле классической газетной заметки (федеральный выпуск)"
]

def get_last_id(filename):
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        return 0
    try:
        df = pd.read_csv(filename)
        if df.empty: return 0
        return int(df['id'].max())
    except:
        return 0

def is_mostly_russian(text):
    """Проверка: минимум 50% слов должны содержать кириллицу."""
    words = text.split()
    if not words: return False
    
    russian_words_count = sum(1 for w in words if re.search('[а-яА-ЯёЁ]', w))
    percent = (russian_words_count / len(words)) * 100
    return percent >= 50

def main():
    token_idx = 0
    error_streak = 0 
    
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=TOKENS[token_idx])

    try:
        df_real = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Ошибка чтения {INPUT_FILE}: {e}")
        return

    current_id = get_last_id(OUTPUT_FILE)
    start_count = current_id 
    start_time = time.time()
    
    print(f"=== ПРОДОЛЖАЕМ ГЕНЕРАЦИЮ С ID: {current_id + 1} ===")
    print(f"--- Используется токен №{token_idx + 1} ---")

    file_exists = os.path.isfile(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(OUTPUT_FILE).st_size == 0:
            writer.writerow(["id", "text", "label", "source", "topic", "language", 
                            "temperature", "top_p", "length_words", "length_chars", "redacted"])

        while current_id < TOTAL_TO_GENERATE:
            if current_id > start_count and (current_id - start_count) % 100 == 0:
                elapsed = time.time() - start_time
                per_item = elapsed / (current_id - start_count)
                remaining = TOTAL_TO_GENERATE - current_id
                eta = str(timedelta(seconds=int(remaining * per_item)))
                print(f"\n>>> [ ПРОГРЕСС: {current_id}/{TOTAL_TO_GENERATE} | Осталось примерно: {eta} ] <<<\n")

            model_index = current_id % len(MODELS)
            model_cfg = MODELS[model_index]
            
            current_temp = round(random.uniform(0.3, 0.9), 2)
            current_top_p = round(random.uniform(0.8, 1.0), 2)
            #current_top_p = 0.9
            is_sensitive_model = any(m in model_cfg['id'].lower() for m in ["mistral", "hermes", "gemma", "phi"])
            target_length = random.choice(["300", "400", "600"])
            if "deepseek" in model_cfg['id'].lower():
                current_temp = 0.3
                current_top_p = 0.9
                
            sample = df_real.sample(n=1).iloc[0]
            title = sample['title']
            style = random.choice(STYLES)
            topic = str(sample.get('rubric', 'General')).strip()
            if topic == 'nan' or not topic:
                topic = 'General'

            print(f"[{current_id + 1}] {model_cfg['label']} | T={current_temp}...", end=" ", flush=True)

            try:
                payload = {
                    "model": model_cfg['id'],
                    "messages": [
                        {
                            "role": "system", 
                            "content": (
                                f"Ты — опытный российский журналист. Пиши СТРОГО на русском языке (кириллица). "
                                f"Стиль: {style}. Текст должен быть неотличим от профессиональной прессы. "
                                "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО: использовать иероглифы, латиницу, давать заголовки. "
                                "Выдавай ТОЛЬКО основной текст новости без приветствий. Start immediately with the text. No thinking. No <think> tags. No talk. "
                            )
                        },
                        {
                            "role": "user", 
                            "content": f"Напиши подробно новость на русском языке на основе заголовка: «{title}». Объем: минимум {target_length} слов."
                        }
                    ],
                    "max_tokens": 1500
                }

                if not is_sensitive_model:
                    payload["temperature"] = current_temp
                    payload["top_p"] = current_top_p
                    print(f"T={current_temp} | P={current_top_p} |", end=" ", flush=True)
                else:
                    # Для "чувствительных" моделей убираем всё лишнее
                    print(f"T=AUTO | P=AUTO |", end=" ", flush=True)
                
                chat_completion = client.chat.completions.create(**payload)   
                text = chat_completion.choices[0].message.content.strip()
                
                if text.startswith(title) or text.startswith(f"«{title}»"):
                    text = text.split('\n', 1)[-1].strip()
                
                words = len(text.split())
                chars = len(text)

                # Проверка: длина + отсутствие иероглифов + наличие русских букв
                if words >= 100 and is_mostly_russian(text):
                    current_id += 1 
                    error_streak = 0 
                    writer.writerow([
                        current_id, text, "deepfake", model_cfg['id'], topic, "ru", 
                        current_temp, current_top_p, words, chars, "false"
                    ])
                    f.flush()
                    print(f"OK ({words} сл.)")
                else:
                    reason = "Мало слов" if words < 100 else "Языковой брак"
                    print(f"⚠️ ПЕРЕГЕНЕРАЦИЯ ({reason})")

            except Exception as e:
                error_str = str(e)
                error_streak += 1
                
                # ВЫВОДИМ ПОЛНУЮ ОШИБКУ ТУТ:
                print(f"\nОШИБКА API ({model_cfg['label']}):")
                print("-" * 50)
                print(error_str)  # Теперь выведет всё сообщение целиком
                print("-" * 50)
                
                if "401" in error_str or "404" in error_str:
                    print(f"Критическая ошибка: {error_str}. Проверьте токен.")
                    break
                
                print(f"Ошибка API ({model_cfg['label']}): {error_str[:40]}...")

                if error_streak >= 3:
                    token_idx = (token_idx + 1) % len(TOKENS)
                    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=TOKENS[token_idx])
                    error_streak = 0
                    print(f"\n[СМЕНА ТОКЕНА] Переключился на токен №{token_idx + 1}. Ждем 10 сек...")
                    time.sleep(10)
                else:
                    print(f"Спим 5 сек. (Попытка {error_streak}/3 для этого токена)")
                    time.sleep(5)   

            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Процесс прерван пользователем. Прогресс сохранен.")