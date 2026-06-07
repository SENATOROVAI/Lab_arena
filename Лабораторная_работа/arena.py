"""
Мини LM Arena (лекция 10).
Сравниваем ответы локальных моделей и ведём Elo-рейтинг.

Что добавил к базовой версии:
- запись каждой битвы в журнал и экспорт в JSON (имена моделей, вопросы,
  ответы и история изменения рейтингов);
- сохранение состояния между запусками (arena_state.json);
- авто-режим, где вопрос задаёт сама программа, а оценивают
  модели-судьи (LLM-as-a-judge) с защитой от позиционного смещения;
- определение моделей и для LM Studio, и для Ollama.

Нужен OpenAI-совместимый сервер на :1234 c парой chat-моделей.
Запуск:    python arena.py                        (http://localhost:5010)
Генерация JSON:  python run_arena_battles.py --battles 12
"""

import os
import json
import random
import re
import threading
import urllib.request
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

# ********************* КОНФИГУРАЦИЯ *********************
# Настройки можно переопределять через переменные окружения.

BASE_URL       = os.getenv("ARENA_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY        = os.getenv("ARENA_API_KEY", "lm-studio")  # любой непустой ключ
PORT           = int(os.getenv("ARENA_PORT", "5010"))
MAX_TOKENS     = int(os.getenv("ARENA_MAX_TOKENS", "1024"))
TEMPERATURE    = float(os.getenv("ARENA_TEMPERATURE", "0.7"))
ELO_K          = int(os.getenv("ARENA_ELO_K", "32"))
ELO_START      = int(os.getenv("ARENA_ELO_START", "1000"))

# Внутренний REST API LM Studio (для probe загруженных моделей). На Ollama его
# нет (404) — тогда автоматически используется стандартный /v1/models.
LM_STUDIO_API  = os.getenv("ARENA_LMSTUDIO_API",
                           "http://127.0.0.1:1234/api/v1/models")

# Файлы данных (рядом со скриптом)
HERE        = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(HERE, "arena_state.json")    # персистентность
EXPORT_FILE = os.path.join(HERE, "arena_export.json")   # экспорт-результат

# Темы для авто-генерации вопросов — несколько блоков для разнообразия.
TOPIC_BLOCKS = {
    "код":        "напиши или объясни фрагмент кода (Python/SQL/алгоритмы)",
    "математика": "математическая или логическая задача с рассуждением",
    "текст":      "объяснение понятия простыми словами / работа с текстом",
    "диалог":     "практический совет или сравнение подходов",
}

# ********************* ИНИЦИАЛИЗАЦИЯ *********************

app    = Flask(__name__)
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Потокобезопасность: и Flask, и драйвер могут писать в общее состояние.
_lock = threading.Lock()

# Словарь рейтингов: { model_id: {elo, wins, losses, ties, battles, elo_history} }
#   elo_history — рейтинг после каждой коррекции, начиная со стартового.
ratings = {}

# Журнал всех битв — полные данные для экспорта в JSON.
battles = []


# ********************* ПЕРСИСТЕНТНОСТЬ *********************

def save_state():
    """Сохраняет рейтинги и журнал битв на диск (атомарно через temp-файл)."""
    data = {"ratings": ratings, "battles": battles,
            "saved_at": _now(), "config": _config_snapshot()}
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def load_state():
    """Восстанавливает состояние из arena_state.json, если он есть."""
    global ratings, battles
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        ratings = data.get("ratings", {})
        battles = data.get("battles", [])
        print(f"[load_state] Восстановлено: моделей={len(ratings)}, битв={len(battles)}")
    except Exception as e:
        print(f"[load_state] Не удалось прочитать {STATE_FILE}: {e}")


# ********************* ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ *********************

def _now():
    """ISO-время в UTC — для меток в журнале."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config_snapshot():
    return {"base_url": BASE_URL, "elo_k": ELO_K, "elo_start": ELO_START,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}


def get_loaded_models():
    """
    Возвращает список доступных chat-моделей. BACKEND-AGNOSTIC:

      1) Сначала пробуем внутренний REST API LM Studio (/api/v1/models) —
         он отдаёт поле loaded_instances и позволяет показать только реально
         ЗАГРУЖЕННЫЕ В ПАМЯТЬ модели (как в оригинале).
      2) Если его нет (Ollama, llama.cpp → 404/ошибка) — берём стандартный
         OpenAI-эндпоинт /v1/models. Эмбеддинг-модели отфильтровываем по имени.
    """
    # --- Попытка 1: LM Studio внутренний API ---
    try:
        with urllib.request.urlopen(LM_STUDIO_API, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        loaded = []
        for m in data.get("models", []):
            if m.get("type") != "llm":         # пропускаем embedding
                continue
            inst = m.get("loaded_instances", [])
            if inst:
                loaded.append(inst[0]["id"])
        if loaded:
            print(f"[get_loaded_models] LM Studio: загружено {len(loaded)} LLM")
            return loaded
    except Exception:
        pass  # тихо переходим к стандартному эндпоинту

    # --- Попытка 2: стандартный OpenAI /v1/models ---
    try:
        resp = client.models.list()
        models = [m.id for m in resp.data]
    except Exception as e:
        print(f"[get_loaded_models] Не удалось получить /v1/models: {e}")
        return []

    # Отсеиваем эмбеддинг-модели по характерным подстрокам в имени.
    def is_embedding(name):
        n = name.lower()
        return any(k in n for k in ("embed", "embedding", "bge", "e5-", "minilm"))

    chat = [m for m in models if not is_embedding(m)]
    print(f"[get_loaded_models] OpenAI /v1/models: {len(chat)} chat-моделей: {chat}")
    return chat


def ensure_rating(model_id):
    """Инициализирует запись рейтинга для модели. Идемпотентно."""
    if model_id not in ratings:
        ratings[model_id] = {
            "elo":     float(ELO_START),
            "wins":    0, "losses": 0, "ties": 0, "battles": 0,
            "elo_history": [float(ELO_START)],  # последовательные коррекции
        }
    elif "elo_history" not in ratings[model_id]:
        ratings[model_id]["elo_history"] = [ratings[model_id]["elo"]]


def _call_model(model_id, messages, result, idx,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                allow_reasoning=True):
    """
    Запрашивает ответ у одной модели (для запуска в потоке).
    Поддерживает reasoning-модели: если content пуст — берём reasoning_content.
    Для судьи allow_reasoning=False: «поток размышлений» ломает парсинг JSON,
    поэтому подмену на reasoning_content для судьи отключаем.
    """
    short = model_id.split("/")[-1]
    try:
        resp = client.chat.completions.create(
            model=model_id, messages=messages,
            temperature=temperature, max_tokens=max_tokens)
        msg = resp.choices[0].message
        content = msg.content
        if not content and allow_reasoning:
            content = getattr(msg, "reasoning_content", None)
        if not content:
            content = "[Модель вернула пустой ответ]"
        print(f"[_call_model] {short}: {len(content)} символов")
        result[idx] = content
    except Exception as e:
        print(f"[_call_model] {short}: ОШИБКА — {e}")
        result[idx] = f"[Ошибка модели {short}: {e}]"


def ask_two(question, model_a, model_b):
    """Параллельно спрашивает две модели. Возвращает (ответ_a, ответ_b)."""
    messages = [{"role": "user", "content": question}]
    out = [None, None]
    t1 = threading.Thread(target=_call_model, args=(model_a, messages, out, 0))
    t2 = threading.Thread(target=_call_model, args=(model_b, messages, out, 1))
    t1.start(); t2.start(); t1.join(); t2.join()
    return out[0], out[1]


def generate_question(topic=None, generator=None):
    """
    Генерирует тестовый вопрос моделью. Если задан topic — просим вопрос
    из этого тематического блока (для разнообразия в авто-битвах).
    """
    available = get_loaded_models()
    if not available:
        return "Объясни разницу между процессом и потоком в ОС.", None
    generator = generator or random.choice(available)

    topic_hint = ""
    if topic and topic in TOPIC_BLOCKS:
        topic_hint = f"\nТЕМА вопроса: {TOPIC_BLOCKS[topic]}.\n"

    system_prompt = (
        "Ты генератор тестовых вопросов для сравнения языковых моделей. "
        "Придумай ОДИН содержательный вопрос, на который разные модели дадут "
        "заметно разные ответы. Отвечай ТОЛЬКО вопросом, без предисловий и "
        "нумерации." + topic_hint + "\n"
        "Примеры:\n"
        "- Напиши функцию на Python для решета Эратосфена до N\n"
        "- Объясни разницу между TCP и UDP за 3 предложения\n"
        "- Плюсы и минусы микросервисов против монолита?\n"
        "- Что такое переобучение нейросети и как с ним бороться?\n"
        "- Чем отличается merge sort от quick sort и когда что использовать?\n"
    )
    result = [None]
    _call_model(generator,
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": "Придумай новый вопрос в том же стиле."}],
                result, 0, temperature=0.9, max_tokens=200)
    q = (result[0] or "").strip().strip('"').strip("'").strip()
    # Убираем возможный ведущий маркер списка
    q = re.sub(r"^[-•\d.\)\s]+", "", q).strip()
    return (q or "Объясни разницу между процессом и потоком в ОС."), generator


# ********************* ELO *********************

def elo_update(winner_id, loser_id):
    """
    Пересчёт Elo после победы winner над loser. Возвращает детали вычисления
    (для UI и журнала). Здесь же дописываем elo_history обеих моделей —
    это и есть «последовательная коррекция рейтингов».
    """
    ew = ratings[winner_id]["elo"]
    el = ratings[loser_id]["elo"]
    expected_w = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))
    delta = round(ELO_K * (1 - expected_w), 1)

    ratings[winner_id]["elo"] = round(ew + delta, 1)
    ratings[loser_id]["elo"]  = round(el - delta, 1)
    ratings[winner_id]["elo_history"].append(ratings[winner_id]["elo"])
    ratings[loser_id]["elo_history"].append(ratings[loser_id]["elo"])

    return {
        "winner": winner_id, "loser": loser_id,
        "ew_before": ew, "el_before": el,
        "expected_pct": round(expected_w * 100, 1),
        "delta": delta,
        "ew_after": ratings[winner_id]["elo"],
        "el_after": ratings[loser_id]["elo"],
        "K": ELO_K,
    }


def record_battle(question, model_a, model_b, ans_a, ans_b,
                  shown_left, winner, vote_source, judge_details=None,
                  topic=None):
    """
    Применяет результат битвы к рейтингам и СОХРАНЯЕТ полную запись в журнал.

    winner: "a" | "b" | "tie" (в терминах model_a/model_b, не left/right).
    Возвращает запись битвы (dict).
    """
    with _lock:
        for m in (model_a, model_b):
            ensure_rating(m)
            ratings[m]["battles"] += 1

        elo_details = None
        if winner == "a":
            elo_details = elo_update(model_a, model_b)
            ratings[model_a]["wins"] += 1
            ratings[model_b]["losses"] += 1
        elif winner == "b":
            elo_details = elo_update(model_b, model_a)
            ratings[model_b]["wins"] += 1
            ratings[model_a]["losses"] += 1
        else:  # ничья — Elo не меняется (как в оригинале)
            ratings[model_a]["ties"] += 1
            ratings[model_b]["ties"] += 1

        # Снимок коррекции рейтингов именно для этой битвы
        elo_snapshot = {
            "model_a": {
                "before": elo_details["ew_before"] if winner == "a"
                else (elo_details["el_before"] if winner == "b" else ratings[model_a]["elo"]),
                "after": ratings[model_a]["elo"],
            },
            "model_b": {
                "before": elo_details["el_before"] if winner == "a"
                else (elo_details["ew_before"] if winner == "b" else ratings[model_b]["elo"]),
                "after": ratings[model_b]["elo"],
            },
            "delta": (elo_details["delta"] if elo_details else 0.0),
            "expected_winner_pct": (elo_details["expected_pct"] if elo_details else None),
            "K": ELO_K,
        }
        elo_snapshot["model_a"]["delta"] = round(
            elo_snapshot["model_a"]["after"] - elo_snapshot["model_a"]["before"], 1)
        elo_snapshot["model_b"]["delta"] = round(
            elo_snapshot["model_b"]["after"] - elo_snapshot["model_b"]["before"], 1)

        record = {
            "index": len(battles) + 1,
            "timestamp": _now(),
            "topic": topic,
            "question": question,
            "model_a": model_a,
            "model_b": model_b,
            "answer_a": ans_a,
            "answer_b": ans_b,
            "shown_left": shown_left,        # какая модель была слева (слепой тест)
            "vote": {
                "source": vote_source,       # "human" | "judge"
                "winner": winner,            # "a" | "b" | "tie"
                "winner_model": (model_a if winner == "a"
                                 else model_b if winner == "b" else None),
                "judge_details": judge_details,
            },
            "elo": elo_snapshot,             # последовательная коррекция рейтингов
        }
        battles.append(record)
        save_state()
    return record


# ********************* LLM-as-a-JUDGE (панель судей) *********************

JUDGE_SYSTEM = (
    "Ты — объективный и беспристрастный судья качества ответов языковых моделей. "
    "Тебе дают вопрос и два ответа (Ответ 1 и Ответ 2). Оцени каждый по трём "
    "критериям от 1 до 5: точность (accuracy), полезность (helpfulness), "
    "краткость без потери смысла (brevity). Затем выбери победителя.\n"
    "НЕ поддавайся на длину ответа и на порядок — суди по сути.\n"
    'Верни СТРОГО JSON вида: '
    '{"answer1": {"accuracy": n, "helpfulness": n, "brevity": n}, '
    '"answer2": {"accuracy": n, "helpfulness": n, "brevity": n}, '
    '"winner": "1" | "2" | "tie", "reason": "одно короткое предложение"}'
)


def _extract_json_object(text):
    """Находит ПЕРВЫЙ сбалансированный {...} объект (надёжнее жадного regex,
    который склеивал бы несколько объектов / текст вокруг)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _parse_judge_json(text):
    """Терпимый парсер JSON от модели-судьи (локальные модели шумят)."""
    if not text:
        return None
    raw = _extract_json_object(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        try:  # последняя попытка: вычистить хвостовые запятые
            return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
        except Exception:
            return None


def _score_of(parsed, key):
    """Сумма баллов (accuracy+helpfulness+brevity) для answer1/answer2."""
    block = (parsed or {}).get(key) or {}
    total = 0.0
    for k in ("accuracy", "helpfulness", "brevity"):
        try:
            total += float(block.get(k, 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _judge_once(judge_model, question, ans1, ans2):
    """
    Один проход судьи. Возвращает (winner '1'/'2'/'tie', s1, s2, parsed),
    где s1/s2 — суммарные баллы каждого ответа (для агрегации по панели).
    """
    user = (f"ВОПРОС:\n{question}\n\nОтвет 1:\n{ans1}\n\nОтвет 2:\n{ans2}\n\n"
            "Оцени и верни JSON.")
    result = [None]
    _call_model(judge_model,
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": user}],
                result, 0, temperature=0.0, max_tokens=400, allow_reasoning=False)
    parsed = _parse_judge_json(result[0])
    s1, s2 = _score_of(parsed, "answer1"), _score_of(parsed, "answer2")
    winner = "tie"
    if parsed and parsed.get("winner") in ("1", "2", "tie"):
        winner = parsed["winner"]
    elif parsed:
        winner = "1" if s1 > s2 else "2" if s2 > s1 else "tie"
    return winner, s1, s2, (parsed or {"raw": result[0]})


def judge_panel(question, model_a, model_b, ans_a, ans_b, judges):
    """
    Панель судей оценивает битву model_a vs model_b.

    Митигация bias:
      • POSITION-SWAP (против position bias): каждый судья судит ДВАЖДЫ — один
        раз (A=Ответ1,B=Ответ2), второй раз с переставленными ответами. Баллы
        обоих порядков СУММИРУЮТСЯ — так систематическая надбавка «за позицию»
        одинаково попадает и к A, и к B и взаимно гасится. Поле `consistent`
        фиксирует, флипнул ли судья при перестановке (диагностика bias).
      • SELF-PREFERENCE: судьи-участники битвы исключаются вызывающим кодом.

    Решение — по СУММЕ БАЛЛОВ обоих ответов по всем судьям и обоим проходам
    (а не по «совпал ли вердикт» — малые модели часто флипают, что давало бы
    сплошные ничьи). Усреднение по двум порядкам как раз и гасит position bias.

    Возвращает (winner 'a'/'b'/'tie', список деталей по судьям).
    """
    sum_a = sum_b = 0.0
    valid_passes = 0
    votes = {"a": 0, "b": 0, "tie": 0}   # «сырые» вердикты — для наглядности
    details = []
    for jm in judges:
        # Проход 1 (прямой): A=Ответ1, B=Ответ2
        w1, s1a, s1b, d1 = _judge_once(jm, question, ans_a, ans_b)
        # Проход 2 (swap): B=Ответ1, A=Ответ2  → баллы answer1 относятся к B
        w2, s2b, s2a, d2 = _judge_once(jm, question, ans_b, ans_a)

        # вердикты в терминах a/b (для лога)
        v1 = {"1": "a", "2": "b", "tie": "tie"}[w1]
        v2 = {"1": "b", "2": "a", "tie": "tie"}[w2]
        for v in (v1, v2):
            votes[v] += 1

        # агрегируем баллы (если проход распарсился)
        ja = sa = sb = 0.0
        if s1a or s1b:
            sum_a += s1a; sum_b += s1b; valid_passes += 1; ja += 1; sa += s1a; sb += s1b
        if s2a or s2b:
            sum_a += s2a; sum_b += s2b; valid_passes += 1; ja += 1; sa += s2a; sb += s2b

        details.append({
            "judge": jm,
            "pass1_winner": v1, "pass2_winner_swapped": v2,
            "consistent": v1 == v2,           # устойчивость к перестановке (bias-метрика)
            "score_a": round(sa, 1), "score_b": round(sb, 1),
            "raw": {"direct": d1, "swapped": d2},
        })

    if valid_passes == 0:
        winner = "tie"
    else:
        winner = "a" if sum_a > sum_b else "b" if sum_b > sum_a else "tie"
    return winner, {"votes": votes, "score_a": round(sum_a, 1),
                    "score_b": round(sum_b, 1), "valid_passes": valid_passes,
                    "per_judge": details}


def _is_big(name):
    """Эвристика: «тяжёлая» модель (медленная как судья)."""
    n = name.lower()
    return any(s in n for s in ("7b", "8b", "13b", "14b", "70b"))


def pick_judges(model_a, model_b, available, max_judges=2):
    """
    Выбирает судей с митигацией bias и оглядкой на скорость:
      • SELF-PREFERENCE: исключаем участников битвы (модель не судит себя).
      • СКОРОСТЬ: судьи — быстрые малые модели (большие 7B+ только участвуют).
    Если быстрых чистых судей не осталось — деградируем до любого не-участника.
    """
    fast_clean = [m for m in available
                  if m not in (model_a, model_b) and not _is_big(m)]
    if fast_clean:
        return fast_clean[:max_judges]
    # больших исключаем из судей по скорости, но как независимые судьи — можно
    any_clean = [m for m in available if m not in (model_a, model_b)]
    # ВАЖНО: участника битвы судьёй НЕ назначаем (self-preference bias).
    # Если независимых судей нет (всего 2 модели) — вернём [], тогда битва
    # фиксируется без судейского вердикта (для авто-режима нужно ≥3 модели).
    return any_clean[:max_judges]


def run_autobattle(topic=None, judges=None, available=None):
    """
    Полный цикл одной АВТО-битвы:
      вопрос → 2 случайные модели → ответы → панель судей → запись в журнал.
    Возвращает запись битвы.
    """
    available = available or get_loaded_models()
    with _lock:
        for m in available:
            ensure_rating(m)
    if len(available) < 2:
        raise RuntimeError("Нужно минимум 2 модели")

    model_a, model_b = random.sample(available, 2)
    question, gen = generate_question(topic=topic)
    ans_a, ans_b = ask_two(question, model_a, model_b)

    jd = judges if judges is not None else pick_judges(model_a, model_b, available)
    winner, panel = judge_panel(question, model_a, model_b, ans_a, ans_b, jd)
    panel["judges"] = jd
    panel["question_generator"] = gen

    # Слепой порядок (для совместимости со схемой; в авто-режиме судья всё
    # равно видит оба порядка, но фиксируем «как показали бы человеку»).
    shown_left = "model_a" if random.random() < 0.5 else "model_b"

    return record_battle(question, model_a, model_b, ans_a, ans_b,
                         shown_left, winner, "judge",
                         judge_details=panel, topic=topic)


# ********************* ЭКСПОРТ JSON (задание 2) *********************

def build_export():
    """Собирает итоговый JSON со всеми данными тестов."""
    with _lock:
        leaderboard = sorted(
            ratings.items(), key=lambda kv: kv[1]["elo"], reverse=True)
        return {
            "meta": {
                "generated_at": _now(),
                "backend": BASE_URL,
                "elo_k": ELO_K,
                "elo_start": ELO_START,
                "total_battles": len(battles),
                "models": list(ratings.keys()),
                "description": "Экспорт данных Мини-LM-Arena: имена моделей, "
                               "вопросы, ответы и последовательные коррекции Elo.",
            },
            # Имена моделей + финальные рейтинги/статистика
            "final_ratings": ratings,
            "leaderboard": [
                {"rank": i + 1, "model": mid, "elo": r["elo"],
                 "wins": r["wins"], "losses": r["losses"],
                 "ties": r["ties"], "battles": r["battles"]}
                for i, (mid, r) in enumerate(leaderboard)
            ],
            # Последовательные коррекции рейтингов (траектория Elo по моделям)
            "rating_history": {mid: r.get("elo_history", [])
                               for mid, r in ratings.items()},
            # Полный журнал битв: вопросы, ответы, голоса, коррекции Elo
            "battles": battles,
        }


def export_json(path=EXPORT_FILE):
    """Пишет итоговый JSON на диск и возвращает его."""
    data = build_export()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[export_json] Записано {len(battles)} битв в {path}")
    return data


# ********************* FLASK МАРШРУТЫ *********************

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/models")
def models_route():
    models = get_loaded_models()
    with _lock:
        for m in models:
            ensure_rating(m)
    return jsonify({"models": models})


@app.route("/ask", methods=["POST"])
def ask():
    """Слепой тест: 2 случайные модели отвечают на вопрос (порядок случаен)."""
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Пустой вопрос"}), 400

    available = get_loaded_models()
    with _lock:
        for m in available:
            ensure_rating(m)
    if len(available) < 2:
        return jsonify({"error": "Нужно минимум 2 загруженные модели"}), 400

    model_a, model_b = random.sample(available, 2)
    ans_a, ans_b = ask_two(question, model_a, model_b)

    if random.random() < 0.5:
        left, right, al, ar = model_a, model_b, ans_a, ans_b
    else:
        left, right, al, ar = model_b, model_a, ans_b, ans_a
    return jsonify({"left": left, "right": right, "ans_left": al, "ans_right": ar})


@app.route("/autoquestion", methods=["POST"])
def autoquestion():
    data = request.get_json(silent=True) or {}
    q, gen = generate_question(topic=data.get("topic"))
    return jsonify({"question": q, "generator": gen})


@app.route("/vote", methods=["POST"])
def vote():
    """Ручной голос. left/right — модели, winner — left|right|tie."""
    data = request.get_json() or {}
    winner_lr = data.get("winner")
    left = data.get("left")
    right = data.get("right")
    if not left or not right:
        return jsonify({"error": "Не указаны модели"}), 400

    # Переводим left/right → a/b (a == left, b == right)
    winner = {"left": "a", "right": "b", "tie": "tie"}.get(winner_lr, "tie")
    # Здесь model_a := left, model_b := right; ответы в ручном режиме не
    # пересохраняем (они уже показаны), но фиксируем как пустые-плейсхолдеры,
    # чтобы JSON-схема была единой.
    ans = data.get("ans_left", ""), data.get("ans_right", "")
    rec = record_battle(data.get("question", "[ручной тест]"),
                        left, right, ans[0], ans[1],
                        "model_a", winner, "human", topic=data.get("topic"))
    return jsonify({"ratings": ratings, "elo_details": rec["elo"],
                    "battle": rec})


@app.route("/autobattle", methods=["POST"])
def autobattle():
    """Запускает N авто-битв с панелью LLM-судей. Возвращает результаты."""
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 1))
    n = max(1, min(n, 30))
    topics = list(TOPIC_BLOCKS.keys())
    results = []
    available = get_loaded_models()
    for i in range(n):
        topic = topics[i % len(topics)]
        try:
            rec = run_autobattle(topic=topic, available=available)
            results.append(rec)
        except Exception as e:
            results.append({"error": str(e)})
    return jsonify({"battles": results, "ratings": ratings,
                    "total": len(battles)})


@app.route("/stats")
def stats():
    with _lock:   # читаем общее состояние под локом (Flask threaded=True)
        return jsonify({"ratings": ratings, "battles_count": len(battles)})


@app.route("/history")
def history():
    """Траектории Elo по моделям — для графика «последовательных коррекций»."""
    with _lock:
        return jsonify({"history": {m: list(r.get("elo_history", []))
                                    for m, r in ratings.items()}})


@app.route("/export", methods=["GET", "POST"])
def export_route():
    data = export_json()
    return jsonify({"status": "ok", "file": EXPORT_FILE,
                    "battles": len(battles), "data": data})


@app.route("/reset", methods=["POST"])
def reset_route():
    global ratings, battles
    with _lock:
        ratings = {}
        battles = []
        save_state()
    return jsonify({"status": "ok"})


# ********************* HTML / JS ФРОНТЕНД *********************

HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Мини LM Arena+</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root{--bg:#f1f5f9;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b;
    --blue:#3b82f6;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--purple:#8b5cf6;}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;min-height:100vh}
  header{background:var(--card);border-bottom:1px solid var(--border);padding:12px 24px;
    display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  header h1{font-size:17px;font-weight:700;flex:1}
  .badge{font-size:12px;color:var(--muted);background:#f1f5f9;padding:3px 10px;border-radius:20px}
  .main{max-width:1000px;margin:0 auto;padding:18px 16px;display:flex;flex-direction:column;gap:14px}
  .section{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
  .section-title{font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;
    letter-spacing:.4px;margin-bottom:10px}
  .ask-row{display:flex;gap:8px;align-items:flex-end}
  .ask-row textarea{flex:1;border:1px solid var(--border);border-radius:8px;padding:10px 12px;
    font-size:14px;resize:vertical;min-height:60px;font-family:inherit;color:var(--text);background:var(--bg)}
  .ask-btns{display:flex;flex-direction:column;gap:6px}
  .btn{padding:9px 16px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;
    transition:opacity .2s;white-space:nowrap}
  .btn:disabled{opacity:.45;cursor:not-allowed}
  .btn-primary{background:var(--blue);color:#fff}.btn-auto{background:var(--purple);color:#fff}
  .btn-judge{background:var(--green);color:#fff}.btn-ghost{background:#eef2f7;color:var(--text)}
  .btn:hover:not(:disabled){opacity:.85}
  .toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .arena{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .card{background:var(--card);border:2px solid var(--border);border-radius:12px;padding:14px;
    display:flex;flex-direction:column;gap:8px;min-height:160px;transition:border-color .25s,box-shadow .25s}
  .card.winner{border-color:var(--green);box-shadow:0 0 0 3px #22c55e22}.card.loser{opacity:.7}
  .card-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase}
  .card-answer{font-size:13.5px;line-height:1.7;white-space:pre-wrap;overflow-y:auto;max-height:300px;flex:1}
  .card-model{font-size:11px;color:var(--blue);font-style:italic;display:none;padding:4px 0 0;
    border-top:1px solid var(--border)}
  .vote-row{display:flex;gap:10px;justify-content:center;padding:4px 0}
  .btn-vote{padding:10px 24px;border:2px solid var(--border);background:var(--card);border-radius:8px;
    font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
  .btn-vote:hover:not(:disabled){transform:translateY(-1px)}.btn-vote:disabled{opacity:.4}
  .elo-panel{background:#0f172a;border-radius:12px;padding:16px;color:#e2e8f0;
    font-family:'Courier New',monospace;font-size:13px;line-height:1.9}
  .ep-title{font-family:system-ui;font-size:13px;font-weight:700;color:#94a3b8;margin-bottom:10px;
    text-transform:uppercase}
  .elo-step{display:flex;gap:8px;flex-wrap:wrap}.elo-step .lbl{color:#475569;min-width:210px}
  .elo-step .val{color:#f8fafc;font-weight:700}.elo-step .val.up{color:#4ade80}.elo-step .val.down{color:#f87171}
  .elo-step .formula{color:#64748b;font-size:11.5px}
  table.rt{width:100%;border-collapse:collapse;font-size:13px}
  table.rt th{padding:6px 10px;text-align:left;color:var(--muted);font-size:11px;font-weight:700;
    text-transform:uppercase;border-bottom:2px solid var(--border)}
  table.rt td{padding:8px 10px;border-bottom:1px solid var(--border)}
  .elo-num{font-weight:700;font-size:14px}.rank-1{color:#f59e0b}.rank-2{color:#94a3b8}.rank-3{color:#cd7c2f}
  .chart-wrap{height:220px;position:relative;margin-top:12px}
  .msg{text-align:center;color:var(--muted);font-size:14px;padding:30px 0}
  .spinner{display:inline-block;width:18px;height:18px;border:3px solid var(--border);
    border-top-color:var(--blue);border-radius:50%;animation:spin .7s linear infinite;
    vertical-align:middle;margin-right:8px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1e293b;color:#fff;
    padding:9px 18px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none;z-index:100}
  .toast.show{opacity:1}
  .hint{font-size:11px;color:var(--muted);margin-top:5px}
  .judge-log{font-size:12px;color:var(--muted);max-height:200px;overflow-y:auto;margin-top:8px;
    border-top:1px dashed var(--border);padding-top:8px}
  .judge-row{padding:4px 0;border-bottom:1px solid #f1f5f9}
  select{border:1px solid var(--border);border-radius:8px;padding:8px;font-size:13px;background:var(--bg)}
</style>
</head>
<body>
<header>
  <span style="font-size:20px">⚔️</span>
  <h1>Мини LM Arena+ <span style="font-weight:400;color:var(--muted);font-size:13px">(лекция 10, доработка)</span></h1>
  <span class="badge" id="models-badge">⏳ определяю модели…</span>
</header>

<div class="main">
  <!-- ВОПРОС -->
  <div class="section">
    <div class="section-title">Вопрос</div>
    <div class="ask-row">
      <textarea id="question" placeholder="Задайте вопрос — или «Авто» для генерации"></textarea>
      <div class="ask-btns">
        <button class="btn btn-primary" id="btn-ask" onclick="doAsk()">▶ Спросить</button>
        <button class="btn btn-auto" id="btn-auto" onclick="doAutoQuestion()">🎲 Авто</button>
      </div>
    </div>
    <div class="hint" id="auto-hint"></div>
  </div>

  <!-- АВТО-БИТВЫ С СУДЬЁЙ -->
  <div class="section">
    <div class="section-title">⚖️ Авто-битвы с панелью LLM-судей (генерация данных)</div>
    <div class="toolbar">
      <span style="font-size:13px">Сколько битв:</span>
      <select id="nbattles">
        <option>1</option><option>3</option><option selected>5</option>
        <option>10</option><option>12</option>
      </select>
      <button class="btn btn-judge" id="btn-autobattle" onclick="doAutoBattle()">🤖 Запустить авто-битвы</button>
      <button class="btn btn-ghost" onclick="doExport()">💾 Экспорт JSON</button>
      <button class="btn btn-ghost" onclick="doReset()">♻︎ Сброс</button>
    </div>
    <div class="hint" id="autobattle-hint">Система сама придумает вопросы, спросит модели и
      позовёт судей. Митигация bias: position-swap + исключение участников из жюри.</div>
    <div class="judge-log" id="judge-log" style="display:none"></div>
  </div>

  <!-- АРЕНА -->
  <div id="arena-area">
    <div class="section"><div class="msg">Задайте вопрос или запустите авто-битвы</div></div>
  </div>

  <!-- ELO ВЫЧИСЛЕНИЕ -->
  <div id="elo-calc-wrap" style="display:none">
    <div class="elo-panel"><div class="ep-title">📐 Вычисление Elo</div><div id="elo-calc-body"></div></div>
  </div>

  <!-- СТАТИСТИКА -->
  <div class="section" id="stats-box" style="display:none">
    <div class="section-title">📊 Рейтинг Elo</div>
    <table class="rt"><thead><tr>
      <th>#</th><th>Модель</th><th>Elo</th><th>Побед</th><th>Пораж.</th><th>Ничьих</th><th>Битв</th><th>Win%</th>
    </tr></thead><tbody id="ratings-tbody"></tbody></table>
    <div class="chart-wrap"><canvas id="eloChart"></canvas></div>
    <div class="section-title" style="margin-top:18px">📈 Траектория Elo (последовательные коррекции)</div>
    <div class="chart-wrap"><canvas id="trajChart"></canvas></div>
  </div>
</div>

<div class="toast" id="toast"></div>
<script>
let currentLeft=null,currentRight=null,currentAnsLeft='',currentAnsRight='',
    eloChart=null,trajChart=null,voteDone=false;
const COLORS=['#3b82f6','#22c55e','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#ec4899'];
function escHtml(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function showToast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2600)}

async function loadModels(){
  const b=document.getElementById('models-badge');
  try{const d=await(await fetch('/models')).json();
    if(!d.models.length){b.textContent='⚠️ Нет моделей — запустите сервер :1234';b.style.color='#ef4444';}
    else{b.textContent='✅ '+d.models.length+' моделей: '+d.models.map(m=>m.split('/').pop()).join(', ');
      b.style.color='#22c55e';}
    refreshStats();
  }catch(e){b.textContent='⚠️ Сервер :1234 недоступен';b.style.color='#ef4444';}
}

async function doAutoQuestion(){
  const btn=document.getElementById('btn-auto'),h=document.getElementById('auto-hint');
  btn.disabled=true;h.textContent='⏳ Генерирую вопрос…';
  try{const d=await(await fetch('/autoquestion',{method:'POST'})).json();
    if(d.error){h.textContent='⚠️ '+d.error;}else{
      document.getElementById('question').value=d.question;
      h.textContent='✨ Сгенерировал: '+(d.generator||'').split('/').pop();h.style.color='#8b5cf6';}
  }catch(e){h.textContent='⚠️ Ошибка';}
  btn.disabled=false;
}

async function doAsk(){
  const q=document.getElementById('question').value.trim();
  if(!q){showToast('Введите вопрос');return;}
  const btn=document.getElementById('btn-ask');btn.disabled=true;voteDone=false;
  document.getElementById('elo-calc-wrap').style.display='none';
  document.getElementById('arena-area').innerHTML=
    '<div class="section"><div class="msg"><span class="spinner"></span>Спрашиваю обе модели…</div></div>';
  try{const d=await(await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})})).json();
    if(d.error){showToast('Ошибка: '+d.error);btn.disabled=false;return;}
    currentLeft=d.left;currentRight=d.right;
    currentAnsLeft=d.ans_left;currentAnsRight=d.ans_right;renderArena(d,q);
  }catch(e){showToast('Ошибка соединения');}
  btn.disabled=false;
}

function renderArena(d,q){
  document.getElementById('arena-area').innerHTML=`
  <div class="section"><div class="section-title">Ответы — выберите лучший (имена скрыты)</div>
    <div class="arena">
      <div class="card" id="card-left"><div class="card-label">⬛ Модель A</div>
        <div class="card-answer">${escHtml(d.ans_left)}</div>
        <div class="card-model" id="name-left">🤖 ${escHtml(d.left)}</div></div>
      <div class="card" id="card-right"><div class="card-label">⬛ Модель B</div>
        <div class="card-answer">${escHtml(d.ans_right)}</div>
        <div class="card-model" id="name-right">🤖 ${escHtml(d.right)}</div></div>
    </div>
    <div class="vote-row" style="margin-top:14px">
      <button class="btn-vote" onclick="doVote('left','${escHtml(q)}')">👈 A лучше</button>
      <button class="btn-vote" onclick="doVote('tie','${escHtml(q)}')">🤝 Ничья</button>
      <button class="btn-vote" onclick="doVote('right','${escHtml(q)}')">B лучше 👉</button>
    </div></div>`;
}

async function doVote(choice,q){
  if(voteDone)return;voteDone=true;
  document.querySelectorAll('.btn-vote').forEach(b=>b.disabled=true);
  document.getElementById('name-left').style.display='block';
  document.getElementById('name-right').style.display='block';
  if(choice==='left'){card('card-left','winner');card('card-right','loser');}
  else if(choice==='right'){card('card-right','winner');card('card-left','loser');}
  try{const d=await(await fetch('/vote',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({winner:choice,left:currentLeft,right:currentRight,question:q,
        ans_left:currentAnsLeft,ans_right:currentAnsRight})})).json();
    renderEloCalc(choice,d.elo_details);refreshStats();
    showToast(choice==='tie'?'🤝 Ничья':'✅ Голос засчитан');
  }catch(e){showToast('Ошибка голоса');}
}
function card(id,cls){document.getElementById(id).classList.add(cls)}

function renderEloCalc(choice,det){
  const wrap=document.getElementById('elo-calc-wrap'),body=document.getElementById('elo-calc-body');
  wrap.style.display='block';
  if(choice==='tie'||!det||!det.delta){body.innerHTML='<div style="color:#94a3b8">🤝 Ничья — рейтинги не изменились.</div>';return;}
  const a=det.model_a,b=det.model_b;
  body.innerHTML=`
   <div class="elo-step"><span class="lbl">Δ (передано очков):</span>
     <span class="val">${Math.abs(det.delta)}</span>
     <span class="formula">= K×(1−ожид.) = ${det.K}×(...) </span></div>
   <div class="elo-step"><span class="lbl">Модель A:</span>
     <span class="val ${a.delta>=0?'up':'down'}">${a.before} → ${a.after} (${a.delta>=0?'+':''}${a.delta})</span></div>
   <div class="elo-step"><span class="lbl">Модель B:</span>
     <span class="val ${b.delta>=0?'up':'down'}">${b.before} → ${b.after} (${b.delta>=0?'+':''}${b.delta})</span></div>
   <div class="elo-step" style="margin-top:6px;color:#475569;font-size:11.5px">
     Сумма Elo сохранена: ${a.before}+${b.before} = ${a.after}+${b.after}</div>`;
}

async function doAutoBattle(){
  const n=parseInt(document.getElementById('nbattles').value);
  const btn=document.getElementById('btn-autobattle'),h=document.getElementById('autobattle-hint');
  const log=document.getElementById('judge-log');
  btn.disabled=true;log.style.display='block';
  log.innerHTML='<div class="spinner"></div> Идут '+n+' авто-битв (вопрос → ответы → судьи)…';
  try{const d=await(await fetch('/autobattle',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({n:n})})).json();
    log.innerHTML='';
    d.battles.forEach((bt)=>{
      if(bt.error){log.innerHTML+='<div class="judge-row">⚠️ '+escHtml(bt.error)+'</div>';return;}
      const w=bt.vote.winner_model?bt.vote.winner_model.split('/').pop():'ничья';
      const ma=bt.model_a.split('/').pop(),mb=bt.model_b.split('/').pop();
      log.innerHTML+=`<div class="judge-row"><b>#${bt.index}</b> [${bt.topic||''}]
        ${escHtml(ma)} vs ${escHtml(mb)} → 🏆 <b>${escHtml(w)}</b>
        <span style="color:#94a3b8">| Δ=${bt.elo.delta} | ${escHtml((bt.question||'').slice(0,70))}…</span></div>`;
    });
    refreshStats();showToast('Готово: +'+d.battles.length+' битв (всего '+d.total+')');
  }catch(e){log.innerHTML='⚠️ Ошибка авто-битв';}
  btn.disabled=false;
}

async function doExport(){
  try{const d=await(await fetch('/export',{method:'POST'})).json();
    showToast('💾 Экспортировано '+d.battles+' битв в arena_export.json');
  }catch(e){showToast('Ошибка экспорта');}
}
async function doReset(){
  if(!confirm('Сбросить все рейтинги и журнал?'))return;
  await fetch('/reset',{method:'POST'});refreshStats();showToast('Сброшено');
}

async function refreshStats(){
  try{
    const s=await(await fetch('/stats')).json();
    if(Object.keys(s.ratings).length) renderStats(s.ratings);
    const h=await(await fetch('/history')).json();
    renderTraj(h.history);
  }catch(e){}
}

function renderStats(ratings){
  document.getElementById('stats-box').style.display='block';
  const sorted=Object.entries(ratings).sort((a,b)=>b[1].elo-a[1].elo);
  const tb=document.getElementById('ratings-tbody');tb.innerHTML='';
  sorted.forEach(([id,v],i)=>{
    const rc=i===0?'rank-1':i===1?'rank-2':i===2?'rank-3':'';
    const wp=v.battles>0?((v.wins/v.battles)*100).toFixed(0)+'%':'—';
    tb.innerHTML+=`<tr><td class="${rc}" style="font-weight:700">${i===0?'🥇':i===1?'🥈':i===2?'🥉':i+1}</td>
      <td>${escHtml(id.split('/').pop())}</td>
      <td><span class="elo-num ${rc}">${v.elo}</span></td>
      <td style="color:#22c55e;font-weight:600">${v.wins}</td>
      <td style="color:#ef4444">${v.losses}</td><td style="color:#94a3b8">${v.ties}</td>
      <td>${v.battles}</td><td style="font-weight:600">${wp}</td></tr>`;
  });
  const labels=sorted.map(([id])=>id.split('/').pop().slice(0,18));
  const elos=sorted.map(([,v])=>v.elo);
  const ctx=document.getElementById('eloChart').getContext('2d');
  if(eloChart)eloChart.destroy();
  eloChart=new Chart(ctx,{type:'bar',data:{labels,datasets:[{label:'Elo',data:elos,
    backgroundColor:elos.map((_,i)=>COLORS[i%COLORS.length]+'bb'),
    borderColor:elos.map((_,i)=>COLORS[i%COLORS.length]),borderWidth:2,borderRadius:6}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{y:{beginAtZero:false,min:Math.min(...elos)-40}}}});
}

function renderTraj(history){
  const entries=Object.entries(history).filter(([,h])=>h&&h.length);
  if(!entries.length)return;
  const maxLen=Math.max(...entries.map(([,h])=>h.length));
  const labels=Array.from({length:maxLen},(_,i)=>i);
  const ds=entries.map(([id,h],i)=>({label:id.split('/').pop(),data:h,
    borderColor:COLORS[i%COLORS.length],backgroundColor:'transparent',
    borderWidth:2,tension:.2,pointRadius:2}));
  const ctx=document.getElementById('trajChart').getContext('2d');
  if(trajChart)trajChart.destroy();
  trajChart=new Chart(ctx,{type:'line',data:{labels,datasets:ds},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{font:{size:11}}}},
      scales:{x:{title:{display:true,text:'коррекция #'}},y:{title:{display:true,text:'Elo'}}}}});
}

loadModels();
</script>
</body>
</html>"""


# ********************* ТОЧКА ВХОДА *********************

if __name__ == "__main__":
    load_state()
    print("\n  ⚔️  Мини LM Arena+ : http://localhost:%d" % PORT)
    print(f"  Backend: {BASE_URL}")
    print(f"  Состояние: {STATE_FILE}")
    print(f"  Настройки: K={ELO_K}, start={ELO_START}, max_tokens={MAX_TOKENS}\n")
    app.run(debug=False, port=PORT, threaded=True)
