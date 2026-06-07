"""
Прогон авто-битв для генерации arena_export.json.
Берёт логику из arena.py и без веб-сервера проводит N битв
(вопрос -> два ответа -> оценка судьями -> пересчёт Elo -> запись),
после чего сохраняет результат в JSON. Темы битв чередуются для
разнообразия.

    python run_arena_battles.py --battles 12          # с нуля
    python run_arena_battles.py --battles 8 --keep    # дописать к текущему
"""

import argparse
import time

import arena


def main():
    ap = argparse.ArgumentParser(description="Авто-битвы Мини-LM-Arena")
    ap.add_argument("--battles", type=int, default=12,
                    help="сколько битв провести (по умолчанию 12)")
    ap.add_argument("--keep", action="store_true",
                    help="не сбрасывать существующее состояние, дописать к нему")
    args = ap.parse_args()

    if args.keep:
        arena.load_state()
    else:
        # чистый прогон: пустые рейтинги и журнал
        arena.ratings = {}
        arena.battles = []

    available = arena.get_loaded_models()
    print(f"Доступные модели ({len(available)}): {available}")
    if len(available) < 2:
        print("Нужно минимум 2 модели на сервере :1234. Останов.")
        return

    for m in available:
        arena.ensure_rating(m)

    topics = list(arena.TOPIC_BLOCKS.keys())
    start = len(arena.battles)
    t0 = time.time()

    for i in range(args.battles):
        topic = topics[i % len(topics)]
        n = start + i + 1
        print(f"\n=== Битва {n} | тема: {topic} ===")
        try:
            rec = arena.run_autobattle(topic=topic, available=available)
        except Exception as e:
            print(f"  ОШИБКА битвы: {e}")
            continue
        a = rec["model_a"].split("/")[-1]
        b = rec["model_b"].split("/")[-1]
        win = rec["vote"]["winner_model"]
        win = win.split("/")[-1] if win else "ничья"
        judges = [j.split("/")[-1] for j in rec["vote"]["judge_details"]["judges"]]
        print(f"  Q: {rec['question'][:90]}")
        print(f"  {a} vs {b}  ->  победитель: {win}  (судьи: {', '.join(judges)})")
        print(f"  Голоса панели: {rec['vote']['judge_details']['votes']}  Δelo={rec['elo']['delta']}")

    data = arena.export_json()

    # Краткая сводка
    print("\n" + "=" * 70)
    print(f"  Готово за {time.time()-t0:.0f}с. Всего битв: {len(arena.battles)}")
    print(f"  JSON: {arena.EXPORT_FILE}")
    print("  Итоговый рейтинг:")
    for row in data["leaderboard"]:
        print(f"    {row['rank']}. {row['model']:<24} Elo={row['elo']:<8} "
              f"W/L/T={row['wins']}/{row['losses']}/{row['ties']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
