"""
RAG-приложение (доработка 04_rag_app.py, лекция 11).
Базовая версия умела: загрузку .txt/.md, посимвольный чанкинг,
поиск по эмбеддингам в ChromaDB и чат с показом источников.

Что добавил:
- гибридный поиск: эмбеддинги + BM25, результаты объединяю через RRF
  (помогает находить точные слова, номера и термины);
- чанкинг по абзацам, а не только по символам;
- загрузку .pdf (pypdf) и .docx (python-docx), не только текста;
- метаданные найденных фрагментов и ссылки вида [#1] в ответе модели;
- несколько документов сразу с фильтром по источнику;
- настройки (top-k, размер чанка, стратегия) прямо в интерфейсе.

Нужен OpenAI-совместимый сервер на :1234 (эмбеддинг-модель + LLM).
Запуск: python rag_app.py  (http://localhost:5011)
"""

import io
import os
import re

import chromadb
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ********************* КОНФИГУРАЦИЯ *********************

BASE_URL        = os.getenv("RAG_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY         = os.getenv("RAG_API_KEY", "lm-studio")
# Дефолты под Ollama. Для LM Studio задайте, например:
#   RAG_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
#   RAG_LLM_MODEL=google/gemma-4-e4b
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("RAG_LLM_MODEL", "qwen2.5:3b")

HERE       = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(HERE, "chroma_app_db")
COLLECTION = "rag_app"

# Параметры по умолчанию (переопределяются из UI на каждый запрос)
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 3
CANDIDATE_POOL = 10     # сколько кандидатов берём из каждого метода до RRF
RRF_K          = 60     # константа сглаживания в Reciprocal Rank Fusion

MAX_TOKENS  = 700
TEMPERATURE = 0.3
PORT = int(os.getenv("RAG_PORT", "5011"))


# ********************* ИНИЦИАЛИЗАЦИЯ *********************

app    = Flask(__name__)
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"})


# ********************* ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ ФАЙЛОВ *********************

def extract_text(filename: str, raw: bytes) -> str:
    """Достаёт текст из .txt/.md/.pdf/.docx. Бросает ValueError при проблеме."""
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("cp1251", errors="replace")  # запасная кодировка
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(raw))
        return "\n\n".join(p.text for p in d.paragraphs)
    raise ValueError("Поддерживаются .txt, .md, .pdf, .docx")


# ********************* ЧАНКИНГ *********************

def chunk_chars(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Посимвольное скользящее окно (как в оригинале), с offset'ами."""
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        piece = text[start:end]
        s = piece.strip()
        if s:
            # offset с учётом обрезанных пробелов слева
            lead = len(piece) - len(piece.lstrip())
            chunks.append({"text": s, "start": start + lead,
                           "end": start + lead + len(s)})
        start += size - overlap
    return chunks


def chunk_paragraphs(text, max_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Чанкинг по абзацам (structure-aware). Абзац = блок непустых строк,
    разделённый пустой строкой. Абзацы упаковываются в чанк до max_size
    символов, не разрывая абзац; слишком длинный абзац дробится посимвольно.

    Текст чанка берётся как ОРИГИНАЛЬНАЯ подстрока text[start:end], поэтому
    offset'ы всегда точны: text[start:end] == chunk["text"] (важно для метаданных).
    """
    # границы абзацев (с поправкой на ведущие/висячие пробелы)
    blocks = []
    for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
        g = m.group()
        lead = len(g) - len(g.lstrip())
        body = g.strip()
        if not body:
            continue
        s = m.start() + lead
        blocks.append((s, s + len(body)))

    spans, cur_s, cur_e = [], None, None
    for s, e in blocks:
        if e - s > max_size:                       # длинный абзац — режем окном
            if cur_s is not None:
                spans.append((cur_s, cur_e)); cur_s = cur_e = None
            for c in chunk_chars(text[s:e], max_size, overlap):
                spans.append((s + c["start"], s + c["end"]))
            continue
        if cur_s is not None and e - cur_s > max_size:   # не влезает — закрываем чанк
            spans.append((cur_s, cur_e)); cur_s = cur_e = None
        if cur_s is None:
            cur_s = s
        cur_e = e
    if cur_s is not None:
        spans.append((cur_s, cur_e))

    return [{"text": text[s:e], "start": s, "end": e} for s, e in spans]


def make_chunks(text, strategy="paragraph", size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if strategy == "char":
        return chunk_chars(text, size, overlap)
    return chunk_paragraphs(text, size, overlap)


# ********************* ЭМБЕДДИНГИ *********************

def embed_batch(texts):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_one(text):
    return embed_batch([text])[0]


# ********************* ЛЕКСИЧЕСКИЙ ПОИСК (BM25) *********************

def _tokenize(text):
    """Токенизация для BM25: слова из букв/цифр (включая кириллицу), lower."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def fetch_corpus(source_filter=None):
    """Возвращает все чанки коллекции (id, text, meta) — для BM25."""
    where = {"source": source_filter} if source_filter else None
    got = collection.get(where=where, include=["documents", "metadatas"])
    return list(zip(got["ids"], got["documents"], got["metadatas"]))


# Кэш BM25-индекса, чтобы не пересобирать его на каждый запрос.
# Инвалидируется явно в /upload и /clear (см. _invalidate_bm25).
_bm25_cache = {}   # source_filter|"__all__" -> (corpus, bm25)


def _invalidate_bm25():
    _bm25_cache.clear()


def get_bm25(source_filter=None):
    """Возвращает (corpus, bm25) для источника, строя индекс лениво и кэшируя."""
    key = source_filter or "__all__"
    if key not in _bm25_cache:
        corpus = fetch_corpus(source_filter)
        bm25 = BM25Okapi([_tokenize(doc) for _, doc, _ in corpus]) if corpus else None
        _bm25_cache[key] = (corpus, bm25)
    return _bm25_cache[key]


def bm25_search(query, corpus, bm25, top_n):
    """Лексический BM25-поиск. Возвращает [(id, score), ...] по убыванию."""
    if not corpus or bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return [(c[0], float(s)) for c, s in ranked[:top_n] if s > 0]


# ********************* RRF-СЛИЯНИЕ *********************

def rrf_fuse(rank_lists, k=RRF_K):
    """
    Reciprocal Rank Fusion. rank_lists — список списков id (каждый уже
    отсортирован по релевантности своего метода). Возвращает {id: score}.
    """
    scores = {}
    for ids in rank_lists:
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ********************* ПОИСК (dense / hybrid) *********************

def retrieve(question, top_k=TOP_K, hybrid=True, source_filter=None):
    """
    Возвращает список источников с метаданными:
      [{id, source, chunk_index, char_start, char_end, similarity,
        bm25_score, rrf_score, methods, text}]
    """
    q_vec = embed_one(question)
    where = {"source": source_filter} if source_filter else None

    # --- DENSE (Chroma) ---
    dres = collection.query(query_embeddings=[q_vec],
                            n_results=CANDIDATE_POOL, where=where,
                            include=["documents", "metadatas", "distances"])
    dense_ids   = dres["ids"][0]
    dense_docs  = dict(zip(dres["ids"][0], dres["documents"][0]))
    dense_meta  = dict(zip(dres["ids"][0], dres["metadatas"][0]))
    # cosine distance → similarity, с клипом отрицательных значений в 0
    dense_sim   = {i: max(0.0, 1.0 - d)
                   for i, d in zip(dres["ids"][0], dres["distances"][0])}

    if not hybrid:
        chosen = dense_ids[:top_k]
        bm25_scores, rrf_scores = {}, {}
        lex_ids = []
    else:
        # --- LEXICAL (BM25, кэшированный индекс) ---
        corpus, bm25 = get_bm25(source_filter)
        lex = bm25_search(question, corpus, bm25, CANDIDATE_POOL)
        lex_ids = [i for i, _ in lex]
        bm25_scores = dict(lex)
        # тексты/метаданные лексических кандидатов (могут не пересекаться с dense)
        corpus_map = {cid: (doc, meta) for cid, doc, meta in corpus}
        for cid in lex_ids:
            if cid not in dense_docs and cid in corpus_map:
                dense_docs[cid] = corpus_map[cid][0]
                dense_meta[cid] = corpus_map[cid][1]
        # --- RRF ---
        rrf_scores = rrf_fuse([dense_ids, lex_ids])
        chosen = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    sources = []
    for rank, cid in enumerate(chosen, start=1):
        meta = dense_meta.get(cid, {})
        methods = []
        if cid in dense_sim:    methods.append("dense")
        if cid in bm25_scores:  methods.append("lexical")
        sources.append({
            "rank": rank,
            "id": cid,
            "source": meta.get("source", "?"),
            "chunk_index": meta.get("chunk_index"),
            "char_start": meta.get("char_start"),
            "char_end": meta.get("char_end"),
            "similarity": (round(dense_sim[cid], 3) if cid in dense_sim else None),
            "bm25_score": round(bm25_scores.get(cid, 0.0), 3),
            "rrf_score": round(rrf_scores.get(cid, 0.0), 4) if hybrid else None,
            "methods": methods,
            "text": dense_docs.get(cid, ""),
        })
    return sources


# ********************* ПРОМПТ + ГЕНЕРАЦИЯ *********************

PROMPT_TEMPLATE = """Ты — ассистент, отвечающий на вопросы СТРОГО на основе контекста ниже.
Если в контексте нет ответа — честно скажи «В предоставленных документах ответа нет».
Не выдумывай факты и не используй внешние знания.
Ссылайся на использованные фрагменты в квадратных скобках: [#1], [#2] и т.д.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ (с ссылками на фрагменты):"""


def answer_question(question, top_k=TOP_K, hybrid=True, source_filter=None):
    sources = retrieve(question, top_k=top_k, hybrid=hybrid,
                       source_filter=source_filter)
    if not sources:
        return "В предоставленных документах ответа нет.", []

    context = "\n\n---\n\n".join(
        f"[#{s['rank']}] (источник: {s['source']}, фрагмент {s['chunk_index']})\n{s['text']}"
        for s in sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    msg = resp.choices[0].message
    answer = msg.content or getattr(msg, "reasoning_content", None) or "[пустой ответ]"
    return answer, sources


# ********************* МАРШРУТЫ API *********************

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    f = request.files["file"]
    strategy = request.form.get("strategy", "paragraph")
    try:
        size = int(request.form.get("chunk_size", CHUNK_SIZE))
    except ValueError:
        size = CHUNK_SIZE

    try:
        text = extract_text(f.filename, f.read())
    except Exception as e:
        return jsonify({"error": f"Не удалось прочитать файл: {e}"}), 400
    if not text.strip():
        return jsonify({"error": "Файл пустой или текст не извлёкся"}), 400

    source = f.filename
    chunks = make_chunks(text, strategy=strategy, size=size, overlap=CHUNK_OVERLAP)
    if not chunks:
        return jsonify({"error": "Не удалось разбить на чанки"}), 400

    embeddings = embed_batch([c["text"] for c in chunks])

    # удаляем старые записи этого источника (повторная загрузка)
    existing = collection.get(where={"source": source})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i,
                  "char_start": c["start"], "char_end": c["end"],
                  "length": len(c["text"]), "strategy": strategy}
                 for i, c in enumerate(chunks)]
    collection.add(ids=ids, documents=[c["text"] for c in chunks],
                   metadatas=metadatas, embeddings=embeddings)
    _invalidate_bm25()   # корпус изменился — сбрасываем кэш лексического индекса

    return jsonify({"source": source, "chunks": len(chunks),
                    "strategy": strategy, "total_in_db": collection.count()})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Пустой вопрос"}), 400
    if collection.count() == 0:
        return jsonify({"error": "База пуста — загрузите документ"}), 400

    top_k = int(data.get("top_k", TOP_K))
    hybrid = bool(data.get("hybrid", True))
    source_filter = data.get("source") or None

    answer, sources = answer_question(question, top_k=top_k, hybrid=hybrid,
                                      source_filter=source_filter)
    return jsonify({"answer": answer, "sources": sources,
                    "mode": "hybrid (dense+BM25/RRF)" if hybrid else "dense"})


@app.route("/sources", methods=["GET"])
def sources():
    n = collection.count()
    if n == 0:
        return jsonify({"sources": [], "total": 0})
    sample = collection.get(include=["metadatas"])
    counts = {}
    for meta in sample["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return jsonify({"sources": [{"name": k, "chunks": v} for k, v in counts.items()],
                    "total": n})


@app.route("/clear", methods=["POST"])
def clear():
    global collection
    chroma.delete_collection(COLLECTION)
    collection = chroma.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"})
    _invalidate_bm25()
    return jsonify({"status": "ok"})


# ********************* HTML / JS ФРОНТЕНД *********************

HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"><title>RAG App+ — лекция 11 (доработка)</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root{--bg:#f1f5f9;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b;
    --primary:#3b82f6;--green:#22c55e;--purple:#8b5cf6}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  header{background:var(--card);border-bottom:1px solid var(--border);padding:14px 24px;
    display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  header h1{font-size:18px;font-weight:700;flex:1}
  .badge{font-size:12px;color:var(--muted);padding:3px 10px;border-radius:20px;background:#f1f5f9}
  .layout{max-width:1150px;margin:0 auto;padding:18px 16px;display:grid;
    grid-template-columns:1fr 300px;gap:16px}
  .panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
  .panel h2{font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;
    letter-spacing:.4px;margin-bottom:10px}
  .dropzone{border:2px dashed var(--border);border-radius:10px;padding:18px;text-align:center;
    cursor:pointer;transition:all .2s;background:#fafafa}
  .dropzone:hover,.dropzone.over{border-color:var(--primary);background:#eff6ff}
  .dropzone p{color:var(--muted);font-size:13px}.dropzone strong{color:var(--text)}
  .settings{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:12px;font-size:13px}
  .settings label{display:flex;align-items:center;gap:6px;color:var(--muted)}
  .settings select,.settings input[type=number]{border:1px solid var(--border);border-radius:6px;
    padding:5px 8px;font-size:13px;background:var(--bg)}
  #chat{display:flex;flex-direction:column;gap:10px;min-height:200px;max-height:480px;
    overflow-y:auto;padding:4px}
  .msg{padding:10px 14px;border-radius:10px;line-height:1.6;font-size:14px}
  .msg.user{align-self:flex-end;background:var(--primary);color:#fff;max-width:78%}
  .msg.bot{align-self:flex-start;background:#f8fafc;border:1px solid var(--border);max-width:90%;white-space:pre-wrap}
  .msg.error{align-self:center;background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;font-size:13px}
  .cite{background:#ede9fe;color:#6d28d9;border-radius:4px;padding:0 4px;font-weight:600;cursor:pointer}
  .mode-tag{font-size:11px;color:var(--purple);margin-top:6px}
  .sources{margin-top:8px;border-top:1px dashed var(--border);padding-top:8px}
  .sources-toggle{font-size:12px;color:var(--purple);cursor:pointer;user-select:none}
  .sources-toggle:hover{text-decoration:underline}
  .sources-list{display:none;margin-top:8px;flex-direction:column;gap:6px}.sources-list.open{display:flex}
  .source-item{background:#faf5ff;border:1px solid #e9d5ff;border-radius:6px;padding:8px 10px;font-size:12px}
  .source-item .meta{color:var(--purple);font-weight:700;margin-bottom:4px;display:flex;
    gap:6px;flex-wrap:wrap;align-items:center}
  .source-item .text{color:var(--text);line-height:1.55;max-height:90px;overflow-y:auto;white-space:pre-wrap}
  .mbadge{font-size:10px;padding:1px 6px;border-radius:10px;font-weight:700}
  .m-dense{background:#dbeafe;color:#1d4ed8}.m-lex{background:#dcfce7;color:#15803d}
  .m-both{background:#f3e8ff;color:#7c3aed}
  .ask-row{display:flex;gap:8px;margin-top:12px}
  .ask-row input{flex:1;border:1px solid var(--border);border-radius:8px;padding:9px 12px;
    font-size:14px;font-family:inherit;outline:none}
  .ask-row input:focus{border-color:var(--primary)}
  .btn{padding:9px 16px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer}
  .btn:disabled{opacity:.45;cursor:not-allowed}.btn-primary{background:var(--primary);color:#fff}
  .btn-clear{background:#fee2e2;color:#b91c1c}.btn:hover:not(:disabled){opacity:.85}
  .src-item{background:#f8fafc;border:1px solid var(--border);border-radius:6px;padding:8px 10px;font-size:12px}
  .src-item b{color:var(--text)}.src-item span{color:var(--muted);font-size:11px}
  .empty{color:var(--muted);font-size:12px;font-style:italic;padding:6px}
  .toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1e293b;color:#fff;
    padding:9px 18px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none}
  .toast.show{opacity:1}
  .spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--border);
    border-top-color:var(--primary);border-radius:50%;animation:spin .7s linear infinite;
    vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  @media (max-width:860px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <span style="font-size:22px">📚</span>
  <h1>RAG App+ <span style="font-weight:400;color:var(--muted);font-size:13px">(лекция 11, доработка)</span></h1>
  <span class="badge" id="backend-badge">hybrid search · мультиформат · метаданные</span>
</header>

<div class="layout">
  <main style="display:flex;flex-direction:column;gap:14px">
    <div class="panel">
      <h2>📄 Загрузить документ (.txt .md .pdf .docx)</h2>
      <div class="dropzone" id="dropzone">
        <p><strong>Перетащите файл сюда</strong> или <a href="#" id="pick">выберите</a></p>
        <input type="file" id="file-input" accept=".txt,.md,.pdf,.docx" hidden>
      </div>
      <div class="settings">
        <label>Чанкинг:
          <select id="strategy"><option value="paragraph">по абзацам</option>
            <option value="char">посимвольно</option></select></label>
        <label>Размер чанка:
          <input type="number" id="chunk_size" value="500" min="100" max="2000" step="50" style="width:80px"></label>
      </div>
    </div>

    <div class="panel">
      <h2>💬 Чат с документами</h2>
      <div id="chat">
        <div class="msg bot">Загрузите документ и задайте вопрос. Отвечаю только по содержимому файлов, со ссылками на фрагменты.</div>
      </div>
      <div class="settings" style="margin:12px 0 0">
        <label><input type="checkbox" id="hybrid" checked> Hybrid (dense+BM25)</label>
        <label>top-k: <input type="number" id="top_k" value="3" min="1" max="8" style="width:60px"></label>
        <label>Источник:
          <select id="source-filter"><option value="">все</option></select></label>
      </div>
      <div class="ask-row">
        <input type="text" id="question" placeholder="Введите вопрос...">
        <button class="btn btn-primary" id="ask-btn">Спросить</button>
      </div>
    </div>
  </main>

  <aside style="display:flex;flex-direction:column;gap:14px">
    <div class="panel">
      <h2>📚 Загружено</h2>
      <div id="sources-list" class="src-list"></div>
      <button class="btn btn-clear" id="clear-btn" style="margin-top:12px;width:100%;font-size:12px">🗑 Очистить базу</button>
    </div>
    <div class="panel" style="font-size:12px;color:var(--muted);line-height:1.6">
      <h2>ℹ️ Как читать источники</h2>
      <div><span class="mbadge m-dense">🔵 смысл</span> — нашёл семантический (dense) поиск<br>
      <span class="mbadge m-lex">🟢 слова</span> — нашёл лексический BM25<br>
      <span class="mbadge m-both">🟣 оба</span> — подтверждён обоими (RRF поднимает выше)</div>
    </div>
  </aside>
</div>

<div class="toast" id="toast"></div>
<script>
const dropzone=document.getElementById('dropzone'),fileInput=document.getElementById('file-input'),
  chat=document.getElementById('chat'),question=document.getElementById('question'),
  askBtn=document.getElementById('ask-btn'),srcList=document.getElementById('sources-list'),
  srcFilter=document.getElementById('source-filter');
function escHtml(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function showToast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2500)}
function addMsg(text,cls){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}

dropzone.addEventListener('click',()=>fileInput.click());
document.getElementById('pick').addEventListener('click',e=>{e.preventDefault();fileInput.click()});
fileInput.addEventListener('change',e=>{if(e.target.files[0])uploadFile(e.target.files[0])});
['dragenter','dragover'].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.classList.add('over')}));
['dragleave','drop'].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.classList.remove('over')}));
dropzone.addEventListener('drop',e=>{if(e.dataTransfer.files[0])uploadFile(e.dataTransfer.files[0])});

async function uploadFile(file){
  const form=new FormData();form.append('file',file);
  form.append('strategy',document.getElementById('strategy').value);
  form.append('chunk_size',document.getElementById('chunk_size').value);
  const note=addMsg(`⏳ Индексирую "${file.name}"...`,'bot');
  try{const r=await fetch('/upload',{method:'POST',body:form});const d=await r.json();
    if(!r.ok||d.error){note.className='msg error';note.textContent='⚠️ '+(d.error||r.status);return;}
    note.textContent=`✅ "${d.source}" — ${d.chunks} чанков (${d.strategy}). Всего в базе: ${d.total_in_db}.`;
    refreshSources();
  }catch(e){note.className='msg error';note.textContent='⚠️ '+e.message;}
}

async function refreshSources(){
  try{const d=await(await fetch('/sources')).json();
    if(!d.sources.length){srcList.innerHTML='<div class="empty">Документов пока нет</div>';
      srcFilter.innerHTML='<option value="">все</option>';return;}
    srcList.innerHTML=d.sources.map(s=>`<div class="src-item"><b>${escHtml(s.name)}</b><br><span>${s.chunks} чанков</span></div>`).join('');
    srcFilter.innerHTML='<option value="">все</option>'+d.sources.map(s=>`<option value="${escHtml(s.name)}">${escHtml(s.name)}</option>`).join('');
  }catch(e){}
}

function renderCitations(text){
  return escHtml(text).replace(/\[#(\d+)\]/g,'<span class="cite" data-r="$1">[#$1]</span>');
}

async function sendQuestion(){
  const q=question.value.trim();if(!q)return;
  addMsg(q,'user');question.value='';askBtn.disabled=true;
  const thinking=addMsg('','bot');thinking.innerHTML='<span class="spinner"></span>Ищу и думаю...';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,top_k:parseInt(document.getElementById('top_k').value),
        hybrid:document.getElementById('hybrid').checked,source:srcFilter.value})});
    const d=await r.json();
    if(!r.ok||d.error){thinking.className='msg error';thinking.textContent='⚠️ '+(d.error||r.status);return;}
    thinking.innerHTML='';
    const ans=document.createElement('div');ans.innerHTML=renderCitations(d.answer);thinking.appendChild(ans);
    const mode=document.createElement('div');mode.className='mode-tag';mode.textContent='🔎 режим: '+d.mode;
    thinking.appendChild(mode);
    if(d.sources&&d.sources.length){
      const wrap=document.createElement('div');wrap.className='sources';
      const toggle=document.createElement('div');toggle.className='sources-toggle';
      toggle.textContent=`▶ Показать источники (${d.sources.length})`;
      const list=document.createElement('div');list.className='sources-list';
      d.sources.forEach(s=>{
        const both=s.methods.length>1;
        const badge=both?'<span class="mbadge m-both">🟣 оба</span>':
          s.methods.includes('lexical')?'<span class="mbadge m-lex">🟢 слова</span>':
          '<span class="mbadge m-dense">🔵 смысл</span>';
        const it=document.createElement('div');it.className='source-item';it.id='src-'+s.rank;
        it.innerHTML=`<div class="meta">#${s.rank} ${badge}
          <span>${escHtml(s.source)} · чанк ${s.chunk_index} · симв. ${s.char_start}–${s.char_end}</span>
          <span>${[s.similarity!=null?'cos='+s.similarity:'',s.bm25_score?'bm25='+s.bm25_score:'',s.rrf_score!=null?'rrf='+s.rrf_score:''].filter(Boolean).join(' · ')}</span></div>
          <div class="text">${escHtml(s.text)}</div>`;
        list.appendChild(it);
      });
      toggle.addEventListener('click',()=>{const o=list.classList.toggle('open');
        toggle.textContent=(o?'▼ Скрыть':'▶ Показать')+` источники (${d.sources.length})`});
      wrap.appendChild(toggle);wrap.appendChild(list);thinking.appendChild(wrap);
      // клик по цитате [#n] раскрывает и подсвечивает источник
      thinking.querySelectorAll('.cite').forEach(c=>c.addEventListener('click',()=>{
        list.classList.add('open');toggle.textContent='▼ Скрыть источники ('+d.sources.length+')';
        const el=document.getElementById('src-'+c.dataset.r);
        if(el){el.scrollIntoView({block:'nearest'});el.style.outline='2px solid #8b5cf6';
          setTimeout(()=>el.style.outline='',1500);}
      }));
    }
  }catch(e){thinking.className='msg error';thinking.textContent='⚠️ '+e.message;}
  finally{askBtn.disabled=false;question.focus();}
}
askBtn.addEventListener('click',sendQuestion);
question.addEventListener('keydown',e=>{if(e.key==='Enter')sendQuestion()});
document.getElementById('clear-btn').addEventListener('click',async()=>{
  if(!confirm('Удалить все документы?'))return;
  await fetch('/clear',{method:'POST'});refreshSources();showToast('База очищена');
});
refreshSources();
</script>
</body>
</html>"""


# ********************* ТОЧКА ВХОДА *********************

if __name__ == "__main__":
    print("=" * 70)
    print(f"  📚 RAG App+ : http://localhost:{PORT}")
    print(f"  Backend: {BASE_URL}")
    print(f"  Embedding: {EMBEDDING_MODEL}   LLM: {LLM_MODEL}")
    print(f"  БД: {CHROMA_DIR}  (записей: {collection.count()})")
    print("  Возможности: hybrid (dense+BM25/RRF), чанкинг по абзацам,")
    print("               PDF/DOCX, метаданные, мультидок, инлайн-цитаты")
    print("=" * 70 + "\n")
    app.run(debug=False, port=PORT, threaded=True)
