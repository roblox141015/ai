from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import openai
from dotenv import load_dotenv
from faiss_store import FaissStore

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError('請先設定 OPENAI_API_KEY 環境變數')
openai.api_key = OPENAI_API_KEY

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)

EMBED_MODEL = 'text-embedding-3-small'  # 可依你的帳號選擇
EMBED_DIM = 1536  # text-embedding-3-small 維度為1536

# 初始化 FAISS store
store = FaissStore(dim=EMBED_DIM)

# 簡單 admin 密碼（實務請使用真正驗證）
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'changeme')

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    pw = data.get('password')
    if pw == ADMIN_PASSWORD:
        return jsonify({'ok': True})
    return jsonify({'ok': False}), 401

@app.route('/api/admin/add', methods=['POST'])
def admin_add():
    data = request.json
    pw = data.get('password')
    if pw != ADMIN_PASSWORD:
        return jsonify({'error': 'unauthorized'}), 401
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error':'empty'}), 400

    # 產生 embedding
    res = openai.Embeddings.create(model=EMBED_MODEL, input=text)
    emb = res['data'][0]['embedding']
    store.add([emb], [text])
    return jsonify({'ok': True})

@app.route('/api/admin/list', methods=['GET'])
def admin_list():
    docs = store.all_docs()
    return jsonify({'docs': docs})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error':'empty'}), 400

    # question embedding
    qres = openai.Embeddings.create(model=EMBED_MODEL, input=question)
    qemb = qres['data'][0]['embedding']
    hits = store.query(qemb, k=4)

    # 構建 system prompt + RAG context
    system = "你是一個根據知識庫回答使用者問題的助理。回答請使用繁體中文，並在回答後標示出使用到的知識（若有）。"
    joined = "\n\n".join([f"知識段落{i+1}: {h}" for i,h in enumerate(hits)])

    messages = [
        {"role":"system","content": system},
        {"role":"system","content": "知識庫內容:\n" + joined},
        {"role":"user","content": question}
    ]

    # 呼叫 ChatCompletion
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=messages,
        max_tokens=600,
        temperature=0.2
    )
    answer = resp['choices'][0]['message']['content']
    return jsonify({'answer': answer, 'sources': hits})

# 若要部署靜態前端，可把 build 放到 frontend/dist
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(port=8000, debug=True)
