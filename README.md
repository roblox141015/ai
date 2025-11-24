# KB Chat - 快速啟動

## 簡介
這是一個範例的知識庫聊天系統，使用 Flask + FAISS (本地) + OpenAI Embeddings/Chat API，前端使用 React (Vite)。

## 啟動步驟

1) 設定環境變數
```
export OPENAI_API_KEY="你的_api_key"
export ADMIN_PASSWORD="你要的管理密碼"
```

2) 後端
```
cd backend
python -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
後端會跑在 http://127.0.0.1:8000

3) 前端
```
cd frontend
npm install
npm run dev
```
或把前端 build 放到 backend/dist 作為靜態檔。

## 注意事項
- EMBED_DIM 必須與你使用的 embedding model 維度一致。
- 這個範例為教學用途，實務上請加強認證與資料庫儲存。
