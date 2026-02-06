
# LLM-RAG-Agent-Hub
這是一個為了展示「**LLM + RAG + Milvus + Multi-Agent + FastAPI** 端到端能力」而設計的概念驗證專案
專案目前包含三個主要場景：

1. **金融知識 RAG 問答**
   - 處理 PDF / HTML / Word / Markdown 等文件
   - 前處理與 chunking → Embedding → 寫入 Milvus
   - 透過 `/rag/ask` 進行文件導向的問答
2. **金融新聞向量推薦雛形**
   - 使用假資料新聞集（title / content / tags / published_at）
   - 依照使用者偏好與最近查詢內容做向量相似度搜尋
   - 透過 `/recommend/news` 回傳 Top-K 推薦
3. **多 Agent 數據成效分析與優化建議**
   - DataAgent：數據預處理與指標運算（Pandas），具備輸入參數合法性校驗，阻絕無效任務。
   
   - AnalysisAgent：利用 LLM 提取數據洞察，最好與最壞的數據並解說。
   
   - AdOptimizationAgent：根據規則產生優化建議，具備 Self-Correction 機制。透過 Pydantic 進行輸出結構化驗證，若格式不符將自動觸發 Feedback Loop 重試，確保 100% 產出高品質、可程式化執行的建議。
   - 由 Orchestrator 透過 `/agent/run` 串成端到端流程

---

## 架構概觀

專案採用簡化的三層式架構：

- **API 層（`app/api`）**
  - 使用 FastAPI 定義各種 REST API：
    - `/health`：健康檢查
    - `/prompt`：一般 LLM 推論（Prompt API）
    - `/embed`：Embedding 服務
    - `/ingest/docs`：文件前處理與寫入 Milvus
    - `/rag/ask`、`/rag/graph_ask`：RAG / 簡化版 GraphRAG 問答
    - `/recommend/news`：新聞推薦
    - `/agent/run`：多 Agent 數據分析流程

- **Service 層（`app/services`）**
  - `prompt_service`：管理不同 domain（金融 / 廣告 / 一般）對應的 system prompt 與模板
  - `ingest_service`：文件解析、清洗、chunking 與批次寫入向量庫
  - `rag_service` / `graph_rag_service`：封裝 RAG 流程（檢索 → 建 context → 組 Prompt → 呼叫 LLM）
  - `recommend_service`：處理金融新聞 Embedding 與相似度搜尋
  - `agent_service`：定義 `BaseAgent`、各專用 Agent 與 Orchestrator 流程

- **Infra / Core 層**
  - `app/vectorstore/milvus_client.py`：負責連線 Milvus、建立 collection、寫入與查詢向量
  - `app/llm/llm_client.py`：統一封裝 LLM 介面，支援：
    - 雲端 LLM（OpenAI 相容 API）
    - 本地 LLM（例如透過 Ollama 呼叫 Qwen2.5:7b）
  - `app/storage/file_storage.py`：管理 demo 用文件與資料路徑
  - `app/utils/`：文字清洗、chunking 等工具函式
  - `app/core/config.py`：讀取 `.env` 並集中管理設定

資料流程（以金融 RAG 問答為例）：

1. 使用 `/ingest/docs` 上傳或指定文件來源（ HTML / Markdown）
2. `ingest_service`：
   - 解析文本、清洗噪音
   - 依設定做 chunking（含 overlap）
   - 呼叫 Embedding 模型產生向量
   - 寫入 Milvus 指定 collection
3. 使用 `/rag/ask` 發問：
   - 問題先做 Embedding
   - 在 Milvus 檢索 top-k 相似 chunk
   - `rag_service` 組合 context + Prompt
   - `llm_client` 呼叫 LLM 產生回答
   - 回傳答案與來源文件／chunk 資訊

---

## 主要 API 一覽

以下僅列出部分重點 API，詳細欄位可參考 `app/api` 內的 Pydantic schema：

- `GET /health/`
  - 用來確認服務啟動與依賴是否正常。

- `POST /prompt/`
  - Input：`{ "system_prompt": "...(optional)", "user_prompt": "...", "domain": "finance|ads|general" }`
  - 用途：單純 LLM 回答，不經過 RAG，可用於測試 LLM 狀態與不同 domain 的 Prompt 模板。

- `POST /embed/`
  - Input：`{ "texts": [...], "collection": "docs|news|custom", "store": true/false }`
  - 用途：呼叫 Embedding 模型，選擇性寫入 Milvus，作為之後檢索或推薦的基礎。

- `POST /ingest/docs/`
  - 用途：處理 PDF / Word / HTML / Markdown 文件：
    - 前處理 → chunking → embedding → 寫入 Milvus `docs` collection。

- `POST /rag/ask/`
  - 用途：標準 RAG 問答，從 Milvus 檢索相關 chunk，組合 context 後讓 LLM 回答。

- `POST /rag/graph_ask/`
  - 用途：示範簡化版 GraphRAG（例如利用 heading / 小節關係建立「鄰近節點」的 context）。

- `POST /recommend/news/`
  - 用途：根據使用者偏好與查詢內容，回傳向量相似度最高的新聞列表。

- `POST /agent/run/`
  - 用途：觸發多 Agent 流程：
    - `DataAgent` → `AnalysisAgent` → `AdOptimizationAgent`
    - 回傳整合後的分析報告與建議。
   
    - 
## 🛠️ 環境需求

* Python 3.10+
* Docker (用於啟動 Milvus / Attu)
* (選用) Ollama 或其他本地 LLM 服務

---

## 🚀 快速開始 (Local Demo)

> ⚠️ 下列步驟是示範用開發流程，實際環境請依系統調整。

建立虛擬環境並安裝套件：
---
並安裝套件：

```bash
pip install -r requirements.txt
複製環境設定檔：

bash
複製程式碼
cp .env.example .env
# 如要使用本地 Qwen2.5 + Ollama，保持 LLM_BACKEND=local
# 如要改用雲端 LLM，調整 LLM_BACKEND、LLM_API_BASE、LLM_API_KEY 等設定
啟動 Milvus

建議使用 docker-compose（可依官方教學啟動 Milvus / Attu）

.env 內的 MILVUS_HOST、MILVUS_PORT 需與實際設定一致

啟動本地 LLM（選用，本專案預設支援 Qwen2.5 via Ollama）：

bash
複製程式碼
ollama pull qwen2.5:7b
ollama serve
啟動 FastAPI：

bash
複製程式碼
uvicorn app.main:app --reload
# 或指定埠：
# uvicorn app.main:app --reload --port 8001
測試健康檢查：

bash
複製程式碼
curl http://localhost:8000/health/
或在瀏覽器開啟：

http://localhost:8000/docs 查看自動生成的 Swagger 介面



Copyright (c) 2025 Li Wei

All rights reserved.

本程式碼僅供個人作品集展示使用。
未經作者書面同意，不得以任何形式複製、修改、再發布或用於商業用途。
