
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
   
   - OptimizationAgent：根據數據給予整體建議優化，內建 Self-Correction 自我修正機制。透過 Pydantic 實施結構化校驗，並結合 AI 審查員 (AI Judge) 針對建議的具體性與邏輯性進行 0-100 評分；若未達 80 分，系統將自動觸發 Feedback Loop 回傳錯誤原因並要求 LLM 重新生成，確保最終決策建議的高合規性與執行品質。
     
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
    - `DataAgent` → `AnalysisAgent` → `OptimizationAgent`
    - 回傳整合後的分析報告與建議。
   
    - 
## 🛠️ 環境需求

* Python 3.10+
* Docker (用於啟動 Milvus / Attu)
* (選用) Ollama 或其他本地 LLM 服務

---

🚀 快速開始 (Quick Start)
請按照以下步驟在本地環境啟動專案：

1. 環境配置
建立虛擬環境並安裝所有依賴套件（包含新增的 elasticsearch）：

Bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
2. 設定環境變數
複製範例檔並根據您的環境調整設定（務必檢查 ES 與 Milvus 的連線資訊）：

Bash
cp .env.example .env
💡 提示： 請確保 .env 中的 ES_HOST、ES_INDEX_PREFIX 與 MILVUS_HOST 設定正確。

3. 啟動基礎設施 (Infrastructure)
本專案依賴 Milvus (向量庫) 與 Elasticsearch (全文索引)。

啟動 Milvus：建議使用 docker-compose 啟動。

啟動 Elasticsearch：

確保 ES 服務運行於 http://localhost:9200。

Docker 快速啟動指令：

Bash
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.11.0
4. 啟動 LLM 服務 (選用)
本專案預設支援 Qwen2.5 via Ollama。若使用雲端模型請跳過此步並修改 .env 中的 LLM_BACKEND=cloud。

Bash
ollama pull qwen2.5:7b
ollama serve
5. 啟動 FastAPI 伺服器
Bash
uvicorn app.main:app --reload
6. 驗證服務狀態
您可以透過以下方式確認服務是否正常運行：

健康檢查：訪問 http://localhost:8000/health。

互動式 API 文檔：開啟瀏覽器進入 http://localhost:8000/docs (Swagger UI)。

🛠️ 開發提示 (Dev Tips)
資料同步寫入：當您呼叫 /ingest/docs 時，系統會自動完成「向量化並寫入 Milvus」以及「全文索引並寫入 Elasticsearch」的同步操作。

切換檢索模式：在 /rag/ask 的 Request Body 中切換 "use_hybrid": true 即可啟用 RRF 混合檢索。



Copyright (c) 2025 Li Wei

All rights reserved.

本程式碼僅供個人作品集展示使用。
未經作者書面同意，不得以任何形式複製、修改、再發布或用於商業用途。
