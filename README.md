
# LLM-RAG-Agent-Hub
📖 專案簡介 (Introduction)
LLM-RAG-Agent-Hub 是一個以系統工程視角 (System Engineering Perspective) 建構的端到端概念驗證專案 (PoC)。

本專案整合了 LLM、RAG、Milvus、Elasticsearch 與 Multi-Agent，其核心目的不僅是展示模型生成的準確度，更在於驗證系統架構層面的議題：

架構穩定性：在高併發場景下，Retrieval 層與 Orchestration 層的吞吐量表現。

資源邊界：比較 I/O-bound (Mock) 與 CPU-bound (Real Local LLM) 場景下的系統行為差異。

決策品質控制：透過 AI Judge 與 Feedback Loop 機制，確保 Agent 輸出的合規性。

📦 核心應用場景 (Core Scenarios)
本專案實作了三個主要場景，展示了從資料檢索到自動化決策的完整能力：

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

## 🏗️ 架構概觀

專案採用簡化的三層式架構：

### 1. API 層 (`app/api`)
使用 FastAPI 定義各種 REST API：
* `/health`：健康檢查。
* `/prompt`：一般 LLM 推論（Prompt API）。
* `/embed`：Embedding 服務。
* `/ingest/docs`：文件前處理，**同步寫入** Milvus 與 Elasticsearch。
* `/rag/ask`：支援 **Vector / Hybrid** 模式切換的 RAG 問答。
* `/rag/graph_ask`：簡化版 GraphRAG 問答。
* `/recommend/news`：新聞推薦。
* `/agent/run`：多 Agent 數據分析流程。

### 2. Service 層 (`app/services`)
* `prompt_service`：管理不同 domain（金融 / 廣告 / 一般）的 system prompt 與模板。
* `ingest_service`：文件解析、清洗、chunking，並負責**雙寫入 (Dual-Write)** 至向量庫與搜尋引擎。
* `rag_service`：封裝 RAG 流程（混合檢索 → RRF 排序 → 資料回補 → 建 context → 組 Prompt → 呼叫 LLM）。
* `agent_service`：定義 `BaseAgent`、各專用 Agent 與 Orchestrator 流程，包含 AI Judge 的評分邏輯。

### 3. Infra / Core 層
* `app/vectorstore/milvus_client.py`：負責 Milvus 連線與向量操作。
* **`app/utils/es_client.py`**：負責 Elasticsearch 連線、BM25 檢索與 mget 批量回補。
* `app/llm/llm_client.py`：統一封裝 LLM 介面，支援雲端 (OpenAI) 與本地 (Ollama/Qwen2.5)。
* `app/core/config.py`：讀取 `.env` 並集中管理設定。

---

## 🔄 資料流程 (以金融 RAG 為例)

1.  **資料寫入**：使用 `/ingest/docs` 上傳文件。
2.  **Ingest Service**：
    * 解析文本、清洗噪音、Chunking。
    * 呼叫 Embedding 模型產生向量。
    * **寫入 Milvus** (儲存 Vector + Metadata)。
    * **寫入 Elasticsearch** (儲存 Text + Keyword Index)。
3.  **RAG 問答**：使用 `/rag/ask` 發問。
    * 若啟用 `use_hybrid: true`：
        * 同時進行 **Milvus 向量檢索** 與 **ES 關鍵字檢索**。
        * 執行 **RRF 演算法**融合排名。
        * 自動回補缺失的文本片段。
    * 組合 Context + Prompt。
    * LLM 生成最終回答。

---

## 🔌 主要 API 一覽

詳細欄位請參考 `http://localhost:8000/docs` (Swagger UI)。

* `POST /embed/`
    * Input: `{ "texts": [...], "collection": "docs|news", "store": true }`
    * 用途: 呼叫 Embedding 模型並選擇性寫入資料庫。

* `POST /ingest/docs/`
    * Input: `{ "file_paths": [...] }`
    * 用途: 處理文件並同步寫入 Milvus 與 ES。

* `POST /rag/ask/`
    * Input: `{ "question": "...", "use_hybrid": true }`
    * 用途: 透過混合檢索進行問答。

* `POST /agent/run/`
    * Input: `{ "task": "分析 Q3 廣告成效" }`
    * 用途: 觸發 Data -> Analysis -> Optimization (w/ AI Judge) 的自動化流程。

---
   
    - 
## 🛠️ 環境需求

* Python 3.10+
* Docker (用於啟動 Milvus / Attu)
* (選用) Ollama 或其他本地 LLM 服務

---
🧪 內建壓測工具詳解 (Benchmark Module)
本專案附帶了一套完整的效能測試腳本 benchmark_rag.py，不需依賴外部工具 (如 JMeter) 即可進行系統級壓測。

1. 測試什麼指標？ (Metrics)
P95 / P99 Latency (長尾延遲)：

排除極端值後的響應時間，比平均值更能反映真實用戶體驗。

用於觀察在高併發下是否發生「排隊阻塞 (Head-of-line blocking)」。

Throughput (TPS)：

系統每秒能處理的請求數量 (Transactions Per Second)。

用於找出系統的「飽和點 (Saturation Point)」。

CPU Usage (資源使用率)：

透過 psutil 監控測試期間的 CPU 平均負載。

用於判斷瓶頸是 CPU-bound (如模型推論) 還是 I/O-bound (如資料庫等待)。

Stdev (Jitter)：

延遲標準差，數值越低代表系統越穩定。

用於偵測系統是否進入「震盪區 (Thrashing)」。

2. 如何執行？ (Usage)
Bash
# 執行壓測腳本
python benchmark_rag.py
程式執行後將自動完成：

併發階梯測試：自動以 [1, 3, 5, 7, 10...] 等不同併發數進行壓力測試。

生成報表：在終端機輸出 SRE 等級的效能矩陣 (Performance Matrix)。

繪製圖表：自動生成 rag_benchmark_chart.png，視覺化 Latency 與 TPS 趨勢。

3. 測試模式切換
透過修改 .env 檔案可切換兩種測試場景：

Mock LLM 模式 (系統層驗證)：

設定 USE_MOCK_LLM=True。

用於驗證 FastAPI 架構與檢索層在高併發下的極限吞吐量 (I/O Bound)。

Real LLM 模式 (硬體邊界分析)：

設定 USE_MOCK_LLM=False 並指定模型 (如 qwen2.5:0.5b)。

用於觀察真實推論運算下的 CPU 飽和點與生成延遲 (CPU Bound)。

## 📊 效能壓測報告 (Performance Analysis)

為了定位系統瓶頸，本專案實作了 A/B 對照壓測，分別針對「架構韌性」與「硬體極限」進行分析。

### 1. 架構穩定性驗證 (Mock LLM 模式)
在隔離模型運算開銷後，觀察檢索層與 API 框架的併發表現。
![Mock LLM Performance](./assets/假LLM.png)
系統展現高度的可擴展性 (Scalability)，TPS 隨併發數線性成長。Hybrid Search 在高併發 (Conc > 20) 時開始顯現 RRF 排序帶來的運算成本。

### 2. 真實環境極限測試 (Qwen-0.5B CPU 推論)
驗證單機 CPU 在真實矩陣運算下的負載上限。
![Real LLM Performance](./assets/千問0.5.png)
系統進入 CPU 飽和區，出現明顯的「計算牆」。延遲受生成 Token 長度影響呈現隨機抖動，TPS 達到物理上限。



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
