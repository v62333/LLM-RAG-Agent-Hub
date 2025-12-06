# tests/test_rag_performance.py
import sys
import os

# 這兩行是為了讓 Python 找得到上一層的 app 資料夾
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 假設你的主程式在 app/main.py 或 app/rag_engine.py
# 請根據你實際的檔案名稱修改 import
# 例如：如果你的 RAG 邏輯寫在 app/service.py 裡的 query_rag 函式
# from app.service import query_rag 

def test_basic_retrieval():
    print("--- 開始測試 RAG 檢索 ---")
    question = "RAG 的全名是什麼？"
    
    # 這裡模擬呼叫你的 RAG (請換成真的呼叫)
    # answer = query_rag(question) 
    answer = "RAG 的全名是 Retrieval-Augmented Generation" # 假裝這是 AI 回傳的
    
    # 斷言測試
    assert "Retrieval-Augmented Generation" in answer
    print("✅ 測試通過：答案包含正確關鍵字")

if __name__ == "__main__":
    # 讓你可以直接用 python tests/test_rag_performance.py 執行它
    try:
        test_basic_retrieval()
    except AssertionError:
        print("❌ 測試失敗！")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
