# 悟空法律通

> 東吳大學 生成式人工智慧與大語言模型 期末專案

一個基於 RAG (Retrieval-Augmented Generation) 架構的中文法律問答系統，結合了向量檢索和大語言模型微調技術，提供準確的法律諮詢服務。

## 系統架構

### 核心組件

1. **RAG 檢索系統** (`rag.py`) - 建立法規向量資料庫
2. **模型微調** (`bart_unsloth.py`) - 兩階段微調流程
3. **主程式** (`main.py`) - 整合 RAG 和 LLM 的問答系統
4. **推理程式** (`run_inference.py`) - 純模型推理介面 (不含 RAG)

### 技術特色

- **向量檢索**: 使用 BAAI/bge-m3 Embedding 模型和 ChromaDB
- **模型微調**: 基於 Llama-3.2-11B-Vision-Instruct 進行兩階段微調
- **RAG 整合**: 結合檢索到的法條內容生成準確回答

## 使用方法

### 1. 建立向量資料庫

首先需要處理法規 JSON 檔案並建立向量索引：

```bash
python rag.py
```

這會：

- 處理 `Law-json/` 目錄下的法規檔案
- 建立 ChromaDB 向量資料庫
- 儲存在 `chroma_db/` 目錄

### 2. 模型微調 (可選)

如果需要重新訓練模型：

```bash
python bart_unsloth.py
```

微調流程包含兩個階段：

1. **階段一**: 使用 `informal.jsonl` 進行非正式語言微調
2. **階段二**: 使用 `bart.jsonl` 進行角色特定微調

### 3. 啟動問答系統

```bash
python main.py
```

系統會載入：

- 微調後的語言模型
- 向量檢索資料庫
- 提供互動式問答介面

### 4. 純模型推理

如果只想使用微調模型而不使用 RAG：

```bash
python run_inference.py
```

## 資料格式

### 法規資料 (Law-json/)

- 支援中華民國法規 JSON 格式
- 包含法規名稱、條文內容、法規類別等資訊
- 自動過濾廢止法規

### 訓練資料格式

```json
{
  "prompt": "法律問題",
  "completion": "回答"
}
```

## 系統特色

### RAG 檢索

- 使用語意相似度檢索相關法條
- 過濾「廢止法規」，確保資訊時效性

### 回答風格

- 採用悟空角色設定，語調親切活潑
- 提供法條引用和來源資訊
- 結合法律專業性與易懂性

## 檔案結構

```
model/
├── Law-json/               # 法規 JSON 檔案
├── chroma_db/              # 向量資料庫
├── bart_finetuned/         # 微調後模型
├── informal_finetuned/     # 第一階段微調模型
├── main.py                 # 主要問答系統
├── rag.py                  # 向量資料庫建立
├── bart_unsloth.py         # 模型微調程式
├── run_inference.py        # 純推理介面
├── bart.jsonl              # 悟空風格訓練資料
├── informal.jsonl          # 非正式語言訓練資料
└── README.md               # 說明文件
```

## 使用展示

![image](/Screenshot.png)

## 注意事項

- 本系統僅供法律資訊參考，不構成正式法律建議
- 重要法律問題請諮詢專業律師
- 定期更新法規資料以確保時效性
