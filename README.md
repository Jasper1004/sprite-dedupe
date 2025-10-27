# Sprite Dedupe (Alpha-CC + pHash)

精準的圖片重複檢測與 spritesheet 子圖分群工具  
使用 Alpha-Connected Components + 多通道 pHash 技術  
支援大量遊戲素材、自動分群與母圖 bbox 標記

![screenshot](docs/screenshot_light.PNG)

---

## 📦 下載與安裝

目前不提供 EXE 版本  
請透過 Python 執行原始碼方式使用：

### 1️⃣ 下載原始碼
**方式 A：透過 Git Clone**
```bash
git clone https://github.com/Jasper1004/sprite-dedupe.git
cd sprite-dedupe
```

**方式 B：手動下載**
> Code → Download ZIP → 解壓縮後進入資料夾

---

### 2️⃣ 建立虛擬環境
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

### 3️⃣ 安裝依賴套件
```bash
pip install -r requirements.txt
```

---

### 4️⃣ 執行圖形介面
```bash
python main.py
```

---

## ✨ 主要功能

| 功能 | 說明 |
|------|------|
| Spritesheet 自動分割 | 透過 Alpha CC 偵測子圖 bbox |
| 多通道 pHash 分群 | primary/secondary/U/V/alpha/edge channels |
| 深色/淺色主題切換 | UI 可切換 |
| 母圖標記對應位置 | 群組內顯示 bbox |
| 避免誤群組 | 雙階段特徵比對 |
| 結果可匯出 | JSON 格式 |

> 📌 至少兩張相似圖片才會生成一個群組  
> 📌 獨立圖片不顯示，避免干擾判讀

---

## 🖥️ 系統需求

| 項目 | 需求 |
|------|------|
| 作業系統 | Windows 10/11 (x64) |
| RAM | 建議 ≥ 8GB |
| Python | 3.10+ |

---

## 📁 專案結構

```text
sprite-dedupe/
├─ app/
│  ├─ __init__.py
│  ├─ constants.py
│  ├─ core/
│  │  ├─ alpha_cc.py      # Alpha Connected Components：子圖框選
│  │  ├─ phash.py         # 多通道 pHash，比對旋轉/翻轉等變化
│  │  └─ features.py      # 直方圖、卡方距離等特徵工具
│  ├─ utils/
│  │  ├─ atomic.py        # 原子化檔案寫入
│  │  └─ image_io.py      # 去白、影像輸出與處理
│  ├─ stores/
│  │  ├─ feature_store.py # 特徵快取
│  │  ├─ index_store.py   # 檔案索引管理
│  │  └─ logger.py        # 事件記錄
│  └─ ui/
│     ├─ widgets.py       # BBoxGraphicsView、ImageLabel
│     ├─ group_widget.py  # GroupResultsWidget：分群顯示
│     ├─ dialogs.py       # PairDecisionDialog
│     └─ main_window.py   # 主視窗與 UI 控制流程
├─ main.py                # 程式入口
├─ requirements.txt
└─ README.md
```

---

## 📘 使用教學

1️⃣ 加入 spritesheet 或散圖：  
→ 點 **新增圖片** 或 **新增資料夾**

2️⃣ 點 **開始處理**：  
→ 自動偵測與分群

3️⃣ 左側顯示分群結果，右側顯示：  
   ✅ 母圖  
   ✅ bbox 標記位置  

4️⃣ 可切換深色 / 淺色主題  

5️⃣ 可調整去白參數改善背景

---

## 📝 License

MIT License  
歡迎自由開發、修改與引用本專案。
