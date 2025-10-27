# Sprite Dedupe (Alpha-CC + pHash)

精準的圖片重複檢測與 spritesheet 子圖分群工具  
使用 Alpha-Connected Components + 多通道 pHash 技術  
支援大量遊戲素材、自動分群與母圖 bbox 標記

![screenshot](docs/screenshot_light.PNG)

---

## 📦 下載 (Windows)

➡ https://github.com/Jasper1004/sprite-dedupe/releases  
下載 `SpriteDedupe_win64.zip` → 解壓縮 → 執行 `SpriteDedupe.exe`
**不需要安裝！**

---

## ✨ 主要功能

| 功能 | 說明 |
|------|------|
| Spritesheet 自動分割 | 透過 Alpha CC 偵測子圖 bbox |
| 多通道 pHash 分群 | primary/secondary/U/V/alpha/edge channels |
| 深色/淺色主題切換 | UI 可手動切換 |
| 同群預覽 | 顯示母圖 + bbox 的位置 |
| 高相似度篩選 | 雙階段特徵比對避免誤群組 |
| 結果可匯出 | JSON 格式，可作為後續 pipeline 資料 |

---

## 🖥️ 系統需求

| 項目 | 需求 |
|------|------|
| 作業系統 | Windows 10/11 (x64) |
| RAM | 建議 ≥ 8GB |
| Python 執行（只對開發者） | Python 3.10+ |

> 使用者直接執行 EXE 無需 Python。

---

## 📁 專案結構

```text
sprite-dedupe/
├─ app/
│  ├─ __init__.py
│  ├─ constants.py
│  ├─ core/
│  │  ├─ alpha_cc.py      # Alpha Connected Components（子圖框選）
│  │  ├─ phash.py         # 多通道 pHash（含旋轉/翻轉搜尋）
│  │  └─ features.py      # 直方圖/卡方距離等特徵工具
│  ├─ utils/
│  │  ├─ atomic.py        # 原子化寫入，避免中斷壞檔
│  │  └─ image_io.py      # QPixmap/QImage、white-key 去白、結果輸出
│  ├─ stores/
│  │  ├─ feature_store.py # 特徵快取（原子寫/載）
│  │  ├─ index_store.py   # 檔案索引/變更追蹤
│  │  └─ logger.py        # 操作/事件紀錄
│  └─ ui/
│     ├─ widgets.py       # BBoxGraphicsView、ImageLabel
│     ├─ group_widget.py  # GroupResultsWidget（群組清單 + 縮圖）
│     ├─ dialogs.py       # PairDecisionDialog
│     └─ main_window.py   # MainWindow（主題切換、流程控制）
├─ main.py                # 入口
├─ requirements.txt
└─ README.md


---

## 📘 使用教學

1. 點選 **新增圖片**，**新增資料夾**，加入 spritesheet 或 散圖
2. 點擊 **開始處理** → 自動偵測與分群
3. 左側為群組結果
4. 點選子圖 → 右側顯示母圖 + 位置標記
5. 可切換深色/淺色主題
6. 可調整去白參數（若啟用）

> 至少兩張相似圖片才會生成一個群組 ✅  
> 獨立圖片不會顯示（避免干擾分析）

---

## 🔧 安裝（開發者）

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py