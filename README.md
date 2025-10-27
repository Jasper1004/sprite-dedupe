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
| 深色/淺色主題切換 | UI 自動切換 |
| 自動去白 White-Key | 清除白底背景，子物件乾淨呈現 |
| 同群預覽 | 顯示母圖 + 所有 bbox 的位置 |
| 高相似度篩選 | 雙階段特徵比對避免誤群組 |
| 結果可匯出 | JSON 格式，可作為後續 pipeline 資料 |

---

## 🔍 適用情境

- 手機/網頁遊戲美術素材管理
- UI Icon 去重與版本檢查
- 圖庫清理、圖片壓縮前比對
- NFT/抽卡/素材大量去重

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

sprite-dedupe/
├─ app/
│ ├─ core/ # Alpha CC、pHash、特徵抽取
│ ├─ utils/ # QPixmap處理、atomic檔案寫入
│ ├─ stores/ # 特徵快取與索引儲存
│ └─ ui/ # PyQt 界面：MainWindow/GroupView/...
├─ main.py # 程式入口
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
> 孤立圖片不會顯示（避免干擾分析）

---

## 🔧 進階設定（開發者用）

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
