# Sprite Dedupe (Alpha-CC + pHash)

以 Alpha-Connected Components + 多通道 pHash 做 spritesheet/散圖分群與重複檢出，提供 PyQt5 圖形介面。

## 安裝

```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS/Linux
source .venv/bin/activate && pip install -r requirements.txt