from PyQt5 import QtCore
from PyQt5.QtCore import Qt

# ====== 預設參數 ======
VERSION = "v0.5.0"
PHASH_HAMMING_MAX_DEFAULT = 8
PHASH_HAMMING_MAX_INTRA_DEFAULT = 7
ALPHA_THR_DEFAULT = 1
MIN_AREA_DEFAULT = 400
MIN_SIZE_DEFAULT = 7
SPRITESHEET_MIN_SEGMENTS_DEFAULT = 3
SPRITESHEET_MIN_COVERAGE_DEFAULT = 0.03

ROT_DEG_STEP_DEFAULT = 15
INCLUDE_FLIP_TEST_DEFAULT = True

CANON_PAD_PRIMARY = 0.08
CANON_PAD_SECONDARY = 0.04

ASPECT_TOL = 0.1
USE_SHAPE_CHECK = True
PHASH_SHAPE_MAX = 6
SHAPE_ALPHA_THR = 1

USE_COLOR_CHECK = True
PHASH_COLOR_MAX = 5

USE_EDGE_CHECK = True
PHASH_EDGE_MAX  = 6

ROT_EARLYSTOP_SLACK = 2

CONTENT_AREA_TOL = 0.40
HGRAM_CHISQ_MAX  = 0.45
STRONG_PASS_MARGIN = 2

APPLY_ASPECT_FOR_INTRA = False
SINGLES_BUCKET_SHIFT = 10
SINGLES_GROUP_KEY = "__SINGLES__"

PHASH_HAMMING_MAX = getattr(__builtins__, "PHASH_HAMMING_MAX", 8)

LIGHT_QSS = """
* { font-family: "Segoe UI", "Microsoft JhengHei UI", Arial; font-size: 12px; color: #1f2937; }
QMainWindow { background: #f7f8fb; }
QToolBar { background: #ffffff; border: none; padding: 6px; }
QToolButton { padding: 6px 10px; border-radius: 8px; }
QToolButton:hover { background: #eef2ff; }

QAbstractScrollArea, QAbstractItemView, QListView, QTreeView, QTableView, QTableWidget {
  background: #f7f8fb;
}

QListWidget { background: #ffffff; border: 1px solid #e6e8f0; border-radius: 12px; padding: 6px; }
QTableWidget { background: #ffffff; border: 1px solid #e6e8f0; border-radius: 12px; gridline-color: #eef0f6; }
QHeaderView::section { background: #f3f4f6; border: none; padding: 6px; font-weight: 600; color: #374151; }

/* Splitter */
QSplitter::handle { background: #f3f4f6; border: 1px solid #e6e8f0; width: 8px; }
QSplitter::handle:hover { background: #e5e7eb; }

/* 一般按鈕（含彈窗） */
QPushButton { background: #4f46e5; color: white; border: none; padding: 8px 12px; border-radius: 10px; }
QPushButton:hover { background: #4338ca; }
QPushButton:disabled { background: #c7cbe6; color: #6b7280; }

/* 表格內的決策按鈕尺寸剛好、不會太大 */
QTableWidget QPushButton { min-width: 68px; min-height: 26px; padding: 4px 10px; border-radius: 14px; }

/* 狀態膠囊 */
QTableWidget QLabel[tag="badge"] { padding: 6px 10px; border-radius: 14px; }
QLabel[tag="badge"] { background-color: #e0e7ff; color: #3730a3; border-radius: 10px; padding: 2px 8px; }
QLabel[tag="badge"][state="pending"] { background-color: #e5e7eb; color: #374151; }
QLabel[tag="badge"][state="keep"]    { background-color: #dcfce7; color: #166534; }
QLabel[tag="badge"][state="both"]    { background-color: #dbeafe; color: #1e3a8a; }
QLabel[tag="badge"][state="skip"]    { background-color: #fee2e2; color: #991b1b; }

/* 進度條 */
QProgressBar { background: #e5e7eb; border: none; border-radius: 8px; height: 12px; }
QProgressBar::chunk { background: #4f46e5; border-radius: 8px; }

/* Dialog 也跟主題一致 */
QDialog, QMessageBox { background: #ffffff; border: 1px solid #e6e8f0; border-radius: 12px; }

/* 表格內客製小容器透明，避免白塊 */
QWidget#cellContainer { background: transparent; }

/* 讓三個主要區塊的外框更顯眼（左/右視圖同底色） */
QTableWidget#fileList,
QTreeWidget#pairTree,
BBoxGraphicsView#sheetView {
  border: 2px solid #5865F2;        /* 主色外框 */
  border-radius: 12px;
  background: #f7f8fb;              /* 與主介面一致的淺灰底 */
  gridline-color: transparent;
}

/* 滑過/聚焦時加粗+更亮 */
QTableWidget#fileList:hover,
QTreeWidget#pairTree:hover,
BBoxGraphicsView#sheetView:hover,
QTableWidget#fileList:focus,
QTreeWidget#pairTree:focus,
BBoxGraphicsView#sheetView:focus {
  border: 3px solid #5865F2;
  background: rgba(88,101,242,0.06);
}

/* 群組樹的列選取/滑過也加強 */
QTreeView#pairTree::item:hover { background: rgba(88,101,242,0.08); }
QTreeView#pairTree::item:selected { background: rgba(88,101,242,0.18); }
QTreeView#pairTree QHeaderView::section {
  border: none;
  border-bottom: 2px solid #5865F2;
  padding: 6px 8px;
}
QTreeView#pairTree QHeaderView::section {
  border: none;
  border-bottom: 2px solid #5865F2;
  padding: 6px 8px;
}

/* 讓所有縮圖（輸入清單、群組清單）都透明底 */
QLabel[role="thumb"] {
  background: transparent;
  border: none;
  border-radius: 0px;
  padding: 0px;
}
"""

DARK_QSS = """
* { color: #e5e7eb; font-family: "Segoe UI","Microsoft JhengHei UI",Arial; font-size: 12px; }
QMainWindow { background: #0b1220; }
QToolBar { background: #0b1220; border: none; padding: 6px; }
QToolButton { padding: 6px 10px; border-radius: 8px; color:#e5e7eb; }
QToolButton:hover { background: #111a2f; }

/* 內容區：深色，避免白塊 */
QAbstractScrollArea, QAbstractItemView, QListView, QTreeView, QTableView, QTableWidget { background: #0b1220; }
QTableWidget {
  border: 1px solid #1f2a44; border-radius: 12px;
  gridline-color: #1f2a44; alternate-background-color: #0f162b;
}
QTableWidget::item { background: #0b1220; }
QTableWidget::item:alternate { background: #0f162b; }
QTableWidget::item:selected { background: #172036; }
QHeaderView::section { background: #111a2f; color: #93a6c6; border: none; padding: 6px; }

QSplitter::handle { background: #0b1220; border: 1px solid #1f2a44; width: 8px; }
QSplitter::handle:hover { background: #111a2f; }

/* 一般按鈕（含彈窗） */
QPushButton {
  background: #6366f1; color: #ffffff; border: none; padding: 8px 12px; border-radius: 10px;
}
QPushButton:hover { background: #5458e3; }
QPushButton:disabled { background: #374151; color: #9ca3af; }

/* 表格裡的決策按鈕更小巧 */
QTableWidget QPushButton { min-width: 68px; min-height: 26px; padding: 4px 10px; border-radius: 14px; }

/* 狀態膠囊 */
QLabel[tag="badge"] { background: #111827; color: #cbd5e1; border-radius: 14px; padding: 4px 10px; }
QLabel[tag="badge"][state="keep"] { background: #064e3b; color: #a7f3d0; }
QLabel[tag="badge"][state="both"] { background: #0a1f44; color: #93c5fd; }
QLabel[tag="badge"][state="skip"], QLabel[tag="badge"][state="pending"] { background: #272b36; color: #9ca3af; }

/* 捲軸深色 */
QScrollBar:vertical, QScrollBar:horizontal { background: transparent; margin: 4px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #334155; border-radius: 6px; min-height: 24px; min-width: 24px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #475569; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; background:transparent; }

/* Dialog 也深色，避免白底與看不到字 */
QDialog, QMessageBox { background: #0b1220; border: 1px solid #1f2a44; border-radius: 12px; color: #e5e7eb; }

/* 表格內客製小容器透明，避免白塊 */
QWidget#cellContainer { background: transparent; }

/* pairTree 的 hover/selected 顏色（維持高對比） */
QTreeView#pairTree::item:hover { background: rgba(140,160,255,0.14); }
QTreeView#pairTree::item:selected { background: rgba(140,160,255,0.22); }
QTreeView#pairTree QHeaderView::section {
  border: none;
  border-bottom: 2px solid #A9B8FF;
  padding: 6px 8px;
}

/* 右側資訊面板卡片化 */
QWidget#infoPanel {
  background: #0f162b;
  border: 1px solid #1f2a44;
  border-radius: 12px;
}

/* 縮圖卡片化（右側直欄、群組橫向縮圖通用） */
QLabel[role="thumb"] {
  background: transparent;
  border: none;
  border-radius: 0px;
  padding: 0px;
}
QLabel[role="thumb"]:hover { border-color: #5865F2; background: #101c36; }

/* 三大主區塊（含左右視圖）— 深色降亮、底色一致 */
QTableWidget#fileList,
QTreeWidget#pairTree,
BBoxGraphicsView#sheetView {
  border: 1px solid #253354;
  border-radius: 12px;
  background: #0f162b;
  gridline-color: transparent;
}
QTableWidget#fileList:hover,
QTreeWidget#pairTree:hover,
BBoxGraphicsView#sheetView:hover,
QTableWidget#fileList:focus,
QTreeWidget#pairTree:focus,
BBoxGraphicsView#sheetView:focus {
  border: 2px solid #3a4b7a;
  background: #141e36;
}

/* 進度條（深色低飽和綠） */
QProgressBar { background: #1f2a44; border: none; border-radius: 8px; height: 12px; }
QProgressBar::chunk { background: #22c55e; border-radius: 8px; }
"""