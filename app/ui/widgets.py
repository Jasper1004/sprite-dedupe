from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsRectItem

from ..utils.image_io import qpixmap_from_rgba
from ..constants import CANON_PAD_SECONDARY, SHAPE_ALPHA_THR, PHASH_SHAPE_MAX, ASPECT_TOL
from ..core.phash import phash_from_canon_alpha
from ..core.features import crop_aspect_ratio

class ImageLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(dict)
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.metadata = metadata
        self.setCursor(Qt.PointingHandCursor)
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.metadata)
        super().mousePressEvent(event)

class BBoxGraphicsView(QtWidgets.QGraphicsView):
    bboxClicked = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sheetView")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setBackgroundBrush(QtCore.Qt.transparent)
        self.setScene(self._scene)
        self._img_item = None
        self._rect_items = {}
        self.viewport().setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self._current_pix = None
        self._fit_to_width = False
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.auto_key_white = True
        self.key_white_threshold = 250
        self.key_white_ratio = 0.60

        self._pen_default = QtGui.QPen(QtGui.QColor(255, 0, 0)); self._pen_default.setWidth(5); self._pen_default.setCosmetic(True)
        self._pen_secondary = QtGui.QPen(QtGui.QColor(255, 149, 0)); self._pen_secondary.setWidth(6); self._pen_secondary.setCosmetic(True)
        self._pen_left = QtGui.QPen(QtGui.QColor(255, 0, 0)); self._pen_left.setWidth(6); self._pen_left.setCosmetic(True)
        self._pen_right = QtGui.QPen(QtGui.QColor(255, 149, 0)); self._pen_right.setWidth(6); self._pen_right.setCosmetic(True)
        self._pen_allboxes = QtGui.QPen(QtGui.QColor(0, 0, 0)); self._pen_allboxes.setWidth(5); self._pen_allboxes.setCosmetic(True)

    def clear(self):
        self._scene.clear()
        self._img_item = None
        self._rect_items.clear()
        self._current_pix = None
        self._fit_to_width = False
        self.resetTransform()

    def _apply_white_key(self, pix: QtGui.QPixmap, thr: int = 250) -> QtGui.QPixmap:
        """
        將接近白色的像素轉為透明：r>=thr 且 g>=thr 且 b>=thr（且原本 a==255）。
        只在顯示端處理，不改動原檔與座標。
        """
        img = pix.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        if w == 0 or h == 0:
            return pix

        bpl = img.bytesPerLine()
        ptr = img.bits()
        ptr.setsize(img.byteCount())      
        buf = memoryview(ptr).cast('B')   

        for y in range(h):
            row_off = y * bpl
            for x in range(w):
                i = row_off + x * 4
                b, g, r, a = buf[i], buf[i+1], buf[i+2], buf[i+3]
                if a == 255 and r >= thr and g >= thr and b >= thr:
                    buf[i+3] = 0        

        return QtGui.QPixmap.fromImage(img)

    def _white_key_if_background(self, pix: QtGui.QPixmap, thr: int = 250, ratio: float = 0.60) -> QtGui.QPixmap:
        """
        若『近白像素』比例 >= ratio，視為白底圖，對其做去白；否則原樣返回。
        """
        img = pix.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        if w == 0 or h == 0:
            return pix

        bpl = img.bytesPerLine()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        buf = memoryview(ptr).cast('B')

        total = w * h
        near_white = 0

        for y in range(h):
            row_off = y * bpl
            for x in range(w):
                i = row_off + x * 4
                b, g, r, a = buf[i], buf[i+1], buf[i+2], buf[i+3]
                if a == 255 and r >= thr and g >= thr and b >= thr:
                    near_white += 1

        if total > 0 and (near_white / total) >= ratio:
            return self._apply_white_key(pix, thr)
        return pix

    def show_image(self, pix: QtGui.QPixmap, fit: bool = True):
        self.resetTransform()
        self.clear()
        if pix.isNull():
            return
        
        if getattr(self, "auto_key_white", True):
            try:
                pix = self._white_key_if_background(
                    pix,
                    getattr(self, "key_white_threshold", 250),
                    getattr(self, "key_white_ratio", 0.60),
                )
            except Exception:
                pass

        self._current_pix = pix
        self._fit_to_width = fit

        self._img_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QtCore.QRectF(pix.rect()))

        self._apply_fit()

    def _apply_fit(self):
        if not self._fit_to_width or not self._current_pix or self._current_pix.isNull():
            self.resetTransform(); return
        view_width = self.viewport().width()
        if view_width <= 0:
            return
        scene_width = self._scene.sceneRect().width()
        if scene_width <= 0:
            self.resetTransform(); return
        scale = view_width / scene_width
        self.resetTransform()
        self.scale(scale, scale)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._apply_fit()

    def draw_bboxes(self, bboxes):
        self._rect_items = {}
        for si in bboxes or []:
            sid  = si.get("sub_id") if isinstance(si, dict) else si[0]
            bbox = si.get("bbox")  if isinstance(si, dict) else si[1]
            if sid is None or not bbox:
                continue
            key = str(sid)
            x, y, w, h = [int(v) for v in bbox]
            r = QtWidgets.QGraphicsRectItem(x, y, w, h)
            r.setPen(self._pen_allboxes)
            r.setBrush(QtGui.QBrush(QtCore.Qt.transparent))
            self.scene().addItem(r)
            self._rect_items[key] = r

    def focus_bbox(self, sub_id: str, margin: int = 16, use_secondary: bool = False):
        key = str(sub_id)
        if not self._rect_items:
            return
        for it in self._rect_items.values():
            it.setPen(self._pen_allboxes); it.update()
        it = self._rect_items.get(key)
        if it:
            it.setPen(self._pen_right if use_secondary else self._pen_left); it.update()
            r = it.rect().adjusted(-margin, -margin, margin, margin)
            self.ensureVisible(r, xMargin=margin, yMargin=margin)
            self.centerOn(it.rect().center())

    def colorize_pair(self, primary_sid: str = None, secondary_sid: str = None):
        for it in self._rect_items.values():
            it.setPen(QtGui.QPen(self._pen_default))
        if secondary_sid and secondary_sid in self._rect_items:
            self._rect_items[secondary_sid].setPen(QtGui.QPen(self._pen_secondary))
        rects = [self._rect_items[s].rect() for s in (primary_sid, secondary_sid) if s and s in self._rect_items]
        if rects:
            r = rects[0]
            for rr in rects[1:]:
                r = r.united(rr)
            self.centerOn(r.center())

    def highlight(self, sub_id):
        for it in self._rect_items.values():
            it.setPen(QtGui.QPen(self._pen_default))
        if not sub_id or sub_id not in self._rect_items:
            return
        self._rect_items[sub_id].setPen(QtGui.QPen(self._pen_left))
        self.centerOn(self._rect_items[sub_id].rect().center())