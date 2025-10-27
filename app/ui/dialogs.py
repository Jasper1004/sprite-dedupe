from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from ..utils.image_io import qpixmap_from_rgba

class PairDecisionDialog(QtWidgets.QDialog):
    """
    簡化版配對決策對話框：
    - 顯示左右兩張縮圖
    - 提供『保留左 / 保留右 / 都保留 / 略過』四種選項
    - 用 self.choice 回傳: "keep_left" / "keep_right" / "keep_both" / "skip"
    """
    def __init__(self, A, B, hamming: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("配對決策")
        self.choice = None
        self.resize(720, 420)

        lay = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QWidget()
        hl  = QtWidgets.QHBoxLayout(top)
        hl.setContentsMargins(12, 8, 12, 0)
        hl.setSpacing(16)

        def _thumb_of(item):
            if getattr(item, "rgba", None) is not None:
                pm = qpixmap_from_rgba(item.rgba)
            elif getattr(item, "src_path", None):
                pm = QtGui.QPixmap(item.src_path)
            else:
                pm = QtGui.QPixmap()
            if pm.isNull():
                pm = QtGui.QPixmap(160, 160); pm.fill(Qt.darkGray)
            return pm.scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        left_pm  = _thumb_of(A)
        right_pm = _thumb_of(B)

        lblL = QtWidgets.QLabel(); lblL.setPixmap(left_pm);  lblL.setAlignment(Qt.AlignCenter)
        lblR = QtWidgets.QLabel(); lblR.setPixmap(right_pm); lblR.setAlignment(Qt.AlignCenter)

        colL = QtWidgets.QVBoxLayout(); wL = QtWidgets.QWidget(); wL.setLayout(colL)
        colL.addWidget(QtWidgets.QLabel(f"<b>{A.display_name}</b>"))
        colL.addWidget(lblL, 1)

        colR = QtWidgets.QVBoxLayout(); wR = QtWidgets.QWidget(); wR.setLayout(colR)
        colR.addWidget(QtWidgets.QLabel(f"<b>{B.display_name}</b>"))
        colR.addWidget(lblR, 1)

        mid = QtWidgets.QVBoxLayout(); wM = QtWidgets.QWidget(); wM.setLayout(mid)
        labHam = QtWidgets.QLabel(f"Hamming: {hamming}")
        labHam.setAlignment(Qt.AlignCenter)
        mid.addStretch(1); mid.addWidget(labHam); mid.addStretch(1)

        hl.addWidget(wL, 1); hl.addWidget(wM); hl.addWidget(wR, 1)
        lay.addWidget(top, 1)

        btns = QtWidgets.QDialogButtonBox()
        bL = btns.addButton("保留左",    QtWidgets.QDialogButtonBox.ActionRole)
        bR = btns.addButton("保留右",    QtWidgets.QDialogButtonBox.ActionRole)
        bB = btns.addButton("兩張都留",  QtWidgets.QDialogButtonBox.ActionRole)
        bS = btns.addButton("略過",      QtWidgets.QDialogButtonBox.RejectRole)

        def _choose(s):
            self.choice = s
            self.accept()

        bL.clicked.connect(lambda: _choose("keep_left"))
        bR.clicked.connect(lambda: _choose("keep_right"))
        bB.clicked.connect(lambda: _choose("keep_both"))
        bS.clicked.connect(lambda: _choose("skip"))

        lay.addWidget(btns, 0)