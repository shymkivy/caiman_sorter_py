"""QApplication bootstrap and top-level entry point."""
import sys


def run():
    from PyQt5.QtWidgets import QApplication
    from caiman_sorter_py.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
