"""QApplication bootstrap and top-level entry point."""
import os
import sys


def _silence_tf() -> None:
    """Suppress TensorFlow / Keras deprecation chatter that fires the first
    time CaImAn imports the CNN classifier. Must run before any caiman import.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
    except Exception:
        pass
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                module=r"tensorflow.*")
        warnings.filterwarnings("ignore", category=FutureWarning,
                                module=r"tensorflow.*")
    except Exception:
        pass


_silence_tf()


def run():
    from PyQt5.QtWidgets import QApplication
    from caiman_sorter_py.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
