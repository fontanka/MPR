"""
Cardiac CT Measurement Application Entry Point
"""
import sys
from PySide6.QtWidgets import QApplication
from src.app import MainWindow


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Cardiac CT Measurements")
    app.setOrganizationName("Innoventric")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
