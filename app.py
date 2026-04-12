import os
import sys
import subprocess
from flask import Flask, send_from_directory

def start_web_mode():
    print("\nStarting ALM-1 Local Web Server...")
    print("Point your browser to http://localhost:5000")

    app = Flask(__name__, static_folder='web')

    @app.route('/')
    def serve_index():
        return send_from_directory('web', 'index.html')

    @app.route('/<path:path>')
    def serve_static(path):
        return send_from_directory('web', path)

    app.run(host='0.0.0.0', port=5000)

def start_terminal_mode():
    print("\nLaunching ALM-1 Terminal Interface...")
    subprocess.run([sys.executable, "cli.py"])

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("========================================")
    print("   ALM-1: Adaptive Language Model       ")
    print("========================================")
    print("1. Web Mode (Browser Interface)")
    print("2. Terminal Mode (TUI Interface)")
    print("3. Exit")
    print("========================================")

    choice = input("\nChoose a mode (1-3): ")

    if choice == '1':
        start_web_mode()
    elif choice == '2':
        start_terminal_mode()
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()
