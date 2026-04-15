import os
import sys
import socket
import threading
import time
import webbrowser
from app import app

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def open_browser(port):
    """Wait a short moment and open the browser."""
    time.sleep(1.5)
    webbrowser.open(f'http://127.0.0.1:{port}')

def main():
    # If running as a bundled executable, set the working directory to _MEIPASS
    # so Flask can find templates/ and static/ from the current directory if needed.
    # Flask uses the app's root_path, but setting cwd is generally safer for PyInstaller.
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    port = find_free_port()
    print(f"Starting server on http://127.0.0.1:{port}")
    
    # Start browser opening thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    # Run the Flask app
    # Disable reloader since it doesn't work well with PyInstaller and threading
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()
