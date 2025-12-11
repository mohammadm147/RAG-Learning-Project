import subprocess

def start_llama_server():
    ps_command = r"""
& "[path to llama.cpp]\llama-server.exe" -m "[Path to model file]\[model file].gguf" --host 0.0.0.0 --port 8000 -c 16384 -ngl 99 --temp 0.7 --repeat-penalty 1.1
"""
    process = subprocess.Popen(
        ["powershell", "-NoExit", "-Command", ps_command],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    return process

if __name__ == "__main__":
    print("Starting llama server...")
    process = start_llama_server()
    print(f"Server started with PID: {process.pid}")
