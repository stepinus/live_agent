import onnxruntime as ort
import platform
import sys
import subprocess

def get_pip_packages():
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    return result.stdout

def check_onnx_providers():
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"ONNXRuntime version: {ort.__version__}")
    
    print("\nУстановленные пакеты ONNX:")
    pip_packages = get_pip_packages()
    for line in pip_packages.split('\n'):
        if 'onnx' in line.lower():
            print(line.strip())
    
    print("\nДоступные провайдеры:")
    providers = ort.get_available_providers()
    for i, provider in enumerate(providers, 1):
        print(f"{i}. {provider}")
    
    print("\nИнформация о системе:")
    if 'CoreMLExecutionProvider' in providers:
        print("✓ CoreML доступен для Apple Silicon")
    if 'MetalExecutionProvider' in providers:
        print("✓ Metal доступен для Apple Silicon")
    if 'CUDAExecutionProvider' in providers:
        print("✓ CUDA доступен для GPU")
    
    print("\nПроверка создания сессии с разными провайдерами:")
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0
    
    for provider in providers:
        try:
            _ = ort.InferenceSession(
                "dummy",  # Фиктивный путь, сессия не будет создана
                providers=[provider],
                session_options=sess_options
            )
            print(f"✓ {provider}: OK")
        except Exception as e:
            print(f"✗ {provider}: {str(e)}")

if __name__ == "__main__":
    check_onnx_providers() 