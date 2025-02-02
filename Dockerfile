FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Установка базовых пакетов
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Устанавливаем VLLM с CUDA
RUN pip3 install --no-cache-dir vllm

# Запускаем приложение
CMD ["python3", "main.py"] 