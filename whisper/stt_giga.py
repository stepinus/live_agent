"""Модуль распознавания речи с использованием GigaAM через ONNX."""

from __future__ import annotations

import dataclasses
import os
import platform
from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional, List
import onnxruntime as ort

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.utils import AudioBuffer

from gigaam.onnx_utils import load_onnx_sessions, transcribe_sample
import gigaam

from .models import GigaAMModels
from .log import logger

# Путь к моделям по умолчанию
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

@dataclass
class _STTOptions:
    language: str
    detect_language: bool
    model: GigaAMModels | str
    cache_dir: str | None = None
    onnx_dir: str | None = None
    providers: List[str] | None = None

class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "ru",
        detect_language: bool = False,
        model: GigaAMModels | str = "v2_ctc",  # По умолчанию используем v2_ctc
        cache_dir: str | None = None,
        onnx_dir: str | None = None,
        providers: List[str] | None = None,  # Добавляем параметр для выбора провайдеров
    ):
        """
        Создать новый экземпляр GigaAM STT с ONNX инференсом.
        
        Args:
            language: Язык для распознавания (поддерживается только русский)
            detect_language: Автоматически определять язык (не используется в GigaAM)
            model: Модель GigaAM для использования:
                  - "v2_ctc" (рекомендуется): CTC модель второго поколения
                  - "v2_rnnt": RNNT модель второго поколения
                  - "v1_ctc": CTC модель первого поколения
                  - "v1_rnnt": RNNT модель первого поколения
            cache_dir: Директория для кэширования моделей
            onnx_dir: Директория с ONNX моделями
            providers: Список провайдеров ONNX в порядке приоритета
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        
        if language != "ru":
            logger.warning("GigaAM поддерживает только русский язык. Установлен язык 'ru'.")
            language = "ru"
        
        if detect_language:
            logger.warning("GigaAM не поддерживает автоопределение языка. Опция игнорируется.")
            detect_language = False

        # Если провайдеры не указаны, выбираем оптимальные для системы
        if providers is None:
            # Определяем операционную систему
            system = platform.system().lower()
            
            # Временно используем только CPU провайдер для отладки
            logger.info("Используем CPU провайдер для стабильной работы")
            providers = ['CPUExecutionProvider']

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
            cache_dir=cache_dir,
            onnx_dir=onnx_dir,
            providers=providers,
        )
        
        self._onnx_sessions = None
        self._initialize_model()

    def _initialize_model(self):
        """Инициализация ONNX сессий для GigaAM."""
        logger.info("Инициализация ONNX сессий для GigaAM...")
        
        # Логируем доступные провайдеры ONNX
        logger.info(f"Доступные провайдеры ONNX: {ort.get_available_providers()}")
        logger.info(f"Выбранные провайдеры: {self._opts.providers}")
        
        try:
            onnx_dir = self._opts.onnx_dir or os.path.join(
                self._opts.cache_dir or DEFAULT_MODELS_DIR, 
                "onnx"
            )
            os.makedirs(onnx_dir, exist_ok=True)
            
            # Проверяем наличие всех необходимых файлов
            model_files = {
                'model': os.path.join(onnx_dir, f"{str(self._opts.model)}.onnx")
            }
            
            missing_files = [f for f in model_files.values() if not os.path.exists(f)]
            
            if missing_files:
                # Если нет ONNX модели, создаем её
                logger.info(f"ONNX модели не найдены, экспортируем...")
                
                # Загружаем базовую модель
                temp_model = gigaam.load_model(
                    str(self._opts.model),
                    fp16_encoder=False,  # Используем fp32 для CPU
                    use_flash=False      # ONNX не поддерживает flash attention
                )
                
                # Экспортируем в ONNX
                logger.info("Экспортируем модель в ONNX формат...")
                temp_model.to_onnx(
                    dir_path=onnx_dir
                )
                logger.info(f"ONNX модели экспортированы в {onnx_dir}")
            
            # Настраиваем опции сессии
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Создаем сессии с выбранными провайдерами
            self._onnx_sessions = []  # Используем список вместо словаря
            
            for name, model_path in model_files.items():
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Файл модели не найден после экспорта: {model_path}")
                    
                logger.info(f"Загрузка модели {name} из {model_path}")
                
                # Создаем сессию только с CPU провайдером
                session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                
                self._onnx_sessions.append(session)
                
                # Логируем используемый провайдер и его свойства
                provider = session.get_providers()[0]
                logger.info(f"Модель {name} использует провайдер: {provider}")
                
                # Проверяем входы/выходы модели
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                logger.info(f"Модель {name} имеет {len(inputs)} входов и {len(outputs)} выходов")
                
                # Логируем информацию о входах
                for inp in inputs:
                    logger.info(f"Вход '{inp.name}': shape={inp.shape}, type={inp.type}")
            
            logger.info("ONNX сессии успешно загружены")
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке ONNX сессий: {e}")
            raise APIConnectionError("Не удалось загрузить ONNX сессии") from e

    def update_options(
        self,
        *,
        model: GigaAMModels | None = None,
        language: str | None = None,
    ) -> None:
        """Обновить параметры STT."""
        if model:
            self._opts.model = model
            self._initialize_model()
        if language and language != "ru":
            logger.warning("GigaAM поддерживает только русский язык. Язык не изменен.")

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        if language and language != "ru":
            logger.warning("GigaAM поддерживает только русский язык. Используется 'ru'.")
        config.language = "ru"
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            config = self._sanitize_options(language=language)
            audio_data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            
            # Конвертируем WAV в путь к временному файлу
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav.write(audio_data)
                temp_wav_path = temp_wav.name
            
            try:
                print("Начинаем распознавание речи...")  # Прямой вывод для отладки
                transcription = transcribe_sample(
                    temp_wav_path,
                    "ctc" if "ctc" in str(config.model) else "rnnt",
                    self._onnx_sessions
                )
                print(f"Распознанный текст: {transcription}")  # Прямой вывод для отладки
                
                # Создаем событие с распознанным текстом
                event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text=transcription or "",
                            language="ru",
                        )
                    ],
                )
                
                return event
                
            finally:
                # Удаляем временный файл
                try:
                    os.unlink(temp_wav_path)
                except Exception as e:
                    print(f"Не удалось удалить временный файл {temp_wav_path}: {e}")

        except Exception as e:
            print(f"Ошибка распознавания: {e}")  # Прямой вывод для отладки
            logger.error("Ошибка распознавания: %s", e, exc_info=True)
            raise APIConnectionError() from e 