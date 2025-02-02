"""Модуль распознавания речи с использованием Whisper."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
import torch
import numpy as np
from faster_whisper import WhisperModel

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.utils import AudioBuffer

from .models import WhisperModels
from .log import logger

# Путь к моделям по умолчанию
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

@dataclass
class _STTOptions:
    language: str
    detect_language: bool
    model: WhisperModels | str
    device: str | None = None
    compute_type: str | None = None
    cache_dir: str | None = None

class WhisperSTT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: WhisperModels | str = "large-v3",
        device: str | None = None,
        compute_type: str | None = None,
        cache_dir: str | None = None,
    ):
        """
        Создать новый экземпляр Whisper STT.
        
        Args:
            language: Язык для распознавания
            detect_language: Автоматически определять язык
            model: Модель Whisper для использования
            device: Устройство для вычислений (cuda/cpu)
            compute_type: Тип вычислений (float16/int8)
            cache_dir: Директория для кэширования моделей
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        
        if detect_language:
            language = ""

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
        )
        
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Инициализация модели Whisper."""
        logger.info("Инициализация модели Whisper...")
        
        device = self._opts.device
        compute_type = self._opts.compute_type
        
        if device is None or compute_type is None:
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
        
        logger.info(f"Используется устройство: {device}, тип вычислений: {compute_type}")
        
        cache_dir = self._opts.cache_dir or DEFAULT_MODELS_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Загрузка модели из директории: {cache_dir}")
        self._model = WhisperModel(
            str(self._opts.model),
            device=device,
            compute_type=compute_type,
            download_root=cache_dir
        )
        logger.info("Модель Whisper успешно загружена")

    def update_options(
        self,
        *,
        model: WhisperModels | None = None,
        language: str | None = None,
    ) -> None:
        """Обновить параметры STT."""
        if model:
            self._opts.model = model
            self._initialize_model()
        if language:
            self._opts.language = language

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
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
            
            # Конвертируем WAV в numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            segments, info = self._model.transcribe(
                audio_array,
                language=config.language if not config.detect_language else None,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=True,
                no_speech_threshold=0.3,
                compression_ratio_threshold=2.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            segments_list = list(segments)
            full_text = " ".join(segment.text.strip() for segment in segments_list)
            
            detected_language = info.language if config.detect_language else config.language

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=full_text or "",
                        language=detected_language or config.language or "",
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}", exc_info=True)
            raise APIConnectionError() from e
