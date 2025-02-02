"""Модели и конфигурации для плагинов Whisper и GigaAM."""

from dataclasses import dataclass
from typing import Optional, Literal

WhisperModels = Literal[
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3"
]

GigaAMModels = Literal[
    "v2_ctc",
    "v2_rnnt",
    "v1_ctc",
    "v1_rnnt"
]

@dataclass
class WhisperConfig:
    """Конфигурация для Whisper STT."""
    
    # Параметры модели
    model_name: WhisperModels = "base"
    language: str = "ru"
    device: Optional[str] = None  # "cuda" или "cpu"
    compute_type: Optional[str] = None  # "float16" или "int8"
    cache_dir: Optional[str] = None
    
    # Параметры детекции речи
    min_silence: float = 1.5  # минимальная длительность тишины в секундах
    rms_threshold: float = 0.004**2  # порог RMS для определения речи
    
    # Параметры распознавания
    beam_size: int = 1
    best_of: int = 1
    no_speech_threshold: float = 0.3
    compression_ratio_threshold: float = 2.0
    condition_on_previous_text: bool = True
    
    # Параметры аудио
    sample_rate: int = 16000
    num_channels: int = 1

@dataclass
class GigaAMConfig:
    """Конфигурация для GigaAM STT."""
    
    # Параметры модели
    model_name: GigaAMModels = "v2_ctc"
    language: str = "ru"
    cache_dir: Optional[str] = None
    onnx_dir: Optional[str] = None
    
    # Параметры детекции речи
    min_silence: float = 0.5  # минимальная длительность тишины в секундах
    rms_threshold: float = 0.004**2  # порог RMS для определения речи
    
    # Параметры аудио
    sample_rate: int = 16000
    num_channels: int = 1
