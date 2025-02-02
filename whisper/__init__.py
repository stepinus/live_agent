"""Модуль распознавания речи с использованием Whisper и GigaAM."""

from .stt import WhisperSTT
from .stt_giga import STT as GigaSTT
from .models import WhisperConfig, GigaAMModels
from .version import __version__
from .tts import SileroTTS, SileroTTSOptions, SileroTTSStream, SileroStreamingTTS

__all__ = [
    "WhisperSTT",
    "GigaSTT",
    "WhisperConfig",
    "GigaAMModels",
    "__version__",
    "SileroTTS",
    "SileroTTSOptions",
    "SileroTTSStream",
    "SileroStreamingTTS"
]

from livekit.agents import Plugin
from .log import logger

class WhisperPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)
        self._stt = None
        self._config = None
        self._engine = "whisper"  # По умолчанию используем whisper

    @property
    def stt(self):
        if self._stt is None:
            if self._engine == "whisper":
                self._stt = WhisperSTT(config=self._config)
            else:
                self._stt = GigaSTT(config=self._config)
        return self._stt

    def configure(self, config, engine: str = "whisper") -> None:
        """Настройка плагина с заданной конфигурацией.
        
        Args:
            config: WhisperConfig для whisper или GigaAMModels для gigaam
            engine: Движок STT ("whisper" или "gigaam")
        """
        self._config = config
        self._engine = engine
        self._stt = None  # Сбрасываем STT для переинициализации

    @staticmethod
    def load(
        model_name: str = "tiny",
        language: str = "ru",
        min_silence: float = 0.5,
        rms_threshold: float = 0.004**2,
        engine: str = "whisper",
        **kwargs
    ):
        """
        Создает и возвращает настроенный экземпляр STT.

        Args:
            model_name: Название модели ("tiny", "base", "small", "medium", "large" для Whisper
                       или "v2_ctc", "v2_rnnt", "v1_ctc", "v1_rnnt" для GigaAM)
            language: Язык распознавания
            min_silence: Минимальная длительность тишины в секундах
            rms_threshold: Порог RMS для определения речи
            engine: Движок STT ("whisper" или "gigaam")
            **kwargs: Дополнительные параметры конфигурации
        """
        config_class = WhisperConfig if engine == "whisper" else GigaAMModels
        config = config_class(
            model_name=model_name,
            language=language,
            min_silence=min_silence,
            rms_threshold=rms_threshold,
            **kwargs
        )
        plugin = WhisperPlugin()
        plugin.configure(config, engine=engine)
        return plugin.stt

Plugin.register_plugin(WhisperPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}
for n in NOT_IN_ALL:
    __pdoc__[n] = False
