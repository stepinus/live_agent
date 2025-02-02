"""Вспомогательные функции для плагина Whisper."""

from enum import Enum
from typing import Callable
import numpy as np
from livekit import rtc
from .log import logger

# This is the magic number during testing that we use to determine if a frame is loud enough
# to possibly contain speech. It's very conservative.
MAGIC_NUMBER_THRESHOLD = 0.004**2

class AudioEnergyFilter:
    """Фильтр для определения наличия речи в аудиопотоке."""
    
    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(self, *, min_silence: float = 1.5, rms_threshold: float = MAGIC_NUMBER_THRESHOLD):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> State:
        """Обновляет состояние фильтра на основе входящего аудиофрейма."""
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))
        
        # logger.debug(f"Аудио данные: размер={len(arr)}, RMS={rms:.6f}, порог={self._rms_threshold}")

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
                logger.info("Обнаружено начало речи")
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                    logger.info("Обнаружен конец речи")
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING
                
        # logger.debug(f"Состояние: {self._state.name}, cooldown={self._cooldown:.2f}")
        return self._state

class PeriodicCollector:
    """Собирает данные о длительности аудио и периодически вызывает callback."""
    
    def __init__(self, *, callback: Callable[[float], None], duration: float = 5.0):
        """
        Инициализирует коллектор.
        
        Args:
            callback: Функция, вызываемая при достижении указанной длительности
            duration: Период вызова callback в секундах
        """
        self._callback = callback
        self._duration = duration
        self._accumulated = 0.0

    def push(self, duration: float) -> None:
        """
        Добавляет длительность аудио фрейма.
        
        Args:
            duration: Длительность фрейма в секундах
        """
        self._accumulated += duration
        if self._accumulated >= self._duration:
            self._callback(self._accumulated)
            self._accumulated = 0.0

    def flush(self) -> None:
        """Принудительно вызывает callback с накопленной длительностью."""
        if self._accumulated > 0:
            self._callback(self._accumulated)
            self._accumulated = 0.0
