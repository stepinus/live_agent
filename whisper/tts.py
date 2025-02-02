# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import re
import torch
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncGenerator, Union, List
from livekit.agents import tts, utils
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

SILERO_SAMPLE_RATE = 48000  # Silero поддерживает 8000, 24000, 48000
SILERO_CHANNELS = 1
CHUNK_SIZE = 4096  # Размер чанка для стриминга

@dataclass
class SileroTTSOptions:
    model_id: str = "v4_ru"  # Используем русскую модель v4
    speaker: str = "xenia"  # Один из доступных голосов
    sample_rate: int = SILERO_SAMPLE_RATE
    device: str = "cpu"
    put_accent: bool = True
    put_yo: bool = True

class SileroTTS(tts.TTS):
    def __init__(
        self,
        *,
        options: Optional[SileroTTSOptions] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,  # Включаем поддержку стриминга
            ),
            sample_rate=SILERO_SAMPLE_RATE,
            num_channels=SILERO_CHANNELS,
        )
        
        self._opts = options or SileroTTSOptions()
        self._device = torch.device(self._opts.device)
        self._model = None
        
    async def _load_model(self):
        """Ленивая загрузка модели при первом использовании"""
        if self._model is None:
            logger.info(f"Загрузка Silero TTS модели {self._opts.model_id}...")
            
            # Загружаем модель через torch.hub
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker=self._opts.model_id
            )
            self._model.to(self._device)
            logger.info("Модель Silero TTS успешно загружена")

    def update_options(self, *, speaker: Optional[str] = None, sample_rate: Optional[int] = None) -> None:
        """Обновление параметров TTS на лету"""
        if speaker:
            self._opts.speaker = speaker
        if sample_rate:
            self._opts.sample_rate = sample_rate

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[Dict[str, Any]] = None,
    ) -> tts.ChunkedStream:
        return SileroTTSStream(
            tts=self,
            input_text=text,
            opts=self._opts,
        )

    def stream(self) -> tts.SynthesizeStream:
        return SileroStreamingTTS(self, self._opts)

class SileroTTSStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: SileroTTS,
        input_text: str,
        opts: SileroTTSOptions,
    ) -> None:
        super().__init__(
            tts=tts, 
            input_text=input_text,
            conn_options=DEFAULT_API_CONNECT_OPTIONS  # Используем дефолтные опции вместо пустого словаря
        )
        self._opts = opts
        self._tts = tts

    async def _run(self):
        try:
            # Загружаем модель при первом использовании
            await self._tts._load_model()
            
            # Генерируем аудио
            with torch.no_grad():
                audio = self._tts._model.apply_tts(
                    text=self.input_text,
                    speaker=self._opts.speaker,
                    sample_rate=self._opts.sample_rate,
                    put_accent=self._opts.put_accent,
                    put_yo=self._opts.put_yo
                )
                
            # Конвертируем в нужный формат
            audio = audio.unsqueeze(0)  # [1, samples]
            
            # Разбиваем на чанки для стриминга
            total_samples = audio.shape[1]
            for start in range(0, total_samples, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, total_samples)
                chunk = audio[:, start:end]
                
                # Отправляем чанк в поток
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        frame=chunk,
                        request_id=utils.shortuuid(),
                    )
                )

        except Exception as e:
            logger.error(f"Ошибка при синтезе речи: {e}")
            raise

class SileroStreamingTTS(tts.SynthesizeStream):
    def __init__(self, tts: SileroTTS, opts: SileroTTSOptions):
        super().__init__(
            tts=tts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS  # Добавляем опции соединения
        )
        self._tts = tts
        self._opts = opts
        self._text_buffer = ""
        self._sentence_queue = asyncio.Queue()
        self._closed = False
        self._current_request_id = None
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        # Разбиваем по знакам препинания, сохраняя их
        sentences = re.split(r'([.!?]+)', text)
        result = []
        
        # Собираем предложения обратно с знаками препинания
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1].strip() in '.!?':
                sentence = sentences[i] + sentences[i + 1]
                i += 2
            else:
                sentence = sentences[i]
                i += 1
            
            if sentence.strip():
                result.append(sentence.strip())
        
        return result

    def pushText(self, text: Optional[str] = None) -> None:
        """Добавляет текст в буфер и обрабатывает готовые предложения"""
        if self._closed or text is None:
            return
            
        self._text_buffer += text
        
        # Разбиваем буфер на предложения
        sentences = self._split_into_sentences(self._text_buffer)
        
        # Если есть полные предложения, добавляем их в очередь
        if len(sentences) > 1:  # Оставляем последнее возможно незаконченное предложение в буфере
            for sentence in sentences[:-1]:
                if sentence.strip():
                    self._sentence_queue.put_nowait(sentence)
            self._text_buffer = sentences[-1]
    
    def markSegmentEnd(self) -> None:
        """Обрабатываем оставшийся текст в буфере"""
        if self._text_buffer.strip():
            self._sentence_queue.put_nowait(self._text_buffer.strip())
            self._text_buffer = ""

    async def next(self) -> Optional[tts.SynthesizedAudio]:
        if self._closed:
            return None
            
        try:
            # Получаем следующее предложение из очереди
            sentence = await self._sentence_queue.get()
            
            # Генерируем аудио для предложения
            await self._tts._load_model()
            
            with torch.no_grad():
                audio = self._tts._model.apply_tts(
                    text=sentence,
                    speaker=self._opts.speaker,
                    sample_rate=self._opts.sample_rate,
                    put_accent=self._opts.put_accent,
                    put_yo=self._opts.put_yo
                )
            
            # Конвертируем в нужный формат
            audio = audio.unsqueeze(0)
            
            # Создаем уникальный ID для запроса
            self._current_request_id = utils.shortuuid()
            
            # Разбиваем на чанки и отправляем
            total_samples = audio.shape[1]
            for start in range(0, total_samples, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, total_samples)
                chunk = audio[:, start:end]
                
                return tts.SynthesizedAudio(
                    frame=chunk,
                    request_id=self._current_request_id,
                )
                
        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            logger.error(f"Ошибка при стриминговом синтезе речи: {e}")
            raise

    async def close(self, wait: bool = True) -> None:
        """Закрываем поток"""
        self._closed = True
        # Обрабатываем оставшийся текст, если wait=True
        if wait:
            self.markSegmentEnd()
            while not self._sentence_queue.empty():
                await self.next()

    async def _run(self) -> None:
        """Внутренний метод для обработки потока"""
        try:
            while not self._closed:
                audio = await self.next()
                if audio is not None:
                    self._event_ch.send_nowait(audio)
                else:
                    # Если нет данных, ждем немного перед следующей попыткой
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Ошибка в потоке TTS: {e}")
            raise
