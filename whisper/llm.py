import os
import logging
from typing import Optional, Dict, Any, AsyncGenerator, Union, Literal
from livekit.agents import llm
from livekit.agents.llm import ToolChoice
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

class VLLMWrapper(llm.LLM):
    def __init__(self):
        super().__init__()
        self.model_name = os.getenv('LIVEKIT_LLM', "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        logger.info(f"Загрузка модели {self.model_name}...")
        self.model = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype="float16",  # Используем float16 для экономии памяти
            gpu_memory_utilization=0.8,  # Контролируем использование GPU памяти
        )
        logger.info("Модель успешно загружена")

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] | None = None,
    ) -> llm.LLMStream:
        return VLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            temperature=temperature or 0.7
        )

class VLLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        llm: VLLMWrapper,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext],
        conn_options: APIConnectOptions,
        temperature: float = 0.7,
    ):
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options
        )
        self._llm = llm
        self._temperature = temperature

    def _format_messages(self) -> str:
        """Форматирует сообщения в единый текст для модели"""
        formatted = []
        for msg in self._chat_ctx.messages:
            if msg.role == "system":
                formatted.append(f"<s>[INST] {msg.content} [/INST]")
            elif msg.role == "user":
                formatted.append(f"<s>[INST] {msg.content} [/INST]")
            else:  # assistant
                formatted.append(f"{msg.content}</s>")
        return "\n".join(formatted)

    async def _run(self) -> None:
        try:
            # Форматируем сообщения
            prompt = self._format_messages()
            
            # Настраиваем параметры генерации
            sampling_params = SamplingParams(
                temperature=self._temperature,
                top_p=0.95,
                max_tokens=1024,
                presence_penalty=1.1,  # Аналог repetition_penalty
                stream=True
            )

            # Генерируем ответ с потоковым выводом
            for output in self._llm.model.generate(prompt, sampling_params):
                if not output.text.strip():
                    continue
                    
                # Отправляем чанк в поток
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        request_id="vllm-1",
                        choices=[
                            llm.Choice(
                                delta=llm.ChoiceDelta(
                                    content=output.text,
                                    role="assistant"
                                ),
                                index=0
                            )
                        ]
                    )
                )

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise 