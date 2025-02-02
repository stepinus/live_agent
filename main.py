import os
import sys
import logging
import json
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe,  JobProcess, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero
from livekit import rtc

from whisper.stt_giga import STT as GigaSTT  # Импортируем GigaAM STT
from whisper.tts import SileroTTS, SileroTTSOptions  # Импортируем Silero TTS
from whisper.llm import TransformersLLM  # Импортируем TransformersLLM

class UTF8JsonFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
            # Форматируем JSON с поддержкой русских символов
            record.msg = json.dumps(record.msg, ensure_ascii=False)
        return super().format(record)

# Настройка основного логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Специальная настройка для pipeline логгера
pipeline_logger = logging.getLogger('livekit.agents.pipeline')
pipeline_logger.handlers.clear()  # Очищаем существующие хендлеры
handler = logging.StreamHandler()
handler.setFormatter(UTF8JsonFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
pipeline_logger.addHandler(handler)

# Устанавливаем кодировку для stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv('.env.local')

def prewarm(proc: JobProcess):
    # Настраиваем VAD с оптимальными параметрами
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration = 0.05,
        min_silence_duration = 0.55,
        prefix_padding_duration = 0.5,
        sample_rate=16000             # Частота дискретизации
    )

async def entrypoint(ctx: JobContext):
    try:
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "Вы голосовой ассистент, созданный LiveKit. Ваш интерфейс с пользователями будет голосовым. "
                "Используйте короткие и четкие ответы, избегая неудобной для произношения пунктуации."
            ),
        )

        # Сначала устанавливаем соединение
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Инициализируем чат
        chat = rtc.ChatManager(ctx.room)

        # Отправляем сообщения через chat
        await chat.send_message("Инициализация голосового ассистента...")
        await chat.send_message("Загрузка моделей...")

        # Создаем экземпляр Silero TTS
        tts_engine = SileroTTS(
            options=SileroTTSOptions(
                model_id="v4_ru",
                speaker="xenia",  # Используем голос Ксении
                sample_rate=48000,
                device="cpu",
                put_accent=True,  # Автоматическая расстановка ударений
                put_yo=True  # Автоматическая расстановка буквы ё
            )
        )

        assistant = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],  
            stt=GigaSTT(
                language="ru",
                model="v2_ctc",
                detect_language=False
            ),
            llm=TransformersLLM(),  # Используем TransformersLLM вместо TransformersLLMStream
            tts=tts_engine,
            chat_ctx=initial_ctx
        )

        await chat.send_message("Голосовой ассистент готов к работе!")
        print("Голосовой ассистент запущен и готов к работе")  # Отладочный вывод
        assistant.start(ctx.room)
        await assistant.say("Привет, чем могу помочь?", allow_interruptions=True)
            
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
        # Проверяем, установлено ли соединение перед отправкой сообщения об ошибке
        if hasattr(ctx, 'room') and hasattr(ctx.room, 'local_participant'):
            chat = rtc.ChatManager(ctx.room)
            await chat.send_message(f"Ошибка при запуске: {e}")
        raise


if __name__ == "__main__":
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        ws_url=os.getenv('LIVEKIT_URL'),
        api_key=os.getenv('LIVEKIT_API_KEY'),
        api_secret=os.getenv('LIVEKIT_API_SECRET'),
        prewarm_fnc=prewarm,
    )
    cli.run_app(worker_options)