import os
import logging
import chainlit as cl
from dotenv import load_dotenv
from ollama import AsyncClient
from ollama._types import ChatResponse

load_dotenv()

# Configure the logging system (defaults to WARNING level and console output)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

host = os.environ['OLLAMA_BASE_URL']
model = os.environ['MODEL']
secret = os.environ['OLLAMA_API_KEY']

async def call_ollama(message: list) -> ChatResponse:
    client = AsyncClient(
        host=host,
        headers={"Authorization": f"Bearer {secret}"},
    )
    response = await client.chat(model=model, messages=message, stream=True)
    logging.info('Ollama response successfully')
    return response

@cl.on_chat_start
async def on_chat_start() -> None:
    """Initializes user session variables at the start of a chat."""
    cl.user_session.set("chat_history", [])

@cl.on_message
async def on_message(message: cl.Message):
    try:
        messages = cl.user_session.get("chat_history", [])
        messages.append({'role': 'user', 'content': message.content})
        
        result = await call_ollama(messages)
        msg = cl.Message(content='')

        async for part in result:
            if token := part.message.content or "":
                await msg.stream_token(token)

        messages.append({'role': 'assistant', 'content': msg.content})
        logging.info('Assistant response successfully')
        await msg.update()
    except Exception as e:
        logging.error(f"Exception: {e}")
        await cl.Message(content=f"You say: {message.content}")