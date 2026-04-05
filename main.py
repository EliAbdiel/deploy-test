import os
import logging
import chainlit as cl
from dotenv import load_dotenv
from ollama import AsyncClient
from typing import Dict, Optional
from chainlit.types import ThreadDict
from ollama._types import ChatResponse
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

load_dotenv()

# Configure the logging system (defaults to WARNING level and console output)
logger = logging.getLogger(__name__)

host = os.getenv('OLLAMA_BASE_URL')
model = os.getenv('MODEL')
secret = os.getenv('OLLAMA_API_KEY')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
host_db = os.getenv('HOST')
port = os.getenv('PORT')
database = os.getenv('DATABASE')

if not all([host, model, secret, user, password, host_db, port, database]):
    logger.error("One or more environment variables are missing. Please check your .env file.")
    raise ValueError("Missing environment variables")


async def call_ollama(message: list) -> ChatResponse:
    client = AsyncClient(
        host=host,
        headers={"Authorization": f"Bearer {secret}"},
    )
    response = await client.chat(model=model, messages=message, stream=True)
    logging.info('Ollama response successfully')
    return response


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initializes user session variables at the start of a chat."""
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    try:
        chat_history = cl.user_session.get("chat_history")
        chat_history.append({"role": "user", "content": message.content})
        
        result = await call_ollama(chat_history)
        msg = cl.Message(content='')

        async for part in result:
            if token := part.message.content or "":
                await msg.stream_token(token)

        chat_history.append({"role": "assistant", "content": msg.content})
        logging.info('Assistant response successfully')
        await msg.update()
    except Exception as e:
        logging.error(f"Exception: {e}")
        await cl.Message(content=f"You say: {message.content}")


def connection_string():
    return f"postgresql+asyncpg://{user}:{password}@{host_db}:{port}/{database}"


@cl.data_layer
def get_data_layer():
    connection_str = connection_string()
    return SQLAlchemyDataLayer(
        conninfo=connection_str, 
        # storage_provider=storage_client
    )


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("chat_history", [])

    for message in thread["steps"]:
        if message["type"] == "user_message":
            cl.user_session.get("chat_history").append(
                {"role": "user", "content": message["output"]}
            )
        elif message["type"] == "assistant_message":
            cl.user_session.get("chat_history").append(
                {"role": "assistant", "content": message["output"]}
            )