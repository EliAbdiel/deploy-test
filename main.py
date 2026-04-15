import os
import logging
import chainlit as cl
from datetime import datetime
from chainlit.types import ThreadDict
from typing import List, Dict, Optional, AsyncIterator
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from ollama import AsyncClient, ChatResponse, web_search, web_fetch


# Configure the logging system (defaults to WARNING level and console output)
logger = logging.getLogger(__name__)

host = os.environ.get('OLLAMA_BASE_URL')
gemma_4 = os.environ.get('MODEL')
secret = os.environ.get('OLLAMA_API_KEY')
connection_string = os.environ.get('POSTGRES_URL')


if all([host, gemma_4, secret, connection_string]):
    logger.info("All environment variables found in your .env file.")
else:
    logger.warning("One or more environment variables are missing. Please check your .env file.")


async def call_ollama(messages: List[Dict]):
    client = AsyncClient(
        host=host,
        headers={"Authorization": f"Bearer {secret}"},
    )

    web_tools = {'web_search': web_search, 'web_fetch': web_fetch}
    max_iterations = 0

    while max_iterations < 3:
        try:
            response: AsyncIterator[ChatResponse] = await client.chat(
                model=gemma_4,
                messages=messages,
                tools=[web_search, web_fetch],
                think=True,
                stream=True,
                options={
                    'temperature': 0.01,
                    'num_ctx':512,
                },
            )
        except Exception as ex:
            logger.error(f'Call LLM Error: {e}')

        if response is None:
            logger.error('No response from ollama')
            break

        tool_calls = []
        thinking = ''
        content = ''

        async for part in response:
            if part.message.tool_calls:
                tool_calls.extend(part.message.tool_calls)
            if part.message.thinking:
                thinking += part.message.thinking
            if part.message.content:
                content += part.message.content
                yield part.message.content

        if thinking != '' or content != '' or len(tool_calls) > 0:
            # logger.info(f"Internal Messages: {internal_message_content}")
            messages.append(
                {
                    'role': 'assistant',
                    'thinking': thinking,
                    'content': str(content)[:2000 * 4],
                    'tool_calls': tool_calls
                }
            )

        if tool_calls:
            tool_executed = False
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                function_to_call = web_tools.get(tool_name)
                if function_to_call:
                    tool_args = tool_call.function.arguments
                    logger.info(f"Executed tool: {tool_name}, with arguments: {tool_args}")
                    tool_result = function_to_call(**tool_args)

                    messages.append({'role': 'tool', 'content': str(tool_result)[:2000 * 4], 'tool_name': tool_name})
                    tool_executed = True
                else:
                    messages.append({'role': 'tool', 'content': f"Tool '{tool_name}' not found", 'tool_name': tool_name})
                    tool_executed = True
            if not tool_executed:
                break
        else:
            break

        max_iterations += 1
        logger.info(f"Max Tool Called: {max_iterations}")


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

        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        system_message = ("You are an intelligent assistant designed to provide accurate, relevant, and up-to-date information.\n"\
        "You have access to:\n"\
           
        "- web_search: to discover relevant and current sources.\n"\
        "- web_fetch: to retrieve and analyze the contents of specific URLs.\n"\
        "- current_time: the current date and time, which should be used to find up-to-date information.\n"\
    
        "Guidelines:\n"\
           
        "1. Use web_search to find relevant sources.\n"\
        "2. Evaluate and select the best sources.\n"\
        "3. Use web_fetch to extract key details.\n"\
        "4. Synthesize the information into a clear and concise answer.\n"\
    
        f"The current time is {current_time}. Always prioritize the topic with current_time.")

        chat_history.append({'role': 'system', 'content': system_message})

        chat_history.append({'role': 'user', 'content': message.content})
        
        msg = cl.Message(content='')

        async for part in call_ollama(chat_history):
            # if token := part or "":
            await msg.stream_token(part)

        chat_history.append({'role': 'assistant', 'content': msg.content})
        logging.info('Assistant response successfully')
        await msg.update()
    except Exception as e:
        logging.error(f"Exception: {e}")
        await cl.Message(content=f"You say: {message.content}")


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=connection_string, 
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