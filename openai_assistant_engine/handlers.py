#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import threading
from queue import Queue
from typing_extensions import override
from openai import OpenAI, AssistantEventHandler
from silvaengine_utility import Utility
from silvaengine_dynamodb_base import (
    monitor_decorator,
    insert_update_decorator,
    resolve_list_decorator,
    delete_decorator,
)

# from .models import AssistantModel, ThreadModel, MessageModel
from .types import AskOpenAIType, LastMessageType, CurrentRunType

client = None
assistant_functions = None


def handlers_init(logger, **setting):
    global client, assistant_functions
    try:
        client = OpenAI(
            api_key=setting["openai_api_key"],
        )
        assistant_functions = setting["assistant_functions"]

    except Exception as e:
        logger.error(e)
        raise e


def get_assistant_function(logger, assistant_id, function_name):
    try:
        ## Get funct by assistant_id and function_name.
        if assistant_functions.get(assistant_id) is None:
            return None

        assistant_function = assistant_functions[assistant_id].get(function_name)
        if assistant_function is None:
            return assistant_function

        assistant_function_class = getattr(
            __import__(assistant_function["module_name"]),
            assistant_function["class_name"],
        )
        return getattr(
            assistant_function_class(
                logger,
                **Utility.json_loads(
                    Utility.json_dumps(assistant_function["configuration"])
                ),
            ),
            function_name,
        )
    except Exception as e:
        logger.error(e)
        raise e


class EventHandler(AssistantEventHandler):
    def __init__(self, logger, queue=None, print_stream=False):
        self.logger = logger
        self.queue = queue
        self.print_stream = print_stream
        AssistantEventHandler.__init__(self)

    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == "thread.run.requires_action":
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.logger.info(f"current_run_id: {run_id}")
            if self.queue is not None:
                self.queue.put(run_id)
            self.handle_requires_action(event.data)

    def handle_requires_action(self, data):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            assistant_function = get_assistant_function(
                self.logger, data.assistant_id, tool.function.name
            )
            if assistant_function is None:
                raise Exception(
                    f"The function ({tool.function.name}) is not supported!!!"
                )

            # Aggregate results text for summarization
            arguments = Utility.json_loads(tool.function.arguments)
            tool_outputs.append(
                {
                    "tool_call_id": tool.id,
                    "output": Utility.json_dumps(assistant_function(**arguments)),
                }
            )

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs)

    def submit_tool_outputs(self, tool_outputs):
        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(self.logger),
        ) as stream:
            if self.print_stream:
                for text in stream.text_deltas:
                    print(text, end="", flush=True)
                print()
            else:
                stream.until_done()


def get_messages_for_the_conversation(
    logger, thread_id, roles=["user", "assistant"], order="asc"
):
    try:
        messages = client.beta.threads.messages.list(thread_id=thread_id, order=order)
        logger.info("# Messages")
        messages = []
        for m in messages:
            if m.role not in roles:
                continue
            logger.info(f"{m.role}: {m.content[0].text.value}")
            messages.append(
                {
                    "id": m.id,
                    "created_at": m.created_at,
                    "role": m.role,
                    "value": m.content[0].text.value,
                }
            )
        return messages
    except Exception as e:
        logger.error(e)
        raise e


def current_run_handler(info, **kwargs):
    try:
        thread_id = kwargs["thread_id"]
        run_id = kwargs["run_id"]
        current_run = client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_id
        )
        return CurrentRunType(
            thread_id=thread_id, run_id=run_id, status=current_run.status
        )
    except Exception as e:
        info.context.get("logger").error(e)
        raise e


def last_message_handler(info, **kwargs):
    try:
        thread_id = kwargs["thread_id"]
        role = kwargs["role"]
        last_message = None
        messages = list(
            filter(
                lambda m: m.role == role,
                client.beta.threads.messages.list(thread_id=thread_id, order="desc"),
            )
        )
        if len(messages) > 0:
            last_message = messages[0].content[0].text.value
        return LastMessageType(thread_id=thread_id, role=role, message=last_message)
    except Exception as e:
        info.context.get("logger").error(e)
        raise e


# Function to handle stream operations and continue processing in the background
def handle_stream(logger, thread_id, assistant_id, queue):
    event_handler = EventHandler(logger, queue=queue)
    with client.beta.threads.runs.stream(
        thread_id=thread_id, assistant_id=assistant_id, event_handler=event_handler
    ) as stream:
        stream.until_done()


def get_current_run_id_and_start_async_task(logger, thread_id, assistant_id):
    # Create a queue to share data between threads
    queue = Queue()

    # Start the thread to handle the stream
    stream_thread = threading.Thread(
        target=handle_stream, args=(logger, thread_id, assistant_id, queue)
    )
    stream_thread.start()

    # Fetch the final_run_id from the queue
    current_run_id = (
        queue.get()
    )  # This will block until the current_run_id is put into the queue by the thread

    return current_run_id


def ask_open_ai_handler(info, **kwargs):
    try:
        question = kwargs["question"]
        assistant_id = kwargs["assistant_id"]
        thread_id = kwargs.get("thread_id")
        if thread_id is None:
            thread = client.beta.threads.create()
            thread_id = thread.id

        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=question
        )

        # Start the stream processing in a thread and fetch the final_run_id
        current_run_id = get_current_run_id_and_start_async_task(
            info.context.get("logger"), thread_id, assistant_id
        )

        return AskOpenAIType(
            assistant_id=assistant_id,
            thread_id=thread_id,
            question=question,
            current_run_id=current_run_id,
        )

    except Exception as e:
        info.context.get("logger").error(e)
        raise e
