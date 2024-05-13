#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import threading, functools, time
from datetime import datetime
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

from .models import AssistantModel, ThreadModel, MessageModel
from .types import (
    AskOpenAIType,
    LastMessageType,
    CurrentRunType,
    AssistantType,
    AssistantListType,
    ThreadType,
    ThreadListType,
    MessageType,
    MessageListType,
)
from tenacity import retry, wait_exponential, stop_after_attempt
from pytz import timezone

client = None
# assistant_functions = None  ## It will be replaced with the table.


def handlers_init(logger, **setting):
    global client
    # , assistant_functions
    try:
        client = OpenAI(
            api_key=setting["openai_api_key"],
        )
        # assistant_functions = setting[
        #     "assistant_functions"
        # ]  ## It will be replaced with the table.

    except Exception as e:
        logger.error(e)
        raise e


def get_assistant_function(logger, assistant_type, assistant_id, function_name):
    try:
        ## Get funct by assistant_id and function_name.
        ## It will be replaced with the table.
        assistant = get_assistant(assistant_type, assistant_id)
        assistant_functions = list(
            filter(lambda x: x["function_name"] == function_name, assistant.functions)
        )
        if len(assistant_functions) == 0:
            return None

        assistant_function = assistant_functions[0]
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


def update_thread_and_insert_message(info, kwargs, result, role):
    update_kwargs = {
        "assistant_id": kwargs["assistant_id"],
        "thread_id": result.thread_id,
        "run_id": getattr(result, "run_id", None) or result.current_run_id,
        "updated_by": kwargs["updated_by"],
    }
    if kwargs.get("thread_id") is None:
        update_kwargs["assistant_type"] = kwargs["assistant_type"]

    insert_update_thread_handler(info, **update_kwargs)

    last_message = last_message_handler(info, thread_id=result.thread_id, role=role)

    insert_update_message_handler(
        info,
        thread_id=last_message.thread_id,
        run_id=last_message.run_id,
        message_id=last_message.message_id,
        role=last_message.role,
        message=last_message.message,
        created_at=last_message.created_at,
    )
    return


## We can move the decorator to the uplevel.
def assistant_decorator():
    def actual_decorator(original_function):
        @functools.wraps(original_function)
        def wrapper_function(*args, **kwargs):

            function_name = original_function.__name__
            try:

                result = original_function(*args, **kwargs)

                if function_name == "ask_open_ai_handler":
                    update_thread_and_insert_message(args[0], kwargs, result, "user")

                elif (
                    function_name == "current_run_handler"
                    and result.status == "completed"
                ):
                    update_thread_and_insert_message(
                        args[0], kwargs, result, "assistant"
                    )

                return result

            except Exception as e:
                args[0].context.get("logger").error(e)
                raise e

        return wrapper_function

    return actual_decorator


class EventHandler(AssistantEventHandler):
    def __init__(self, logger, assistant_type, queue=None, print_stream=False):
        self.logger = logger
        self.assistant_type = assistant_type
        self.queue = queue
        self.print_stream = print_stream
        AssistantEventHandler.__init__(self)

    @override
    def on_event(self, event):
        self.logger.info(f"event: {event.event}")
        if event.event == "thread.run.created":
            self.logger.info(f"current_run_id: {event.data.id}")
            if self.queue is not None:
                self.queue.put({"name": "current_run_id", "value": event.data.id})

        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == "thread.run.requires_action":
            self.handle_requires_action(event.data)

    def handle_requires_action(self, data):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            assistant_function = get_assistant_function(
                self.logger, self.assistant_type, data.assistant_id, tool.function.name
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
            event_handler=EventHandler(self.logger, self.assistant_type),
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


@assistant_decorator()
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
        last_message = LastMessageType(
            thread_id=thread_id,
            message_id=None,
            role=role,
            message=None,
            created_at=None,
        )
        messages = list(
            filter(
                lambda m: m.role == role,
                client.beta.threads.messages.list(thread_id=thread_id, order="desc"),
            )
        )
        if len(messages) > 0:
            last_message.message_id = messages[0].id
            last_message.run_id = messages[0].run_id
            last_message.message = messages[0].content[0].text.value
            last_message.created_at = datetime.fromtimestamp(
                messages[0].created_at
            ).astimezone(timezone("UTC"))
        return last_message
    except Exception as e:
        info.context.get("logger").error(e)
        raise e


# Function to handle stream operations and continue processing in the background
def handle_stream(logger, thread_id, assistant_id, assistant_type, queue):
    event_handler = EventHandler(logger, assistant_type, queue=queue)
    with client.beta.threads.runs.stream(
        thread_id=thread_id, assistant_id=assistant_id, event_handler=event_handler
    ) as stream:
        stream.until_done()


def get_current_run_id_and_start_async_task(
    logger, thread_id, assistant_id, assistant_type
):
    try:
        # Create a queue to share data between threads
        queue = Queue()

        # Start the thread to handle the stream
        stream_thread = threading.Thread(
            target=handle_stream,
            args=(logger, thread_id, assistant_id, assistant_type, queue),
        )
        stream_thread.start()

        # Fetch the final_run_id from the queue
        q = (
            queue.get()
        )  # This will block until the current_run_id is put into the queue by the thread

        if q["name"] == "current_run_id":
            return q["value"]

        raise Exception("Cannot locate the value for current_run_id.")
    except Exception as e:
        logger.context.get("logger").error(e)
        raise e


@assistant_decorator()
def ask_open_ai_handler(info, **kwargs):
    try:
        assistant_type = kwargs["assistant_type"]
        assistant_id = kwargs["assistant_id"]
        question = kwargs["question"]
        thread_id = kwargs.get("thread_id")
        if thread_id is None:
            thread = client.beta.threads.create()
            thread_id = thread.id

        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=question
        )

        # Start the stream processing in a thread and fetch the final_run_id
        current_run_id = get_current_run_id_and_start_async_task(
            info.context.get("logger"), thread_id, assistant_id, assistant_type
        )

        return AskOpenAIType(
            assistant_id=kwargs["assistant_id"],
            thread_id=thread_id,
            question=kwargs["question"],
            current_run_id=current_run_id,
        )

    except Exception as e:
        info.context.get("logger").error(e)
        raise e


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_assistant(assistant_type, assistant_id):
    return AssistantModel.get(assistant_type, assistant_id)


def _get_assistant(assistant_type, assistant_id):
    assistant = get_assistant(assistant_type, assistant_id)
    return {
        "assistant_type": assistant.assistant_type,
        "assistant_id": assistant.assistant_id,
        "assistant_name": assistant.assistant_name,
        "functoins": assistant.functoins,
        "updated_by": assistant.updated_by,
        "created_at": assistant.created_at,
        "updated_at": assistant.updated_at,
    }


def get_assistant_count(assistant_type, assistant_id):
    return AssistantModel.count(
        assistant_type, AssistantModel.assistant_id == assistant_id
    )


def get_assistant_type(info, assistant):
    assistant = assistant.__dict__["attribute_values"]
    return AssistantType(**Utility.json_loads(Utility.json_dumps(assistant)))


def resolve_assistant_handler(info, **kwargs):
    return get_assistant_type(
        info,
        get_assistant(kwargs.get("assistant_type"), kwargs.get("assistant_id")),
    )


@monitor_decorator
@resolve_list_decorator(
    attributes_to_get=["assistant_type", "assistant_id"],
    list_type_class=AssistantListType,
    type_funct=get_assistant_type,
)
def resolve_assistant_list_handler(info, **kwargs):
    assistant_type = kwargs.get("assistant_type")
    assistant_name = kwargs.get("assistant_name")

    args = []
    inquiry_funct = AssistantModel.scan
    count_funct = AssistantModel.count
    if assistant_type:
        args = [assistant_type, None]
        inquiry_funct = AssistantModel.query

    the_filters = None  # We can add filters for the query.
    if assistant_name:
        the_filters &= AssistantModel.assistant_name.contains(assistant_name)
    if the_filters is not None:
        args.append(the_filters)

    return inquiry_funct, count_funct, args


@insert_update_decorator(
    keys={
        "hash_key": "assistant_type",
        "range_key": "assistant_id",
    },
    model_funct=get_assistant,
    count_funct=get_assistant_count,
    type_funct=get_assistant_type,
    # data_attributes_except_for_data_diff=data_attributes_except_for_data_diff,
    # activity_history_funct=None,
)
def insert_update_assistant_handler(info, **kwargs):
    assistant_type = kwargs.get("assistant_type")
    assistant_id = kwargs.get("assistant_id")
    if kwargs.get("entity") is None:
        AssistantModel(
            assistant_type,
            assistant_id,
            **{
                "assistant_name": kwargs.get("assistant_name"),
                "functoins": kwargs.get("functoins"),
                "updated_by": kwargs.get("updated_by"),
                "created_at": datetime.now(tz=timezone("UTC")),
                "updated_at": datetime.now(tz=timezone("UTC")),
            },
        ).save()
        return

    assistant = kwargs.get("entity")
    actions = [
        AssistantModel.updated_by.set(kwargs.get("updated_by")),
        AssistantModel.updated_at.set(datetime.now(tz=timezone("UTC"))),
    ]
    if kwargs.get("assistant_name") is not None:
        actions.append(AssistantModel.assistant_name.set(kwargs.get("assistant_name")))
    if kwargs.get("functoins") is not None:
        actions.append(AssistantModel.functoins.set(kwargs.get("functoins")))
    assistant.update(actions=actions)
    return


@delete_decorator(
    keys={
        "hash_key": "assistant_type",
        "range_key": "assistant_id",
    },
    model_funct=get_assistant,
)
def delete_assistant_handler(info, **kwargs):
    kwargs.get("entity").delete()
    return True


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_thread(assistant_id, thread_id):
    return ThreadModel.get(assistant_id, thread_id)


def _get_thread(assistant_id, thread_id):
    thread = get_thread(assistant_id, thread_id)
    return {
        "assistant_id": thread.assistant_id,
        "thread_id": thread.thread_id,
        "assistant_type": thread.assistant_type,
        "run_ids": thread.run_ids,
        "updated_by": thread.updated_by,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
    }


def get_thread_count(assistant_id, thread_id):
    return ThreadModel.count(assistant_id, ThreadModel.thread_id == thread_id)


def get_thread_type(info, thread):
    thread = thread.__dict__["attribute_values"]
    return ThreadType(**Utility.json_loads(Utility.json_dumps(thread)))


def resolve_thread_handler(info, **kwargs):
    return get_thread_type(
        info,
        get_thread(kwargs.get("assistant_id"), kwargs.get("thread_id")),
    )


@monitor_decorator
@resolve_list_decorator(
    attributes_to_get=["assistant_id", "thread_id"],
    list_type_class=ThreadListType,
    type_funct=get_thread_type,
)
def resolve_thread_list_handler(info, **kwargs):
    assistant_id = kwargs.get("assistant_id")
    assistant_type = kwargs.get("assistant_type")
    run_id = kwargs.get("run_id")

    args = []
    inquiry_funct = ThreadModel.scan
    count_funct = ThreadModel.count
    if assistant_id:
        args = [assistant_id, None]
        inquiry_funct = ThreadModel.query

    the_filters = None  # We can add filters for the query.
    if assistant_type:
        the_filters &= ThreadModel.assistant_type.contains(assistant_type)
    if run_id:
        the_filters &= ThreadModel.run_ids.contains(run_id)
    if the_filters is not None:
        args.append(the_filters)

    return inquiry_funct, count_funct, args


@insert_update_decorator(
    keys={
        "hash_key": "assistant_id",
        "range_key": "thread_id",
    },
    model_funct=get_thread,
    count_funct=get_thread_count,
    type_funct=get_thread_type,
    range_key_required=True,
    # data_attributes_except_for_data_diff=data_attributes_except_for_data_diff,
    # activity_history_funct=None,
)
def insert_update_thread_handler(info, **kwargs):
    assistant_id = kwargs["assistant_id"]
    thread_id = kwargs["thread_id"]
    if kwargs.get("entity") is None:
        ThreadModel(
            assistant_id,
            thread_id,
            **{
                "assistant_type": kwargs["assistant_type"],
                "run_ids": [kwargs["run_id"]],
                "updated_by": kwargs["updated_by"],
                "created_at": datetime.now(tz=timezone("UTC")),
                "updated_at": datetime.now(tz=timezone("UTC")),
            },
        ).save()
        return

    thread = kwargs.get("entity")
    actions = [
        AssistantModel.updated_by.set(kwargs["updated_by"]),
        AssistantModel.updated_at.set(datetime.now(tz=timezone("UTC"))),
    ]
    if kwargs.get("run_id") is not None:
        run_ids = set(thread.run_ids)
        run_ids.add(kwargs["run_id"])
        actions.append(ThreadModel.run_ids.set(list(run_ids)))
    thread.update(actions=actions)
    return


@delete_decorator(
    keys={
        "hash_key": "assistant_id",
        "range_key": "thread_id",
    },
    model_funct=get_thread,
)
def delete_thread_handler(info, **kwargs):
    kwargs.get("entity").delete()
    return True


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_message(thread_id, message_id):
    return MessageModel.get(thread_id, message_id)


def _get_message(thread_id, message_id):
    message = get_message(thread_id, message_id)
    return {
        "thread_id": message.thread_id,
        "message_id": message.message_id,
        "role": message.role,
        "message": message.message,
        "created_at": message.created_at,
    }


def get_message_count(thread_id, message_id):
    return MessageModel.count(thread_id, MessageModel.message_id == message_id)


def get_message_type(info, message):
    message = message.__dict__["attribute_values"]
    return MessageType(**Utility.json_loads(Utility.json_dumps(message)))


def resolve_message_handler(info, **kwargs):
    return get_message_type(
        info,
        get_message(kwargs.get("thread_id"), kwargs.get("message_id")),
    )


@monitor_decorator
@resolve_list_decorator(
    attributes_to_get=["thread_id", "message_id"],
    list_type_class=MessageListType,
    type_funct=get_message_type,
)
def resolve_message_list_handler(info, **kwargs):
    thread_id = kwargs.get("thread_id")
    roles = kwargs.get("roles")
    message = kwargs.get("message")

    args = []
    inquiry_funct = MessageModel.scan
    count_funct = MessageModel.count
    if thread_id:
        args = [thread_id, None]
        inquiry_funct = MessageModel.query

    the_filters = None  # We can add filters for the query.
    if roles:
        the_filters &= MessageModel.role.is_in(*roles)
    if message:
        the_filters &= MessageModel.message.contains(message)
    if the_filters is not None:
        args.append(the_filters)

    return inquiry_funct, count_funct, args


@insert_update_decorator(
    keys={
        "hash_key": "thread_id",
        "range_key": "message_id",
    },
    model_funct=get_message,
    count_funct=get_message_count,
    type_funct=get_message_type,
    range_key_required=True,
    # data_attributes_except_for_data_diff=data_attributes_except_for_data_diff,
    # activity_history_funct=None,
)
def insert_update_message_handler(info, **kwargs):
    thread_id = kwargs["thread_id"]
    message_id = kwargs["message_id"]
    if kwargs.get("entity") is None:
        MessageModel(
            thread_id,
            message_id,
            **{
                "run_id": kwargs.get("run_id"),
                "role": kwargs["role"],
                "message": kwargs["message"],
                "created_at": kwargs["created_at"],
            },
        ).save()
        return

    message = kwargs.get("entity")
    actions = []
    if kwargs.get("run_id") is not None:
        actions.append(MessageModel.run_id.set(kwargs["run_id"]))
    if kwargs.get("role") is not None:
        actions.append(MessageModel.role.set(kwargs["role"]))
    if kwargs.get("message") is not None:
        actions.append(MessageModel.role.set(kwargs["message"]))
    if kwargs.get("created_at") is not None:
        actions.append(MessageModel.created_at.set(kwargs["created_at"]))
    message.update(actions=actions)
    return


@delete_decorator(
    keys={
        "hash_key": "thread_id",
        "range_key": "message_id",
    },
    model_funct=get_message,
)
def delete_message_handler(info, **kwargs):
    kwargs.get("entity").delete()
    return True
