#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import base64
import functools
import logging
import threading
import time
import traceback
from io import BytesIO
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

import pendulum
from graphene import ResolveInfo
from httpx import Response
from openai import AssistantEventHandler, OpenAI
from openai.types.beta import AssistantStreamEvent
from silvaengine_dynamodb_base import (
    delete_decorator,
    insert_update_decorator,
    monitor_decorator,
    resolve_list_decorator,
)
from silvaengine_utility import Utility
from tenacity import retry, stop_after_attempt, wait_exponential
from typing_extensions import override

from .models import AssistantModel, MessageModel, ThreadModel
from .types import (
    AskOpenAIType,
    AssistantListType,
    AssistantType,
    CurrentRunType,
    LiveMessageType,
    MessageListType,
    MessageType,
    OpenAIFileType,
    ThreadListType,
    ThreadType,
)

client = None


def handlers_init(logger: logging.Logger, **setting: Dict[str, Any]) -> None:
    global client
    try:
        client = OpenAI(
            api_key=setting["openai_api_key"],
        )
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


def get_assistant_function(
    logger: logging.Logger, assistant_type: str, assistant_id: str, function_name: str
) -> Optional[Callable]:
    try:
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

        configuration = (
            assistant.configuration.__dict__["attribute_values"]
            if assistant.__dict__["attribute_values"].get("configuration")
            else {}
        )
        if assistant_function.get("configuration"):
            configuration = dict(configuration, **assistant_function["configuration"])

        return getattr(
            assistant_function_class(
                logger,
                **Utility.json_loads(Utility.json_dumps(configuration)),
            ),
            function_name,
        )
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


def update_thread_and_insert_message(
    info: ResolveInfo, kwargs: Dict[str, Any], result: Any, role: str
) -> None:
    update_kwargs = {
        "assistant_id": kwargs["assistant_id"],
        "thread_id": result.thread_id,
        "run": {
            "run_id": getattr(result, "current_run_id", None) or result.run_id,
            "usage": getattr(result, "usage", {}),
        },
        "updated_by": kwargs["updated_by"],
    }
    if kwargs.get("thread_id") is None:
        update_kwargs["assistant_type"] = kwargs["assistant_type"]

    insert_update_thread_handler(info, **update_kwargs)

    last_message = resolve_last_message_handler(
        info, thread_id=result.thread_id, role=role
    )

    insert_update_message_handler(
        info,
        thread_id=last_message.thread_id,
        run_id=last_message.run_id,
        message_id=last_message.message_id,
        role=last_message.role,
        message=last_message.message,
        created_at=last_message.created_at,
    )


def assistant_decorator() -> Callable:
    def actual_decorator(original_function: Callable) -> Callable:
        @functools.wraps(original_function)
        def wrapper_function(*args: List, **kwargs: Dict[str, any]) -> Any:
            function_name = original_function.__name__
            try:
                result = original_function(*args, **kwargs)
                if function_name == "resolve_ask_open_ai_handler":
                    update_thread_and_insert_message(args[0], kwargs, result, "user")
                elif (
                    function_name == "resolve_current_run_handler"
                    and result.status == "completed"
                ):
                    update_thread_and_insert_message(
                        args[0], kwargs, result, "assistant"
                    )
                return result
            except Exception as e:
                log = traceback.format_exc()
                args[0].context.get("logger").error(log)
                raise e

        return wrapper_function

    return actual_decorator


def extract_requested_fields(info):
    # Accessing the field nodes to determine the requested fields
    field_nodes = info.field_asts[0].selection_set.selections

    # Function to recursively extract fields from fragments and selections
    def extract_fields(selection_set):
        fields = []
        for selection in selection_set:
            if type(selection).__name__ == "Field":
                fields.append(selection.name.value)
                if selection.selection_set:
                    fields.extend(extract_fields(selection.selection_set.selections))
            elif type(selection).__name__ == "FragmentSpread":
                fragment = info.fragments[selection.name.value]
                fields.extend(extract_fields(fragment.selection_set.selections))
            elif type(selection).__name__ == "InlineFragment":
                fields.extend(extract_fields(selection.selection_set.selections))
            else:
                continue
        return fields

    requested_fields = extract_fields(field_nodes)
    return requested_fields


class EventHandler(AssistantEventHandler):
    def __init__(
        self,
        logger: logging.Logger,
        assistant_type: str,
        queue: Optional[Queue] = None,
        print_progress: bool = False,
    ):
        self.logger = logger
        self.assistant_type = assistant_type
        self.queue = queue
        self.print_progress = print_progress
        AssistantEventHandler.__init__(self)

    @override
    def on_event(self, event: AssistantStreamEvent) -> None:
        self.logger.debug(f"event: {event.event}")
        if event.event == "thread.run.created":
            self.logger.info(f"current_run_id: {event.data.id}")
            if self.queue is not None:
                self.queue.put({"name": "current_run_id", "value": event.data.id})
        if event.event == "thread.run.requires_action":
            self.handle_requires_action(event.data)

    def handle_requires_action(self, data: Any) -> None:
        tool_outputs = []
        for tool in data.required_action.submit_tool_outputs.tool_calls:
            assistant_function = get_assistant_function(
                self.logger, self.assistant_type, data.assistant_id, tool.function.name
            )
            assert (
                assistant_function is not None
            ), f"The function ({tool.function.name}) is not supported!!!"

            arguments = Utility.json_loads(tool.function.arguments)
            tool_outputs.append(
                {
                    "tool_call_id": tool.id,
                    "output": Utility.json_dumps(assistant_function(**arguments)),
                }
            )
        self.submit_tool_outputs(tool_outputs)

    def submit_tool_outputs(self, tool_outputs: List[Dict[str, Any]]) -> None:
        start_time = time.time()
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(self.logger, self.assistant_type),
        ) as stream:
            if self.print_progress:
                for _ in stream.text_deltas:
                    elapsed_time = time.time() - start_time
                    print(
                        f"\rElapsed Time: {elapsed_time:.2f} seconds ({self.current_event.event}).",
                        end="",
                        flush=True,
                    )
                print()  # To move to the next line after completion
            else:
                stream.until_done()


def resolve_file_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> OpenAIFileType:
    requested_fields = extract_requested_fields(info)

    file = client.files.retrieve(kwargs["file_id"])
    openai_file = {
        "id": file.id,
        "object": file.object,
        "filename": file.filename,
        "purpose": file.purpose,
        "created_at": pendulum.from_timestamp(file.created_at, tz="UTC"),
        "bytes": file.bytes,
    }
    if "encodedContent" in requested_fields:
        response: Response = client.files.content(kwargs["file_id"])
        content = response.content  # Get the actual bytes data)
        # Convert the content to a Base64-encoded string
        openai_file["encoded_content"] = base64.b64encode(content).decode("utf-8")

    return OpenAIFileType(**openai_file)


def resolve_files_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> List[OpenAIFileType]:
    if kwargs.get("purpose"):
        file_list = client.files.list(purpose=kwargs["purpose"])
    else:
        file_list = client.files.list()
    return [
        OpenAIFileType(
            id=file.id,
            object=file.object,
            filename=file.filename,
            purpose=file.purpose,
            created_at=pendulum.from_timestamp(file.created_at, tz="UTC"),
            bytes=file.bytes,
        )
        for file in file_list.data
    ]


def insert_file_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> str:
    purpose = kwargs["purpose"]
    encoded_content = kwargs["encoded_content"]
    # Decode the Base64 string
    decoded_content = base64.b64decode(encoded_content)

    # Save the decoded content into a BytesIO object
    content_io = BytesIO(decoded_content)

    # Assign a filename to the BytesIO object
    content_io.name = kwargs["filename"]

    file = client.files.create(file=content_io, purpose=purpose)
    return OpenAIFileType(
        id=file.id,
        object=file.object,
        filename=file.filename,
        purpose=file.purpose,
        created_at=pendulum.from_timestamp(file.created_at, tz="UTC"),
        bytes=file.bytes,
    )


def delete_file_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> None:
    result = client.files.delete(kwargs["file_id"])
    return result.deleted


def resolve_live_messages_handler(
    info: ResolveInfo,
    **kwargs: Dict[str, Any],
) -> List[MessageType]:
    try:
        thread_id = kwargs["thread_id"]
        roles = kwargs.get("roles", ["user", "assistant"])
        order = kwargs.get("order", "asc")
        messages = client.beta.threads.messages.list(thread_id=thread_id, order=order)
        live_messages = []
        for m in messages:
            if m.role not in roles:
                continue

            live_messages.append(
                LiveMessageType(
                    thread_id=m.thread_id,
                    message_id=m.id,
                    created_at=pendulum.from_timestamp(m.created_at, tz="UTC"),
                    role=m.role,
                    message=m.content[0].text.value,
                    run_id=m.run_id,
                )
            )
        return live_messages
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


@assistant_decorator()
def resolve_current_run_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> CurrentRunType:
    try:
        thread_id = kwargs["thread_id"]
        run_id = kwargs["run_id"]
        current_run = client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_id
        )
        return CurrentRunType(
            thread_id=thread_id,
            run_id=run_id,
            status=current_run.status,
            usage=(
                {
                    "prompt_tokens": current_run.usage.prompt_tokens,
                    "completion_tokens": current_run.usage.completion_tokens,
                    "total_tokens": current_run.usage.total_tokens,
                }
                if current_run.usage
                else {}
            ),
        )
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


def resolve_last_message_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> LiveMessageType:
    try:
        thread_id = kwargs["thread_id"]
        role = kwargs["role"]
        last_message = LiveMessageType(
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
            last_message.created_at = pendulum.from_timestamp(
                messages[0].created_at, tz="UTC"
            )
        return last_message
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


def handle_stream(
    logger: logging.Logger,
    thread_id: str,
    assistant_id: str,
    assistant_type: str,
    queue: Queue,
    instructions: str = None,  # Optional parameter added here
) -> None:
    event_handler = EventHandler(logger, assistant_type, queue=queue)
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=event_handler,
        instructions=instructions,  # Pass instructions to the stream if provided
    ) as stream:
        stream.until_done()


def get_current_run_id_and_start_async_task(
    logger: logging.Logger,
    thread_id: str,
    assistant_id: str,
    assistant_type: str,
    instructions: str = None,
) -> str:
    try:
        queue = Queue()
        stream_thread = threading.Thread(
            target=handle_stream,
            args=(logger, thread_id, assistant_id, assistant_type, queue, instructions),
        )
        stream_thread.start()
        q = queue.get()
        if q["name"] == "current_run_id":
            return q["value"]
        raise Exception("Cannot locate the value for current_run_id.")
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


@assistant_decorator()
def resolve_ask_open_ai_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> AskOpenAIType:
    try:
        assistant_type = kwargs["assistant_type"]
        assistant_id = kwargs["assistant_id"]
        user_query = kwargs["user_query"]
        thread_id = kwargs.get("thread_id")
        message = {"role": "user", "content": user_query}
        if kwargs.get("attachments"):
            message["attachments"] = kwargs["attachments"]
        if kwargs.get("message_metadata") is not None:
            message["metadata"] = kwargs["message_metadata"]

        if thread_id is None:
            thread = client.beta.threads.create(
                messages=[message],
                tool_resources=kwargs.get("tool_resources"),
                metadata=kwargs.get("thread_metadata", {}),
            )
            thread_id = thread.id
        else:
            message["thread_id"] = thread_id
            client.beta.threads.messages.create(**message)

        current_run_id = get_current_run_id_and_start_async_task(
            info.context.get("logger"),
            thread_id,
            assistant_id,
            assistant_type,
            instructions=kwargs.get("instructions"),
        )

        return AskOpenAIType(
            assistant_id=kwargs["assistant_id"],
            thread_id=thread_id,
            user_query=kwargs["user_query"],
            current_run_id=current_run_id,
        )

    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_assistant(assistant_type: str, assistant_id: str) -> AssistantModel:
    return AssistantModel.get(assistant_type, assistant_id)


def _get_assistant(assistant_type: str, assistant_id: str) -> Dict[str, Any]:
    _assistant = client.beta.assistants.retrieve(assistant_id)
    assistant = get_assistant(assistant_type, assistant_id)
    return {
        "assistant_type": assistant.assistant_type,
        "assistant_id": assistant.assistant_id,
        "assistant_name": assistant.assistant_name,
        "description": _assistant.description,
        "model": _assistant.model,
        "instructions": _assistant.instructions,
        "tools": _assistant.tools,
        "tool_resources": _assistant.tool_resources,
        "metadata": _assistant.metadata,
        "temperature": _assistant.temperature,
        "top_p": _assistant.top_p,
        "response_format": (
            _assistant.response_format
            if isinstance(_assistant.response_format, str)
            and _assistant.response_format == "auto"
            else _assistant.response_format.type
        ),
        "configuration": assistant.configuration,
        "functions": assistant.functions,
    }


def get_assistant_range_key(info: ResolveInfo, **kwargs: Dict[str, Any]) -> str:
    try:
        assistant = client.beta.assistants.create(
            name=kwargs["assistant_name"],
            description=kwargs.get("description"),
            model=kwargs["model"],
            instructions=kwargs.get("instructions"),
            tools=kwargs.get("tools", []),
            tool_resources=kwargs.get("tool_resources"),
            metadata=kwargs.get("metadata", {}),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            response_format=(
                kwargs.get("response_format", "auto")
                if kwargs.get("response_format", "auto") == "auto"
                else {"type": kwargs["response_format"]}
            ),
        )
        return assistant.id
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


def get_assistant_count(assistant_type: str, assistant_id: str) -> int:
    return AssistantModel.count(
        assistant_type, AssistantModel.assistant_id == assistant_id
    )


def get_assistant_type(info: ResolveInfo, assistant: AssistantModel) -> AssistantType:
    _assistant = client.beta.assistants.retrieve(assistant.assistant_id)
    assistant = assistant.__dict__["attribute_values"]
    assistant["description"] = _assistant.description
    assistant["model"] = _assistant.model
    assistant["instructions"] = _assistant.instructions
    assistant["tools"] = _assistant.tools
    assistant["tool_resources"] = _assistant.tool_resources
    assistant["metadata"] = _assistant.metadata
    assistant["temperature"] = _assistant.temperature
    assistant["top_p"] = _assistant.top_p
    assistant["response_format"] = (
        _assistant.response_format
        if isinstance(_assistant.response_format, str)
        else _assistant.response_format.type
    )

    return AssistantType(**Utility.json_loads(Utility.json_dumps(assistant)))


def resolve_assistant_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> AssistantType:
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
def resolve_assistant_list_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> Any:
    assistant_type = kwargs.get("assistant_type")
    assistant_name = kwargs.get("assistant_name")

    args = []
    inquiry_funct = AssistantModel.scan
    count_funct = AssistantModel.count
    if assistant_type:
        args = [assistant_type, None]
        inquiry_funct = AssistantModel.query

    the_filters = None
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
    range_key_funct=get_assistant_range_key,
    model_funct=get_assistant,
    count_funct=get_assistant_count,
    type_funct=get_assistant_type,
)
def insert_update_assistant_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> None:
    assistant_type = kwargs["assistant_type"]
    assistant_id = kwargs["assistant_id"]
    if kwargs.get("entity") is None:
        AssistantModel(
            assistant_type,
            assistant_id,
            **{
                "assistant_name": kwargs["assistant_name"],
                "configuration": kwargs["configuration"],
                "functions": kwargs["functions"],
                "updated_by": kwargs["updated_by"],
                "created_at": pendulum.now("UTC"),
                "updated_at": pendulum.now("UTC"),
            },
        ).save()
        return

    assistant = kwargs.get("entity")
    actions = [
        AssistantModel.updated_by.set(kwargs.get("updated_by")),
        AssistantModel.updated_at.set(pendulum.now("UTC")),
    ]
    updated_assistant_attributes = {"assistant_id": assistant_id}

    if kwargs.get("assistant_name") is not None:
        updated_assistant_attributes["name"] = kwargs["assistant_name"]
        actions.append(AssistantModel.assistant_name.set(kwargs.get("assistant_name")))
    if kwargs.get("description") is not None:
        updated_assistant_attributes["description"] = kwargs["description"]
    if kwargs.get("model") is not None:
        updated_assistant_attributes["model"] = kwargs["model"]
    if kwargs.get("instructions") is not None:
        updated_assistant_attributes["instructions"] = kwargs["instructions"]
    if kwargs.get("tools") is not None:
        updated_assistant_attributes["tools"] = kwargs["tools"]
    if kwargs.get("tool_resources") is not None:
        updated_assistant_attributes["tool_resources"] = kwargs["tool_resources"]
    if kwargs.get("metadata") is not None:
        updated_assistant_attributes["metadata"] = kwargs["metadata"]
    if kwargs.get("temperature") is not None:
        updated_assistant_attributes["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        updated_assistant_attributes["top_p"] = kwargs["top_p"]
    if kwargs.get("response_format") is not None:
        updated_assistant_attributes["response_format"] = (
            kwargs["response_format"]
            if kwargs["response_format"] == "auto"
            else {"type": kwargs["response_format"]}
        )
    if kwargs.get("configuration") is not None:
        actions.append(AssistantModel.configuration.set(kwargs.get("configuration")))
    if kwargs.get("functions") is not None:
        actions.append(AssistantModel.functions.set(kwargs.get("functions")))

    client.beta.assistants.update(**updated_assistant_attributes)
    assistant.update(actions=actions)


@delete_decorator(
    keys={
        "hash_key": "assistant_type",
        "range_key": "assistant_id",
    },
    model_funct=get_assistant,
)
def delete_assistant_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> bool:
    client.beta.assistants.delete(kwargs.get("assistant_id"))
    kwargs.get("entity").delete()
    return True


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_thread(assistant_id: str, thread_id: str) -> ThreadModel:
    return ThreadModel.get(assistant_id, thread_id)


def get_thread_count(assistant_id: str, thread_id: str) -> int:
    return ThreadModel.count(assistant_id, ThreadModel.thread_id == thread_id)


def get_thread_type(info: ResolveInfo, thread: ThreadModel) -> ThreadType:
    try:
        assistant = _get_assistant(thread.assistant_type, thread.assistant_id)
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").exception(log)
        raise e
    thread = thread.__dict__["attribute_values"]
    thread["assistant"] = assistant
    thread.pop("assistant_type")
    thread.pop("assistant_id")
    return ThreadType(**Utility.json_loads(Utility.json_dumps(thread)))


def resolve_thread_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> ThreadType:
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
def resolve_thread_list_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> Any:
    assistant_id = kwargs.get("assistant_id")
    assistant_types = kwargs.get("assistant_types")

    args = []
    inquiry_funct = ThreadModel.scan
    count_funct = ThreadModel.count
    if assistant_id:
        args = [assistant_id, None]
        inquiry_funct = ThreadModel.query

    the_filters = None
    if assistant_types:
        the_filters &= ThreadModel.assistant_type.is_in(*assistant_types)
    if the_filters is not None:
        args.append(the_filters)

    return inquiry_funct, count_funct, args


def add_or_update_run_in_list(list_of_run_dicts, new_run):
    # Flag to check if the new run was added or updated
    updated = False

    # Iterate over the list to check for existing IDs
    for i, run in enumerate(list_of_run_dicts):
        if run["run_id"] == new_run["run_id"]:
            # Update the existing run with the new run's data
            list_of_run_dicts[i] = new_run
            updated = True
            break

    # If the ID was not found, add the new run to the list
    if not updated:
        list_of_run_dicts.append(new_run)

    return list_of_run_dicts


@insert_update_decorator(
    keys={
        "hash_key": "assistant_id",
        "range_key": "thread_id",
    },
    range_key_required=True,
    model_funct=get_thread,
    count_funct=get_thread_count,
    type_funct=get_thread_type,
)
def insert_update_thread_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> None:
    assistant_id = kwargs["assistant_id"]
    thread_id = kwargs["thread_id"]
    if kwargs.get("entity") is None:
        ThreadModel(
            assistant_id,
            thread_id,
            **{
                "assistant_type": kwargs["assistant_type"],
                "runs": [kwargs["run"]],
                "updated_by": kwargs["updated_by"],
                "created_at": pendulum.now("UTC"),
                "updated_at": pendulum.now("UTC"),
            },
        ).save()
        return

    thread = kwargs.get("entity")
    actions = [
        AssistantModel.updated_by.set(kwargs["updated_by"]),
        AssistantModel.updated_at.set(pendulum.now("UTC")),
    ]
    if kwargs.get("run") is not None:
        actions.append(
            ThreadModel.runs.set(add_or_update_run_in_list(thread.runs, kwargs["run"]))
        )
    thread.update(actions=actions)


@delete_decorator(
    keys={
        "hash_key": "assistant_id",
        "range_key": "thread_id",
    },
    model_funct=get_thread,
)
def delete_thread_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> bool:
    kwargs.get("entity").delete()
    return True


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_message(thread_id: str, message_id: str) -> MessageModel:
    return MessageModel.get(thread_id, message_id)


def get_message_count(thread_id: str, message_id: str) -> int:
    return MessageModel.count(thread_id, MessageModel.message_id == message_id)


def get_message_type(info: ResolveInfo, message: MessageModel) -> MessageType:
    message = message.__dict__["attribute_values"]
    return MessageType(**Utility.json_loads(Utility.json_dumps(message)))


def resolve_message_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> MessageType:
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
def resolve_message_list_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> Any:
    thread_id = kwargs.get("thread_id")
    roles = kwargs.get("roles")
    message = kwargs.get("message")

    args = []
    inquiry_funct = MessageModel.scan
    count_funct = MessageModel.count
    if thread_id:
        args = [thread_id, None]
        inquiry_funct = MessageModel.query

    the_filters = None
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
    range_key_required=True,
    model_funct=get_message,
    count_funct=get_message_count,
    type_funct=get_message_type,
)
def insert_update_message_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> None:
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
        actions.append(MessageModel.message.set(kwargs["message"]))
    if kwargs.get("created_at") is not None:
        actions.append(MessageModel.created_at.set(kwargs["created_at"]))
    message.update(actions=actions)


@delete_decorator(
    keys={
        "hash_key": "thread_id",
        "range_key": "message_id",
    },
    model_funct=get_message,
)
def delete_message_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> bool:
    kwargs.get("entity").delete()
    return True
