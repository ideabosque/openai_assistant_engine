#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import asyncio
import base64
import functools
import json
import logging
import random
import threading
import time
import traceback
import uuid
from io import BytesIO
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

import boto3
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

from .models import (
    AssistantModel,
    AsyncTaskModel,
    FineTuningMessageModel,
    MessageModel,
    ThreadModel,
    ToolCallModel,
)
from .types import (
    AskOpenAIType,
    AssistantListType,
    AssistantType,
    AsyncTaskListType,
    AsyncTaskType,
    CurrentRunType,
    FineTuningMessageListType,
    FineTuningMessageType,
    LiveMessageType,
    MessageListType,
    MessageType,
    OpenAIFileType,
    ThreadListType,
    ThreadType,
    ToolCallListType,
    ToolCallType,
)

client = None
fine_tuning_data_days_limit = None
apigw_client = None
aws_lambda = None
aws_sqs = None
task_queue = None

data_format = "auto"
# Global buffer (queue)
stream_text_deltas_queue = Queue()
# Configurable batch size
stream_text_deltas_batch_size = None

## Test the waters 🧪 before diving in!
##<--Testing Data-->##
endpoint_id = None
connection_id = None
test_mode = None
##<--Testing Data-->##


def handlers_init(logger: logging.Logger, **setting: Dict[str, Any]) -> None:
    global client, fine_tuning_data_days_limit, training_data_rate, apigw_client, aws_lambda, aws_sqs, task_queue, data_format, stream_text_deltas_batch_size, endpoint_id, connection_id, test_mode
    try:
        client = OpenAI(
            api_key=setting["openai_api_key"],
        )
        fine_tuning_data_days_limit = int(setting.get("fine_tuning_data_days_limit", 7))
        training_data_rate = float(setting.get("training_data_rate", 0.6))
        if (
            setting.get("api_id")
            and setting.get("api_stage")
            and setting.get("region_name")
            and setting.get("aws_access_key_id")
            and setting.get("aws_secret_access_key")
            and setting.get("aws_secret_access_key")
        ):
            apigw_client = boto3.client(
                "apigatewaymanagementapi",
                endpoint_url=f"https://{setting['api_id']}.execute-api.{setting['region_name']}.amazonaws.com/{setting['api_stage']}",
                region_name=setting["region_name"],
                aws_access_key_id=setting["aws_access_key_id"],
                aws_secret_access_key=setting["aws_secret_access_key"],
            )
        # Set up AWS credentials in Boto3
        if (
            setting.get("region_name")
            and setting.get("aws_access_key_id")
            and setting.get("aws_secret_access_key")
        ):
            aws_lambda = boto3.client(
                "lambda",
                region_name=setting.get("region_name"),
                aws_access_key_id=setting.get("aws_access_key_id"),
                aws_secret_access_key=setting.get("aws_secret_access_key"),
            )
            aws_sqs = boto3.resource(
                "sqs",
                region_name=setting.get("region_name"),
                aws_access_key_id=setting.get("aws_access_key_id"),
                aws_secret_access_key=setting.get("aws_secret_access_key"),
            )
        else:
            aws_lambda = boto3.client(
                "lambda",
            )
            aws_sqs = boto3.resource(
                "sqs",
            )

        if setting.get("task_queue_name"):
            task_queue = aws_sqs.get_queue_by_name(
                QueueName=setting.get("task_queue_name")
            )
        if setting.get("data_format"):
            data_format = setting.get("data_format")
        stream_text_deltas_batch_size = int(
            setting.get("stream_text_deltas_batch_size", 10)
        )

        ## Test the waters 🧪 before diving in!
        ##<--Testing Data-->##
        endpoint_id = setting.get("endpoint_id")
        connection_id = setting.get("connection_id")
        test_mode = setting.get("test_mode")
        ##<--Testing Data-->##

    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


def send_data_to_websocket_handler(
    logger: logging.Logger, **kwargs: Dict[str, Any]
) -> None:
    try:
        global apigw_client

        # Send the message to the WebSocket client using the connection ID
        connection_id = kwargs["connection_id"]
        data = kwargs["data"]
        apigw_client.post_to_connection(
            ConnectionId=connection_id, Data=Utility.json_dumps(data)
        )

        return True
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


# Utility Function: Validate and Complete JSON
def try_complete_json(accumulated: str) -> str:
    """
    Try to validate and complete the JSON by adding various combinations of closing brackets.

    Args:
        accumulated (str): The accumulated JSON string that may be incomplete.

    Returns:
        str: The ending string that successfully completes the JSON, or an empty string if no valid completion is found.
    """
    possible_endings = ["}", "]", "}", "}]", "}]}", "}}"]
    for ending in possible_endings:
        try:
            completed_json = accumulated + ending
            json.loads(completed_json)  # Attempt to parse with the potential ending
            return ending  # If parsing succeeds, return the successful ending
        except json.JSONDecodeError:
            continue
    return ""  # If no ending works, return empty string


# Helper Function: Process and Send JSON
def process_and_send_json(
    logger: logging.Logger,
    partial_json_accumulator: str,
    complete_json_accumulator: str,
    connection_id: str,
    endpoint_id: str,
    message_group_id: str,
    setting: Dict[str, Any],
) -> str:
    """
    Process and send JSON if it forms a valid structure.

    Args:
        logger: Logger instance for logging.
        partial_json_accumulator: The accumulated partial JSON string.
        complete_json_accumulator: The complete JSON string for validation.
        connection_id: WebSocket connection ID.
        endpoint_id: AWS Lambda endpoint identifier.
        message_group_id: Unique identifier for message grouping.
        setting: Configuration settings.

    Returns:
        str: The updated complete JSON string after sending the partial JSON.
    """
    combined_json = complete_json_accumulator + partial_json_accumulator
    ending = try_complete_json(combined_json)

    if ending:
        print(partial_json_accumulator, flush=True)  # Print the JSON to console
        if connection_id and message_group_id:
            invoke_funct_on_aws_lambda(
                logger,
                endpoint_id,
                "send_data_to_websocket",
                params={
                    "connection_id": connection_id,
                    "data": {"text_delta": partial_json_accumulator},
                },
                message_group_id=message_group_id,
                setting=setting,
            )
        complete_json_accumulator += partial_json_accumulator  # Update complete JSON
        partial_json_accumulator = ""  # Reset the partial JSON accumulator

    return complete_json_accumulator, partial_json_accumulator


# JSON Processing Loop
def json_processing_loop(
    logger: logging.Logger,
    endpoint_id: str,
    setting: Dict[str, Any],
    connection_id: str,
    message_group_id: str,
) -> None:
    """
    JSON processing loop to handle JSON-formatted text deltas.

    Args:
        logger: Logger instance to log messages.
        endpoint_id: AWS Lambda endpoint identifier.
        setting: Configuration settings for processing.
        connection_id: WebSocket connection ID.
        stream_text_deltas_queue: Queue from which to consume the text deltas.
        message_group_id: Message group ID for the WebSocket connection.
    """
    complete_json_accumulator, partial_json_accumulator = "", ""
    timeout = 60

    while True:
        try:
            item = stream_text_deltas_queue.get(timeout=timeout)
            stream_text_deltas_queue.task_done()
            timeout = 1

            # Parse and accumulate
            json_data = Utility.json_loads(item)
            text_delta = json_data.get("text_delta", "")
            partial_json_accumulator += text_delta

            # Process and send if it forms a complete JSON structure
            complete_json_accumulator, partial_json_accumulator = process_and_send_json(
                logger,
                partial_json_accumulator,
                complete_json_accumulator,
                connection_id,
                endpoint_id,
                message_group_id,
                setting,
            )

        except Empty:
            # Send the last fragment without examination
            if partial_json_accumulator:
                print(partial_json_accumulator, flush=True)
                if connection_id and message_group_id:
                    invoke_funct_on_aws_lambda(
                        logger,
                        endpoint_id,
                        "send_data_to_websocket",
                        params={
                            "connection_id": connection_id,
                            "data": {"text_delta": partial_json_accumulator},
                        },
                        message_group_id=message_group_id,
                        setting=setting,
                    )
                complete_json_accumulator += partial_json_accumulator
                partial_json_accumulator = ""

            logger.info(
                "No more items to process in JSON format. Consumer is stopping."
            )
            break


# Batch Processing Loop
def batch_processing_loop(
    logger: logging.Logger,
    endpoint_id: str,
    setting: Dict[str, Any],
    connection_id: str,
    message_group_id: str,
) -> None:
    """
    Batch processing loop to handle batched text deltas.

    Args:
        logger: Logger instance to log messages.
        endpoint_id: AWS Lambda endpoint identifier.
        setting: Configuration settings for processing.
        connection_id: WebSocket connection ID.
        stream_text_deltas_queue: Queue from which to consume the text deltas.
        message_group_id: Message group ID for the WebSocket connection.
        stream_text_deltas_batch_size: Size of each batch to process.
    """
    timeout = 60

    while True:
        stream_text_deltas_batch = []

        # Collect items up to the specified batch size
        while len(stream_text_deltas_batch) < stream_text_deltas_batch_size:
            try:
                item = stream_text_deltas_queue.get(timeout=timeout)
                stream_text_deltas_batch.append(item)
                stream_text_deltas_queue.task_done()
                timeout = 1
            except Empty:
                break

        # Process the batch if we have collected any items
        if stream_text_deltas_batch:
            assembled_text_delta = "".join(
                Utility.json_loads(item).get("text_delta", "")
                for item in stream_text_deltas_batch
            )
            print(assembled_text_delta.strip(), end="", flush=True)

            if connection_id and message_group_id:
                invoke_funct_on_aws_lambda(
                    logger,
                    endpoint_id,
                    "send_data_to_websocket",
                    params={
                        "connection_id": connection_id,
                        "data": {"text_delta": assembled_text_delta},
                    },
                    message_group_id=message_group_id,
                    setting=setting,
                )
        else:
            logger.info(
                "No more items to process in batch format. Consumer is stopping."
            )
            break


# Main Function: Stream Text Deltas Consumer
def stream_text_deltas_consumer(
    logger: logging.Logger,
    task_uuid: str,
    consumer_event: threading.Event,
    endpoint_id: str,
    setting: Dict[str, Any],
    connection_id: str,
) -> None:
    """
    Consumes and processes data from the queue in batches or JSON format based on the configuration.

    Args:
        logger: Logger instance to log messages.
        task_uuid: Task identifier.
        endpoint_id: Endpoint identifier for invoking AWS Lambda.
        setting: Configuration settings for processing.
        connection_id: WebSocket connection ID.
        stream_text_deltas_queue: Queue from which to consume the text deltas.
        stream_text_deltas_batch_size: Size of each batch to process. Ignored if data_format is 'json'.
        data_format: Format to process data ('batch' or 'json'). Default is 'batch'.
    """
    try:
        message_group_id = (
            f"{connection_id}-{str(uuid.uuid1().int >> 64)}" if connection_id else None
        )

        # Decide which processing loop to use based on the data format
        if data_format == "json":
            json_processing_loop(
                logger,
                endpoint_id,
                setting,
                connection_id,
                message_group_id,
            )
        else:
            batch_processing_loop(
                logger,
                endpoint_id,
                setting,
                connection_id,
                message_group_id,
            )
        consumer_event.set()

    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        insert_update_async_task(
            "async_openai_assistant_stream", task_uuid, status="fail", log=log
        )
        consumer_event.set()
        raise e


def invoke_funct_on_local(
    logger: logging.Logger,
    setting: Dict[str, Any],
    funct: str,
    **params: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        funct_on_local = setting["functs_on_local"].get(funct)
        assert funct_on_local is not None, f"Function ({funct}) not found."

        Utility.invoke_funct_on_local(
            logger,
            funct,
            funct_on_local,
            setting,
            **params,
        )
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


def invoke_funct_on_aws_lambda(
    logger: logging.Logger,
    endpoint_id: str,
    funct: str,
    params: Dict[str, Any] = {},
    message_group_id: str = None,
    setting: Dict[str, Any] = None,
) -> Dict[str, Any]:

    ## Test the waters 🧪 before diving in!
    ##<--Testing Function-->##
    if test_mode:
        if test_mode == "local_for_all":
            # Jump to the local function if these conditions meet.
            return invoke_funct_on_local(logger, setting, funct, **params)
        elif (
            test_mode == "local_for_sqs" and not message_group_id
        ):  # Test websocket callback with SQS from local.
            # Jump to the local function if these conditions meet.
            return invoke_funct_on_local(logger, setting, funct, **params)
        elif (
            test_mode == "local_for_aws_lambda" and task_queue is None
        ):  # Test AWS Lambda calls from local.
            pass
    ##<--Testing Function-->##

    # When we have both a message group and a task queue, hit the SQS 📨
    if message_group_id and task_queue:
        Utility.invoke_funct_on_aws_sqs(
            logger,
            task_queue,
            message_group_id,
            **{
                "endpoint_id": endpoint_id,
                "funct": funct,
                "params": params,
            },
        )
        return  # No need to proceed after sending the SQS message.

    # If we're at the top-level, let's call the AWS Lambda directly 💻
    Utility.invoke_funct_on_aws_lambda(
        logger,
        aws_lambda,
        **{
            "endpoint_id": endpoint_id,
            "funct": funct,
            "params": params,
        },
    )
    return


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
        "assistant_type": kwargs.get("assistant_type"),
        "assistant_id": kwargs["assistant_id"],
        "thread_id": result.thread_id,
        "run": {
            "run_id": getattr(result, "current_run_id", None) or result.run_id,
            "usage": getattr(result, "usage", kwargs.get("usage", {})),
        },
        "updated_by": kwargs["updated_by"],
    }
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
                if kwargs.get("function_name"):
                    async_task = get_async_task(
                        kwargs["function_name"], kwargs["task_uuid"]
                    )
                    assert async_task.status in [
                        "in_progress",
                        "completed",
                    ], f"{function_name}:{kwargs['function_name']} is at the status ({async_task.status}) with the log {async_task.log}."

                result = original_function(*args, **kwargs)
                if function_name == "resolve_ask_open_ai_handler":
                    update_thread_and_insert_message(args[0], kwargs, result, "user")
                elif function_name == "resolve_current_run_handler":
                    if result.status == "completed":
                        update_thread_and_insert_message(
                            args[0], kwargs, result, "assistant"
                        )
                        args[0].context.get("logger").info(
                            f"run_id: {result.run_id} is completed at {time.strftime('%X')}."
                        )
                    elif result.status == "fail":
                        raise Exception(
                            f"run_id: {result.run_id} is fail at {time.strftime('%X')}."
                        )
                # Update the status for the thread when using websocket.
                if args[0].context.get("connectionId"):
                    current_run = client.beta.threads.runs.retrieve(
                        thread_id=result.thread_id, run_id=result.current_run_id
                    )
                    if current_run.status == "completed":
                        kwargs["usage"] = {
                            "prompt_tokens": current_run.usage.prompt_tokens,
                            "completion_tokens": current_run.usage.completion_tokens,
                            "total_tokens": current_run.usage.total_tokens,
                        }
                        update_thread_and_insert_message(
                            args[0], kwargs, result, "assistant"
                        )
                        args[0].context.get("logger").info(
                            f"run_id: {result.current_run_id} is completed at {time.strftime('%X')}."
                        )
                    elif current_run.status == "fail":
                        raise Exception(
                            f"run_id: {result.current_run_id} is fail at {time.strftime('%X')}."
                        )

                return result
            except Exception as e:
                log = traceback.format_exc()
                args[0].context.get("logger").error(log)
                raise e

        return wrapper_function

    return actual_decorator


def insert_update_async_task(
    function_name: str,
    task_uuid: str,
    status: str = None,
    arguments: dict[str, any] = None,
    result: str = None,
    log: str = None,
) -> None:
    if status is None:
        async_task = AsyncTaskModel(
            function_name,
            task_uuid,
            **{
                "created_at": pendulum.now("UTC"),
                "updated_at": pendulum.now("UTC"),
            },
        )
    else:
        async_task = get_async_task(function_name, task_uuid)
        if status == "completed" and async_task.status == "fail":
            return  # Skip the process if the task is already fail.

        async_task.status = status
        async_task.updated_at = pendulum.now("UTC")

    if arguments is not None:
        async_task.arguments = arguments

    if result is not None:
        async_task.result = result

    if log is not None:
        async_task.log = log

    async_task.save()


def async_task_decorator() -> Callable:
    def actual_decorator(original_function: Callable) -> Callable:
        @functools.wraps(original_function)
        def wrapper_function(*args: List, **kwargs: Dict[str, any]) -> Any:
            function_name = original_function.__name__
            task_uuid = args[1]
            try:
                args[0].info(
                    f"task_uuid: {task_uuid} is started at {time.strftime('%X')}."
                )

                ## insert an entry into sync_tasks table.
                insert_update_async_task(function_name, task_uuid, arguments=args[-1])

                result = original_function(*args, **kwargs)

                insert_update_async_task(function_name, task_uuid, status="completed")

                ## Update the status with result for the entry.
                insert_update_async_task(
                    function_name,
                    task_uuid,
                    status="completed",
                    result=(
                        Utility.json_dumps(result)
                        if result is not None and not isinstance(result, bool)
                        else None
                    ),
                )

                args[0].info(
                    f"task_uuid: {task_uuid} is completed at {time.strftime('%X')}."
                )

                return
            except Exception as e:
                log = traceback.format_exc()
                args[0].error(log)

                ## Update the status with log for the entry.
                insert_update_async_task(
                    function_name, task_uuid, status="fail", log=log
                )

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
    ):
        self.logger = logger
        self.assistant_type = assistant_type
        self.queue = queue
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
                self.logger,
                self.assistant_type,
                data.assistant_id,
                tool.function.name,
            )
            assert (
                assistant_function is not None
            ), f"The function ({tool.function.name}) is not supported!!!"

            arguments = Utility.json_loads(tool.function.arguments)
            output = assistant_function(**arguments)

            run_id = data.id
            tool_call_id = tool.id
            tool_call = ToolCallModel(
                run_id,
                tool_call_id,
                **{
                    "tool_type": "function",
                    "name": tool.function.name,
                    "arguments": Utility.json_loads(
                        Utility.json_dumps(arguments), parser_number=False
                    ),
                    "created_at": pendulum.from_timestamp(data.started_at, tz="UTC"),
                },
            )
            if output is not None:
                tool_call.content = Utility.json_dumps(output)
            tool_call.save()

            tool_outputs.append(
                {
                    "tool_call_id": tool.id,
                    "output": Utility.json_dumps(output),
                }
            )
        self.submit_tool_outputs(tool_outputs)

    def submit_tool_outputs(self, tool_outputs: List[Dict[str, Any]]) -> None:
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(self.logger, self.assistant_type),
        ) as stream:
            # Print, flush, and send each `text_delta` via WebSocket
            for text in stream.text_deltas:
                stream_text_deltas_queue.put(json.dumps({"text_delta": text}))


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


def insert_file_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> OpenAIFileType:
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


@async_task_decorator()
def async_openai_assistant_stream(
    logger: logging.Logger,
    task_uuid: str,
    stream_event: threading.Event,
    queue: Queue,
    arguments: Dict[str, Any],
) -> None:
    try:
        event_handler = EventHandler(
            logger,
            arguments["assistant_type"],
            queue=queue,
        )
        with client.beta.threads.runs.stream(
            thread_id=arguments["thread_id"],
            assistant_id=arguments["assistant_id"],
            event_handler=event_handler,
            instructions=arguments.get(
                "instructions"
            ),  # Pass instructions to the stream if provided
        ) as stream:
            stream.until_done()
        stream_event.set()
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        insert_update_async_task(
            "async_openai_assistant_stream", task_uuid, status="fail", log=log
        )
        stream_event.set()
        raise e


def async_openai_assistant_stream_handler(
    logger: logging.Logger,
    **kwargs: Dict[str, Any],
) -> None:
    try:
        global data_format
        endpoint_id = kwargs.get("endpoint_id")
        task_uuid = kwargs["task_uuid"]
        arguments = kwargs["arguments"]
        connection_id = kwargs.get("connection_id")
        setting = kwargs.get("setting")

        if data_format == "auto":
            data_format = "batch"
            _assistant = client.beta.assistants.retrieve(arguments["assistant_id"])
            response_format = _get_assistant_response_format(_assistant)
            if response_format in ["json_object", "json_schema"]:
                data_format = "json"

        stream_queue = Queue()
        stream_event = threading.Event()
        stream_thread = threading.Thread(
            target=async_openai_assistant_stream,
            args=(logger, task_uuid, stream_event, stream_queue, arguments),
        )
        stream_thread.start()

        consumer_event = threading.Event()
        consumer_thread = threading.Thread(
            target=stream_text_deltas_consumer,
            args=(
                logger,
                task_uuid,
                consumer_event,
                endpoint_id,
                setting,
                connection_id,
            ),
        )
        consumer_thread.start()

        q = stream_queue.get()
        assert (
            q["name"] == "current_run_id"
        ), "Cannot locate the value for current_run_id."
        current_run_id = q["value"]
        insert_update_async_task(
            "async_openai_assistant_stream",
            task_uuid,
            status="in_progress",
            result=Utility.json_dumps({"current_run_id": current_run_id}),
        )

        # Wait for the thread to complete using join or check if it is alive
        stream_event.wait()
        logger.info(f"async_openai_assistant_stream is done!!")

        consumer_event.wait()
        logger.info(f"stream_text_deltas_consumer is done!!")

        return True
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        insert_update_async_task(
            "async_openai_assistant_stream", task_uuid, status="fail", log=log
        )
        raise e


def get_current_run_id_and_start_async_task(
    info: ResolveInfo,
    **arguments: Dict[str, Any],
) -> str:
    try:
        task_uuid = str(uuid.uuid1().int >> 64)
        params = {
            "task_uuid": task_uuid,
            "arguments": arguments,
        }
        if info.context.get("connectionId"):
            params["connection_id"] = info.context.get("connectionId")

        invoke_funct_on_aws_lambda(
            info.context["logger"],
            info.context["endpoint_id"],
            "async_openai_assistant_stream",
            params=params,
            setting=info.context["setting"],
        )

        async_task_initiated = False
        inspect_count = 0
        while True:
            if inspect_count > 1000:
                raise Exception("Timeout Error")

            if async_task_initiated:
                async_task = get_async_task("async_openai_assistant_stream", task_uuid)
                if async_task.status in ["in_progress", "completed"]:
                    break
                elif async_task.status == "fail":
                    raise Exception(async_task.log)

            count = get_async_task_count("async_openai_assistant_stream", task_uuid)
            async_task_initiated = True if count == 1 else False

            if not async_task_initiated:
                time.sleep(0.05)

            inspect_count += 1

        current_run_id = Utility.json_loads(async_task.result)["current_run_id"]
        return "async_openai_assistant_stream", task_uuid, current_run_id

    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


def get_thread_id(info: ResolveInfo, **kwargs: Dict[str, Any]) -> str:
    try:
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

        return thread_id
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


@assistant_decorator()
def resolve_ask_open_ai_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> AskOpenAIType:
    try:
        ## Test the waters 🧪 before diving in!
        ##<--Testing Data-->##
        global connection_id, endpoint_id
        if info.context.get("connectionId") is None:
            info.context["connectionId"] = connection_id
        if info.context.get("endpoint_id") is None:
            info.context["endpoint_id"] = endpoint_id
        ##<--Testing Data-->##

        assistant_type = kwargs["assistant_type"]
        assistant_id = kwargs["assistant_id"]
        thread_id = get_thread_id(info, **kwargs)

        function_name, task_uuid, current_run_id = (
            get_current_run_id_and_start_async_task(
                info,
                **{
                    "thread_id": thread_id,
                    "assistant_id": assistant_id,
                    "assistant_type": assistant_type,
                    "instructions": kwargs.get("instructions"),
                    "updated_by": kwargs["updated_by"],
                },
            )
        )

        return AskOpenAIType(
            assistant_id=kwargs["assistant_id"],
            thread_id=thread_id,
            user_query=kwargs["user_query"],
            function_name=function_name,
            task_uuid=task_uuid,
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
            else (
                _assistant.response_format["type"]
                if isinstance(_assistant.response_format, dict)
                else _assistant.response_format.type
            )
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


def _get_assistant_response_format(assistant: object) -> str:
    return (
        assistant.response_format
        if isinstance(assistant.response_format, str)
        else (
            assistant.response_format["type"]
            if isinstance(assistant.response_format, dict)
            else assistant.response_format.type
        )
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
    assistant["response_format"] = _get_assistant_response_format(_assistant)

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


def _get_thread(assistant_id: str, thread_id: str) -> Dict[str, Any]:
    thread = get_thread(assistant_id, thread_id)
    return {
        "assistant_id": thread.assistant_id,
        "thread_id": thread.thread_id,
        "assistant_type": thread.assistant_type,
        "run": thread.runs,
    }


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


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_tool_call(run_id: str, tool_call_id: str) -> ToolCallModel:
    return ToolCallModel.get(run_id, tool_call_id)


def get_tool_call_count(run_id: str, tool_call_id: str) -> int:
    return ToolCallModel.count(run_id, ToolCallModel.tool_call_id == tool_call_id)


def get_tool_call_type(info: ResolveInfo, tool_call: ToolCallModel) -> ToolCallType:
    return ToolCallType(**Utility.json_loads(Utility.json_dumps(tool_call)))


def resolve_tool_call_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> ToolCallType:
    return get_tool_call_type(
        info,
        get_tool_call(kwargs.get("run_id"), kwargs.get("tool_call_id")),
    )


@monitor_decorator
@resolve_list_decorator(
    attributes_to_get=["run_id", "tool_call_id"],
    list_type_class=ToolCallListType,
    type_funct=get_tool_call_type,
)
def resolve_tool_call_list_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> Any:
    run_id = kwargs.get("run_id")
    tool_types = kwargs.get("tool_types")
    name = kwargs.get("name")

    args = []
    inquiry_funct = ToolCallModel.scan
    count_funct = ToolCallModel.count
    if run_id:
        args = [run_id, None]
        inquiry_funct = ToolCallModel.query

    the_filters = None
    if tool_types:
        the_filters &= ToolCallModel.tool_type.is_in(*tool_types)
    if name:
        the_filters &= ToolCallModel.name.contains(name)
    if the_filters is not None:
        args.append(the_filters)

    return inquiry_funct, count_funct, args


@insert_update_decorator(
    keys={
        "hash_key": "run_id",
        "range_key": "tool_call_id",
    },
    range_key_required=True,
    model_funct=get_tool_call,
    count_funct=get_tool_call_count,
    type_funct=get_tool_call_type,
)
def insert_update_tool_call_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> None:
    run_id = kwargs["run_id"]
    tool_call_id = kwargs["tool_call_id"]
    cols = {
        "tool_type": kwargs["tool_type"],
        "name": kwargs["name"],
        "arguments": kwargs["arguments"],
        "created_at": kwargs["created_at"],
    }
    if kwargs.get("content") is not None:
        cols["content"] = kwargs["content"]
    if kwargs.get("entity") is None:
        ToolCallModel(
            run_id,
            tool_call_id,
            **cols,
        ).save()
        return

    tool_call = kwargs.get("entity")
    actions = []
    if kwargs.get("tool_type") is not None:
        actions.append(ToolCallModel.tool_type.set(kwargs["tool_type"]))
    if kwargs.get("name") is not None:
        actions.append(ToolCallModel.name.set(kwargs["name"]))
    if kwargs.get("arguments") is not None:
        actions.append(ToolCallModel.arguments.set(kwargs["arguments"]))
    if kwargs.get("content") is not None:
        actions.append(ToolCallModel.content.set(kwargs["content"]))
    tool_call.update(actions=actions)


@delete_decorator(
    keys={
        "hash_key": "run_id",
        "range_key": "tool_call_id",
    },
    model_funct=get_tool_call,
)
def delete_tool_call_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> bool:
    kwargs.get("entity").delete()
    return True


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_fine_tuning_message(
    assistant_id: str, message_uuid: str
) -> FineTuningMessageModel:
    return FineTuningMessageModel.get(assistant_id, message_uuid)


def get_fine_tuning_message_count(assistant_id: str, message_uuid: str) -> int:
    return FineTuningMessageModel.count(
        assistant_id, FineTuningMessageModel.message_uuid == message_uuid
    )


def get_fine_tuning_message_type(
    info: ResolveInfo, fine_tuning_message: FineTuningMessageModel
) -> FineTuningMessageType:
    try:
        thread = _get_thread(
            fine_tuning_message.assistant_id, fine_tuning_message.thread_id
        )
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").exception(log)
        raise e
    fine_tuning_message = fine_tuning_message.__dict__["attribute_values"]
    fine_tuning_message["thread"] = thread
    fine_tuning_message.pop("thread_id")
    fine_tuning_message.pop("assistant_id")
    return FineTuningMessageType(
        **Utility.json_loads(Utility.json_dumps(fine_tuning_message))
    )


def resolve_fine_tuning_message_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> FineTuningMessageType:
    return get_fine_tuning_message_type(
        info,
        get_fine_tuning_message(kwargs.get("assistant_id"), kwargs.get("message_uuid")),
    )


def upload_fine_tune_file_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> List[OpenAIFileType]:
    try:
        assistant_type = kwargs["assistant_type"]
        assistant_id = kwargs["assistant_id"]
        from_date = kwargs["from_date"]
        to_date = kwargs.get("to_date")

        assistant = get_assistant_type(
            info, get_assistant(assistant_type, assistant_id)
        )

        # Convert the date to UTC
        utc_from_date = pendulum.instance(from_date).in_timezone("UTC")

        # Convert the UTC date to a Unix timestamp
        from_timestamp = int(time.mktime(utc_from_date.timetuple()))

        utc_to_date = None
        if to_date is not None:
            utc_to_date = pendulum.instance(to_date).in_timezone("UTC")

        if utc_to_date is None:
            # Get current UTC time using pendulum
            utc_to_date = pendulum.now("UTC")

        # Convert the UTC date to a Unix timestamp
        to_timestamp = int(time.mktime(utc_to_date.timetuple()))

        results = FineTuningMessageModel.timestamp_index.query(
            assistant_id,
            FineTuningMessageModel.timestamp.between(from_timestamp, to_timestamp),
            filter_condition=FineTuningMessageModel.trained == False,
            attributes_to_get=["assistant_id", "message_uuid"],
        )
        keys_to_retrieve = [
            (
                result.assistant_id,
                result.message_uuid,
            )
            for result in results
        ]
        # Batch get data
        fine_tuning_messages = []

        for fine_tuning_message in FineTuningMessageModel.batch_get(keys_to_retrieve):
            fine_tuning_messages.append(
                {
                    "assistant_id": fine_tuning_message.assistant_id,
                    "message_uuid": fine_tuning_message.message_uuid,
                    "thread_id": fine_tuning_message.thread_id,
                    "timestamp": fine_tuning_message.timestamp,
                    "role": fine_tuning_message.role,
                    "tool_calls": fine_tuning_message.tool_calls,
                    "tool_call_id": fine_tuning_message.tool_call_id,
                    "content": fine_tuning_message.content,
                    "weight": fine_tuning_message.weight,
                }
            )

        trained_message_uuids = [
            fine_tuning_message["message_uuid"]
            for fine_tuning_message in fine_tuning_messages
        ]
        # Sort the messages by timestamp
        fine_tuning_messages_sorted = sorted(
            fine_tuning_messages, key=lambda x: x["timestamp"]
        )

        # Retrieve the distinct list of thread_ids, keeping order by timestamp
        distinct_thread_ids_ordered = []
        seen_thread_ids = set()

        for fine_tuning_message in fine_tuning_messages_sorted:
            if fine_tuning_message["thread_id"] not in seen_thread_ids:
                distinct_thread_ids_ordered.append(fine_tuning_message["thread_id"])
                seen_thread_ids.add(fine_tuning_message["thread_id"])

        thread_fine_tuning_messages = []

        # Loop through the distinct thread IDs
        for thread_id in distinct_thread_ids_ordered:
            info.context.get("logger").info(f"Thread Id: {thread_id}")
            # Filter messages for the current thread ID
            fine_tuning_messages_filtered = list(
                filter(
                    lambda x: x["thread_id"] == thread_id, fine_tuning_messages_sorted
                )
            )

            # Sort by timestamp and ensure 'assistant' comes before 'tool' when timestamps are identical
            fine_tuning_messages_filtered_sorted = sorted(
                fine_tuning_messages_filtered,
                key=lambda x: (
                    x["timestamp"],
                    0 if x["role"] == "assistant" else 1,
                ),  # Assistant first if same timestamp
            )

            messages = []

            for fine_tuning_message in fine_tuning_messages_filtered_sorted:
                # For 'user' and 'system' roles, append message with content and timestamp
                if fine_tuning_message["role"] in ["user", "system"]:
                    messages.append(
                        {
                            "role": fine_tuning_message["role"],
                            "content": fine_tuning_message["content"],
                        }
                    )

                # For 'assistant' role, append with tool_calls if available, otherwise with content
                if fine_tuning_message["role"] == "assistant":
                    if fine_tuning_message.get("tool_calls"):
                        messages.append(
                            {
                                "role": fine_tuning_message["role"],
                                "tool_calls": fine_tuning_message["tool_calls"],
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": fine_tuning_message["role"],
                                "content": fine_tuning_message["content"],
                                "weight": fine_tuning_message["weight"],
                            }
                        )

                # For 'tool' role, append message with tool_call_id and content
                if fine_tuning_message["role"] == "tool":
                    messages.append(
                        {
                            "role": fine_tuning_message["role"],
                            "tool_call_id": fine_tuning_message["tool_call_id"],
                            "content": fine_tuning_message["content"],
                        }
                    )

            # Append the sorted messages to the thread_fine_tuning_messages list
            thread_fine_tuning_messages.append(
                {"messages": messages, "tools": assistant.tools}
            )

        # Randomly assign to training or validation dataset (e.g., 80% chance for training, 20% for validation)
        training_data = []
        validation_data = []
        for thread_fine_tuning_message in thread_fine_tuning_messages:
            if random.random() < 0.8:
                training_data.append(thread_fine_tuning_message)
            else:
                validation_data.append(thread_fine_tuning_message)

        # Step 1: Create the JSONL formatted string
        training_jsonl_content = "\n".join(
            [
                json.dumps(thread_fine_tuning_message)
                for thread_fine_tuning_message in training_data
            ]
        )

        validation_jsonl_content = "\n".join(
            [
                json.dumps(thread_fine_tuning_message)
                for thread_fine_tuning_message in validation_data
            ]
        )

        # Step 2: Encode the JSONL string into bytes (UTF-8 encoding)
        training_jsonl_bytes = training_jsonl_content.encode("utf-8")

        validation_jsonl_bytes = validation_jsonl_content.encode("utf-8")

        # Step 3: Base64 encode the byte data
        traning_base64_encoded_content = base64.b64encode(training_jsonl_bytes)

        validation_base64_encoded_content = base64.b64encode(validation_jsonl_bytes)

        training_fine_tune_file = insert_file_handler(
            info,
            **{
                "purpose": "fine-tune",
                "filename": f"training_{assistant_id}-{str(uuid.uuid1().int >> 64)}.jsonl",
                "encoded_content": traning_base64_encoded_content.decode("utf-8"),
            },
        )

        validation_fine_tune_file = insert_file_handler(
            info,
            **{
                "purpose": "fine-tune",
                "filename": f"validation_{assistant_id}-{str(uuid.uuid1().int >> 64)}.jsonl",
                "encoded_content": validation_base64_encoded_content.decode("utf-8"),
            },
        )

        async_task = insert_update_fine_tuning_messages_handler(
            info,
            **{
                "assistant_type": assistant_type,
                "assistant_id": assistant_id,
                "trained_message_uuids": trained_message_uuids,
            },
        )
        info.context.get("logger").info(async_task)

        return [training_fine_tune_file, validation_fine_tune_file]

    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").exception(log)
        raise e


@monitor_decorator
@resolve_list_decorator(
    attributes_to_get=["assistant_id", "message_uuid", "thread_id", "timestamp"],
    list_type_class=FineTuningMessageListType,
    type_funct=get_fine_tuning_message_type,
)
def resolve_fine_tuning_message_list_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> Any:
    assistant_id = kwargs.get("assistant_id")
    thread_id = kwargs.get("thread_id")
    roles = kwargs.get("roles")
    trained = kwargs.get("trained")
    from_date = kwargs.get("from_date")
    to_date = kwargs.get("to_date")
    if from_date is not None:
        # Convert the date to UTC
        utc_from_date = pendulum.instance(from_date).in_timezone("UTC")

        # Convert the UTC date to a Unix timestamp
        from_timestamp = int(time.mktime(utc_from_date.timetuple()))

    if to_date is None:
        # Get current UTC time using pendulum
        current_utc = pendulum.now("UTC")

        # Convert the current UTC time to a Unix timestamp
        to_timestamp = int(time.mktime(current_utc.timetuple()))
    else:
        # Convert the date to UTC
        utc_to_date = pendulum.instance(to_date).in_timezone("UTC")

        # Convert the UTC date to a Unix timestamp
        to_timestamp = int(time.mktime(utc_to_date.timetuple()))

    args = []
    inquiry_funct = FineTuningMessageModel.scan
    count_funct = FineTuningMessageModel.count
    if assistant_id:
        args = [assistant_id, None]
        inquiry_funct = FineTuningMessageModel.query
        if thread_id:
            inquiry_funct = FineTuningMessageModel.thread_id_index.query
            args[1] = FineTuningMessageModel.thread_id == thread_id
            count_funct = FineTuningMessageModel.thread_id_index.count
        if from_date and to_date:
            inquiry_funct = FineTuningMessageModel.timestamp_index.query
            args[1] = FineTuningMessageModel.timestamp.between(
                from_timestamp, to_timestamp
            )
            count_funct = FineTuningMessageModel.timestamp_index.count

    the_filters = None
    if roles:
        the_filters &= FineTuningMessageModel.role.is_in(*roles)
    if trained is not None:
        the_filters &= FineTuningMessageModel.trained == trained
    if the_filters is not None:
        args.append(the_filters)

    return inquiry_funct, count_funct, args


async def async_insert_update_fine_tuning_message(
    assistant_id, retrain, raw_fine_tuning_message
):
    results = FineTuningMessageModel.timestamp_index.query(
        assistant_id,
        FineTuningMessageModel.timestamp == raw_fine_tuning_message["timestamp"],
        filter_condition=(
            FineTuningMessageModel.role == raw_fine_tuning_message["role"]
        ),
        attributes_to_get=["assistant_id", "message_uuid"],
        limit=1,
    )

    results = [result for result in results]
    if len(results) == 0:
        FineTuningMessageModel(**raw_fine_tuning_message).save()
        return raw_fine_tuning_message["message_uuid"]

    if retrain is False:
        return raw_fine_tuning_message["message_uuid"]

    fine_tuning_message = get_fine_tuning_message(assistant_id, results[0].message_uuid)
    if fine_tuning_message.trained is False:
        return raw_fine_tuning_message["message_uuid"]

    fine_tuning_message.trained = False
    fine_tuning_message.save()
    return raw_fine_tuning_message["message_uuid"]


# Use an asyncio task to handle the event loop instead of running asyncio.run() inside threads
async def task_wrapper(assistant_id, retrain, raw_fine_tuning_message):
    return await async_insert_update_fine_tuning_message(
        assistant_id, retrain, raw_fine_tuning_message
    )


async def process_task_with_semaphore(semaphore, *args):
    async with semaphore:
        return await task_wrapper(*args)


async def process_tasks(raw_fine_tuning_messages, arguments):
    tasks = []
    max_concurrent_tasks = 50
    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Limit concurrent tasks

    # Loop through raw fine-tuning messages and create tasks
    for raw_fine_tuning_message in raw_fine_tuning_messages:
        # Use semaphore to control concurrency
        tasks.append(
            process_task_with_semaphore(
                semaphore,
                arguments["assistant_id"],
                arguments.get("retrain", False),
                raw_fine_tuning_message,
            )
        )

    # Run all tasks concurrently, but respecting the semaphore limit
    results = await asyncio.gather(*tasks)

    return results


@async_task_decorator()
def async_insert_update_fine_tuning_messages(
    logger: logging.Logger, task_uuid: str, arguments: Dict[str, Any]
) -> bool:
    try:
        insert_update_async_task(
            "async_insert_update_fine_tuning_messages", task_uuid, status="in_progress"
        )

        if (
            arguments.get("trained_message_uuids")
            or arguments.get("weightup_message_uuids")
            or arguments.get("weightdown_message_uuids")
        ):
            message_uuids = list(
                set(
                    arguments.get("trained_message_uuids", [])
                    + arguments.get("weightup_message_uuids", [])
                    + arguments.get("weightdown_message_uuids", [])
                )
            )

            with FineTuningMessageModel.batch_write() as batch:
                for message_uuid in message_uuids:
                    count = get_fine_tuning_message_count(
                        arguments["assistant_id"], message_uuid
                    )
                    if count == 0:
                        continue

                    fine_tuning_message = get_fine_tuning_message(
                        arguments["assistant_id"], message_uuid
                    )
                    if message_uuid in arguments.get("trained_message_uuids", []):
                        fine_tuning_message.trained = True
                    if message_uuid in arguments.get("weightup_message_uuids", []):
                        fine_tuning_message.weight = 1
                    if message_uuid in arguments.get("weightdown_message_uuids", []):
                        fine_tuning_message.weight = 0
                    batch.save(fine_tuning_message)

            return True

        assistant = assistant = client.beta.assistants.retrieve(
            arguments["assistant_id"]
        )

        # Step 1: Query the oae-threads table to get thread_id and run_id values

        if arguments.get("to_date") is None:
            to_date = pendulum.now("UTC")
        else:
            to_date = pendulum.parse(arguments["to_date"]).in_tz("UTC")

        # Ensure days is set and is not more than 7
        days = arguments.get("days")
        if days is None or days > fine_tuning_data_days_limit:
            days = fine_tuning_data_days_limit

        # Calculate from_date based on to_date and days
        from_date = to_date.subtract(days=days)

        threads = ThreadModel.query(
            arguments["assistant_id"],
            None,
            filter_condition=(ThreadModel.updated_at.between(from_date, to_date)),
        )

        raw_fine_tuning_messages = []

        # Step 2: Query the oae-messages and oae-tool_calls tables for each thread_id and run_id, then construct the conversation
        for thread in threads:
            run_ids = [run["run_id"] for run in thread.runs]

            logger.info(
                f"Querying the oae-messages table for thread_id: {thread.thread_id}..."
            )

            _raw_fine_tuning_messages = [
                {
                    "assistant_id": arguments["assistant_id"],
                    "message_uuid": str(uuid.uuid1().int >> 64),
                    "thread_id": thread.thread_id,
                    "timestamp": int(time.mktime(thread.created_at.timetuple())) - 100,
                    "role": "system",
                    "content": assistant.instructions,
                }
            ]

            # Query messages table
            messages = MessageModel.query(thread.thread_id, None)
            for message in messages:
                raw_fine_tuning_message = {
                    "assistant_id": arguments["assistant_id"],
                    "message_uuid": str(uuid.uuid1().int >> 64),
                    "thread_id": thread.thread_id,
                    "timestamp": int(time.mktime(message.created_at.timetuple())),
                    "role": message.role,
                    "content": message.message,
                }
                if message.role == "assistant":
                    raw_fine_tuning_message["weight"] = 1

                _raw_fine_tuning_messages.append(raw_fine_tuning_message)

            for run_id in run_ids:
                logger.info(
                    f"Querying the oae-tool_calls table for run_id: {run_id}..."
                )

                # Query tool_calls table
                # the_tool_calls = ToolCallModel.query(run_id, None)
                the_tool_calls = [
                    the_tool_call for the_tool_call in ToolCallModel.query(run_id, None)
                ]

                if len(the_tool_calls) == 0:
                    continue

                tool_calls = []
                # Find the earliest created_at
                earliest_the_tool_call = min(
                    the_tool_calls,
                    key=lambda x: x.created_at,
                )
                # Process tool call
                for tool_call in the_tool_calls:
                    # Handling function calls and their responses
                    tool_call_entry = {
                        "id": tool_call.tool_call_id,
                        "type": tool_call.tool_type,
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    tool_calls.append(tool_call_entry)

                    function_response = {
                        "assistant_id": arguments["assistant_id"],
                        "message_uuid": str(uuid.uuid1().int >> 64),
                        "thread_id": thread.thread_id,
                        "timestamp": int(time.mktime(tool_call.created_at.timetuple())),
                        "role": "tool",
                        "tool_call_id": tool_call.tool_call_id,
                        "content": tool_call.content,
                    }

                    _raw_fine_tuning_messages.append(function_response)

                # Add aggregated tool calls as a single assistant entry
                if tool_calls:
                    tool_call_message = {
                        "assistant_id": arguments["assistant_id"],
                        "message_uuid": str(uuid.uuid1().int >> 64),
                        "thread_id": thread.thread_id,
                        "timestamp": int(
                            time.mktime(earliest_the_tool_call.created_at.timetuple())
                        ),
                        "role": "assistant",
                        "tool_calls": tool_calls,
                    }
                    _raw_fine_tuning_messages.append(tool_call_message)

            # Sort _raw_fine_tuning_messages by timestamp
            _sorted_raw_fine_tuning_messages = sorted(
                _raw_fine_tuning_messages, key=lambda x: x["timestamp"]
            )

            # Remove the last message if it is not from the assistant with content
            while True:
                if _sorted_raw_fine_tuning_messages[-1]["role"] == "system" or (
                    _sorted_raw_fine_tuning_messages[-1]["role"] == "assistant"
                    and _sorted_raw_fine_tuning_messages[-1].get("content")
                ):
                    break
                _sorted_raw_fine_tuning_messages.pop()

            # Skip the conversation if it only contains the system message
            if len(_sorted_raw_fine_tuning_messages) == 1:
                logger.info(
                    f"Skipping _raw_fine_tuning_messages for thread_id: {thread.thread_id} as it only contains the system message."
                )
                continue

            raw_fine_tuning_messages.extend(_sorted_raw_fine_tuning_messages)

        # Run the tasks using asyncio and control concurrency
        asyncio.run(process_tasks(raw_fine_tuning_messages, arguments))

        return True
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


def async_insert_update_fine_tuning_messages_handler(
    logger: logging.Logger, **kwargs: Dict[str, Any]
) -> bool:
    try:
        task_uuid = kwargs["task_uuid"]
        arguments = kwargs["arguments"]
        thread = threading.Thread(
            target=async_insert_update_fine_tuning_messages,
            args=(logger, task_uuid, arguments),
        )
        thread.start()

        ## Check if the thread is alive every 0.1 seconds
        while thread.is_alive():
            time.sleep(0.1)

        return True
    except Exception as e:
        log = traceback.format_exc()
        logger.error(log)
        raise e


def insert_update_fine_tuning_messages_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> bool:
    try:
        task_uuid = str(uuid.uuid1().int >> 64)
        params = {
            "task_uuid": task_uuid,
            "arguments": kwargs,
        }

        invoke_funct_on_aws_lambda(
            info.context["logger"],
            info.context["endpoint_id"],
            "async_insert_update_fine_tuning_messages",
            params=params,
            setting=info.context["setting"],
        )

        return AsyncTaskType(
            function_name="async_insert_update_fine_tuning_messages",
            task_uuid=task_uuid,
        )
    except Exception as e:
        log = traceback.format_exc()
        info.context.get("logger").error(log)
        raise e


@insert_update_decorator(
    keys={
        "hash_key": "assistant_id",
        "range_key": "message_uuid",
    },
    model_funct=get_fine_tuning_message,
    count_funct=get_fine_tuning_message_count,
    type_funct=get_fine_tuning_message_type,
)
def insert_update_fine_tuning_message_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> None:
    assistant_id = kwargs["assistant_id"]
    message_uuid = kwargs["message_uuid"]
    cols = {
        "thread_id": kwargs["thread_id"],
        "timestamp": kwargs["timestamp"],
        "role": kwargs["role"],
    }
    if kwargs.get("tool_calls") is not None:
        cols["tool_calls"] = kwargs["tool_calls"]
    if kwargs.get("tool_call_id") is not None:
        cols["tool_call_id"] = kwargs["tool_call_id"]
    if kwargs.get("content") is not None:
        cols["content"] = kwargs["content"]
    if kwargs.get("weight") is not None:
        cols["weight"] = kwargs["weight"]
    if kwargs.get("trained") is not None:
        cols["trained"] = kwargs["trained"]
    if kwargs.get("entity") is None:
        FineTuningMessageModel(
            assistant_id,
            message_uuid,
            **cols,
        ).save()
        return

    fine_tuning_message = kwargs.get("entity")
    actions = []
    if kwargs.get("tool_calls") is not None:
        actions.append(FineTuningMessageModel.tool_calls.set(kwargs["tool_calls"]))
    if kwargs.get("tool_call_id") is not None:
        actions.append(FineTuningMessageModel.tool_call_id.set(kwargs["tool_call_id"]))
    if kwargs.get("content") is not None:
        actions.append(FineTuningMessageModel.content.set(kwargs["content"]))
    if kwargs.get("weight") is not None:
        actions.append(FineTuningMessageModel.weight.set(kwargs["weight"]))
    if kwargs.get("trained") is not None:
        actions.append(FineTuningMessageModel.trained.set(kwargs["trained"]))
    fine_tuning_message.update(actions=actions)


@delete_decorator(
    keys={
        "hash_key": "assistant_id",
        "range_key": "message_uuid",
    },
    model_funct=get_fine_tuning_message,
)
def delete_fine_tuning_message_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> bool:
    kwargs.get("entity").delete()
    return True


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
)
def get_async_task(function_name: str, task_uuid: str) -> AsyncTaskModel:
    return AsyncTaskModel.get(function_name, task_uuid)


def get_async_task_count(function_name: str, task_uuid: str) -> int:
    return AsyncTaskModel.count(function_name, AsyncTaskModel.task_uuid == task_uuid)


def get_async_task_type(info: ResolveInfo, async_task: AsyncTaskModel) -> AsyncTaskType:
    return AsyncTaskType(**Utility.json_loads(Utility.json_dumps(async_task)))


def resolve_async_task_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> AsyncTaskModel:
    return get_async_task_type(
        info,
        get_async_task(kwargs.get("function_name"), kwargs.get("task_uuid")),
    )


@monitor_decorator
@resolve_list_decorator(
    attributes_to_get=["function_name", "task_uuid"],
    list_type_class=AsyncTaskListType,
    type_funct=get_async_task_type,
)
def resolve_async_task_list_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> Any:
    function_name = kwargs.get("function_name")
    statuses = kwargs.get("statuses")

    args = []
    inquiry_funct = AsyncTaskModel.scan
    count_funct = AsyncTaskModel.count
    if function_name:
        args = [function_name, None]
        inquiry_funct = AsyncTaskModel.query

    the_filters = None
    if statuses:
        the_filters = AsyncTaskModel.status.is_in(*statuses)
    if the_filters is not None:
        args.append(the_filters)

    return inquiry_funct, count_funct, args


@insert_update_decorator(
    keys={
        "hash_key": "function_name",
        "range_key": "task_uuid",
    },
    range_key_required=True,
    model_funct=get_async_task,
    count_funct=get_async_task_count,
    type_funct=get_async_task_type,
)
def insert_update_async_task_handler(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> None:
    function_name = kwargs["function_name"]
    task_uuid = kwargs["task_uuid"]
    if kwargs.get("entity") is None:
        AsyncTaskModel(
            function_name,
            task_uuid,
            **{
                "arguments": kwargs["arguments"],
                "created_at": pendulum.now("UTC"),
                "updated_at": pendulum.now("UTC"),
            },
        ).save()
        return

    async_task = kwargs.get("entity")
    actions = [
        AsyncTaskModel.updated_at.set(pendulum.now("UTC")),
    ]
    if kwargs.get("status") is not None:
        actions.append(AsyncTaskModel.status.set(kwargs["status"]))
    if kwargs.get("result") is not None:
        actions.append(AsyncTaskModel.result.set(kwargs["result"]))
    if kwargs.get("log") is not None:
        actions.append(AsyncTaskModel.log.set(kwargs["log"]))
    async_task.update(actions=actions)


@delete_decorator(
    keys={
        "hash_key": "function_name",
        "range_key": "task_uuid",
    },
    model_funct=get_async_task,
)
def delete_async_task_handler(info: ResolveInfo, **kwargs: Dict[str, Any]) -> bool:
    kwargs.get("entity").delete()
    return True
