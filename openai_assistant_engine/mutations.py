#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import traceback
from typing import Any, Dict

from graphene import Boolean, DateTime, Field, Float, Int, List, Mutation, String
from silvaengine_utility import JSON

from .handlers import (
    delete_assistant_handler,
    delete_async_task_handler,
    delete_file_handler,
    delete_fine_tuning_message_handler,
    delete_message_handler,
    delete_thread_handler,
    delete_tool_call_handler,
    insert_file_handler,
    insert_update_assistant_handler,
    insert_update_async_task_handler,
    insert_update_fine_tuning_message_handler,
    insert_update_fine_tuning_messages_handler,
    insert_update_message_handler,
    insert_update_thread_handler,
    insert_update_tool_call_handler,
)
from .types import (
    AssistantType,
    AsyncTaskType,
    FineTuningMessageType,
    MessageType,
    OpenAIFileType,
    ThreadType,
    ToolCallType,
)


class InsertFile(Mutation):
    file = Field(OpenAIFileType)

    class Arguments:
        filename = String(required=True)
        encoded_content = String(required=True)
        purpose = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "InsertFile":
        try:
            file = insert_file_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertFile(file=file)


class DeleteFile(Mutation):
    ok = Boolean()

    class Arguments:
        file_id = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "DeleteFile":
        try:
            ok = delete_file_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteFile(ok=ok)


class InsertUpdateAssistant(Mutation):
    assistant = Field(AssistantType)

    class Arguments:
        assistant_type = String(required=True)
        assistant_id = String(required=False)
        assistant_name = String(required=True)
        description = String(required=False)
        model = String(required=True)
        instructions = String(required=True)
        tools = List(JSON, required=False)
        tool_resources = JSON(required=False)
        metadata = JSON(required=False)
        temperature = Float(required=False)
        top_p = Float(required=False)
        response_format = String(required=False)
        configuration = JSON(required=True)
        functions = List(JSON, required=True)
        updated_by = String(required=True)

    @staticmethod
    def mutate(
        root: Any, info: Any, **kwargs: Dict[str, Any]
    ) -> "InsertUpdateAssistant":
        try:
            assistant = insert_update_assistant_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertUpdateAssistant(assistant=assistant)


class DeleteAssistant(Mutation):
    ok = Boolean()

    class Arguments:
        assistant_type = String(required=True)
        assistant_id = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "DeleteAssistant":
        try:
            ok = delete_assistant_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteAssistant(ok=ok)


class InsertUpdateThread(Mutation):
    thread = Field(ThreadType)

    class Arguments:
        assistant_id = String(required=True)
        thread_id = String(required=True)
        assistant_type = String(required=True)
        run = JSON()
        updated_by = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "InsertUpdateThread":
        try:
            thread = insert_update_thread_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertUpdateThread(thread=thread)


class DeleteThread(Mutation):
    ok = Boolean()

    class Arguments:
        assistant_id = String(required=True)
        thread_id = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "DeleteThread":
        try:
            ok = delete_thread_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteThread(ok=ok)


class InsertUpdateMessage(Mutation):
    message = Field(MessageType)

    class Arguments:
        thread_id = String(required=True)
        message_id = String(required=True)
        run_id = String(required=True)
        role = String(required=True)
        message = String(required=True)
        created_at = DateTime(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "InsertUpdateMessage":
        try:
            message = insert_update_message_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertUpdateMessage(message=message)


class DeleteMessage(Mutation):
    ok = Boolean()

    class Arguments:
        thread_id = String(required=True)
        message_id = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "DeleteMessage":
        try:
            ok = delete_message_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteMessage(ok=ok)


class InsertUpdateToolCall(Mutation):
    tool_call = Field(ToolCallType)

    class Arguments:
        run_id = String(required=True)
        tool_call_id = String(required=True)
        tool_type = String(required=True)
        name = String(required=True)
        arguments = JSON(required=True)
        content = String(required=False)
        created_at = DateTime(required=True)

    @staticmethod
    def mutate(
        root: Any, info: Any, **kwargs: Dict[str, Any]
    ) -> "InsertUpdateToolCall":
        try:
            tool_call = insert_update_tool_call_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertUpdateToolCall(tool_call=tool_call)


class DeleteToolCall(Mutation):
    ok = Boolean()

    class Arguments:
        run_id = String(required=True)
        tool_call_id = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "DeleteToolCall":
        try:
            ok = delete_tool_call_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteToolCall(ok=ok)


class InsertUpdateFineTuningMessages(Mutation):
    async_task = Field(AsyncTaskType)

    class Arguments:
        assistant_type = String(required=False)
        assistant_id = String(required=False)
        to_date = DateTime(required=False)
        days = Int(required=False)
        retrain = Boolean(required=False)
        trained_message_uuids = List(String, required=False)
        weightup_message_uuids = List(String, required=False)
        weightdown_message_uuids = List(String, required=False)

    @staticmethod
    def mutate(
        root: Any, info: Any, **kwargs: Dict[str, Any]
    ) -> "InsertUpdateFineTuningMessages":
        try:
            async_task = insert_update_fine_tuning_messages_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertUpdateFineTuningMessages(async_task=async_task)


class InsertUpdateFineTuningMessage(Mutation):
    fine_tuning_message = Field(FineTuningMessageType)

    class Arguments:
        assistant_id = String(required=True)
        message_uuid = String(required=False)
        thread_id = String(required=True)
        timestamp = Int(required=True)
        role = String(required=True)
        tool_calls = List(JSON, required=False)
        tool_call_id = String(required=False)
        content = String(required=False)
        weight = Float(required=False)
        trained = Boolean(required=False)

    @staticmethod
    def mutate(
        root: Any, info: Any, **kwargs: Dict[str, Any]
    ) -> "InsertUpdateFineTuningMessage":
        try:
            fine_tuning_message = insert_update_fine_tuning_message_handler(
                info, **kwargs
            )
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertUpdateFineTuningMessage(fine_tuning_message=fine_tuning_message)


class DeleteFineTuningMessage(Mutation):
    ok = Boolean()

    class Arguments:
        assistant_id = String(required=True)
        message_uuid = String(required=True)

    @staticmethod
    def mutate(
        root: Any, info: Any, **kwargs: Dict[str, Any]
    ) -> "DeleteFineTuningMessage":
        try:
            ok = delete_fine_tuning_message_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteFineTuningMessage(ok=ok)


class InsertUpdateAsyncTask(Mutation):
    async_task = Field(AsyncTaskType)

    class Arguments:
        function_name = String(required=True)
        task_uuid = String(required=True)
        arguments = JSON(required=False)
        status = String(required=False)
        result = String(required=False)
        log = String(required=False)

    @staticmethod
    def mutate(
        root: Any, info: Any, **kwargs: Dict[str, Any]
    ) -> "InsertUpdateAsyncTask":
        try:
            async_task = insert_update_async_task_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return InsertUpdateAsyncTask(async_task=async_task)


class DeleteAsyncTask(Mutation):
    ok = Boolean()

    class Arguments:
        function_name = String(required=True)
        task_uuid = String(required=True)

    @staticmethod
    def mutate(root: Any, info: Any, **kwargs: Dict[str, Any]) -> "DeleteAsyncTask":
        try:
            ok = delete_async_task_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteAsyncTask(ok=ok)
