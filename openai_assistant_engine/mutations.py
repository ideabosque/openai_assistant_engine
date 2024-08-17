#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import traceback
from typing import Any, Dict

from graphene import Boolean, DateTime, Field, Float, List, Mutation, String
from silvaengine_utility import JSON

from .handlers import (
    delete_assistant_handler,
    delete_file_handler,
    delete_message_handler,
    delete_thread_handler,
    insert_file_handler,
    insert_update_assistant_handler,
    insert_update_message_handler,
    insert_update_thread_handler,
)
from .types import AssistantType, MessageType, OpenAIFileType, ThreadType


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
