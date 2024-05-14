#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import traceback
from graphene import (
    String,
    Field,
    Mutation,
    Boolean,
    String,
    List,
    DateTime,
    Int,
    Decimal,
)
from silvaengine_utility import JSON
from .types import AssistantType, ThreadType, MessageType
from .handlers import (
    insert_update_assistant_handler,
    delete_assistant_handler,
    insert_update_thread_handler,
    delete_thread_handler,
    insert_update_message_handler,
    delete_message_handler,
)


class InsertUpdateAssistant(Mutation):
    assistant = Field(AssistantType)

    class Arguments:
        assistant_type = String(required=True)
        assistant_id = String(required=True)
        assistant_name = String(required=True)
        functions = List(JSON, required=True)
        updated_by = String(required=True)

    @staticmethod
    def mutate(root, info, **kwargs):
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
    def mutate(root, info, **kwargs):
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
        run_id = String()
        updated_by = String(required=True)

    @staticmethod
    def mutate(root, info, **kwargs):
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
    def mutate(root, info, **kwargs):
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
    def mutate(root, info, **kwargs):
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
    def mutate(root, info, **kwargs):
        try:
            ok = delete_message_handler(info, **kwargs)
        except Exception as e:
            log = traceback.format_exc()
            info.context.get("logger").error(log)
            raise e

        return DeleteMessage(ok=ok)
