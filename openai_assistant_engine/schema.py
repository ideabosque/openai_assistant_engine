#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import time
from graphene import ObjectType, String, List, Field, Int, DateTime, Boolean
from silvaengine_utility import JSON
from .queries import (
    resolve_ask_open_ai,
    resolve_last_message,
    resolve_current_run,
    resolve_assistant,
    resolve_assistant_list,
    resolve_thread,
    resolve_thread_list,
    resolve_message,
    resolve_message_list,
)
from .mutations import (
    InsertUpdateAssistant,
    DeleteAssistant,
    InsertUpdateThread,
    DeleteThread,
    InsertUpdateMessage,
    DeleteMessage,
)
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


def type_class():
    return [
        AskOpenAIType,
        LastMessageType,
        CurrentRunType,
        AssistantType,
        AssistantListType,
        ThreadType,
        ThreadListType,
        MessageType,
        MessageListType,
    ]


class Query(ObjectType):
    ping = String()
    ask_open_ai = Field(
        AskOpenAIType,
        required=True,
        assistant_type=String(required=True),
        assistant_id=String(required=True),
        question=String(required=True),
        updated_by=String(required=True),
        thread_id=String(),
    )

    last_message = Field(
        LastMessageType,
        required=True,
        thread_id=String(required=True),
        role=String(required=True),
    )

    current_run = Field(
        CurrentRunType,
        required=True,
        assistant_id=String(required=True),
        thread_id=String(required=True),
        run_id=String(required=True),
        updated_by=String(required=True),
    )

    assistant = Field(
        AssistantType,
        required=True,
        assistant_type=String(required=True),
        assistant_id=String(required=True),
    )

    assistant_list = Field(
        AssistantListType,
        page_number=Int(),
        limit=Int(),
        assistant_type=String(),
        assistant_name=String(),
    )

    thread = Field(
        ThreadType,
        required=True,
        assistant_id=String(),
        thread_id=String(),
    )

    thread_list = Field(
        ThreadListType,
        page_number=Int(),
        limit=Int(),
        assistant_id=String(),
        assistant_type=String(),
        run_id=String(),
    )

    message = Field(
        MessageType,
        required=True,
        thread_id=String(required=True),
        message_id=String(required=True),
    )

    message_list = Field(
        MessageListType,
        page_number=Int(),
        limit=Int(),
        assistant_id=String(),
        assistant_type=String(),
        run_id=String(),
    )

    def resolve_ping(self, info):
        return f"Hello at {time.strftime('%X')}!!"

    def resolve_ask_open_ai(self, info, **kwargs):
        return resolve_ask_open_ai(info, **kwargs)

    def resolve_last_message(self, info, **kwargs):
        return resolve_last_message(info, **kwargs)

    def resolve_current_run(self, info, **kwargs):
        return resolve_current_run(info, **kwargs)

    def resolve_assistant(self, info, **kwargs):
        return resolve_assistant(info, **kwargs)

    def resolve_assistant_list(self, info, **kwargs):
        return resolve_assistant_list(info, **kwargs)

    def resolve_thread(self, info, **kwargs):
        return resolve_thread(info, **kwargs)

    def resolve_thread_list(self, info, **kwargs):
        return resolve_thread_list(info, **kwargs)

    def resolve_message(self, info, **kwargs):
        return resolve_message(info, **kwargs)

    def resolve_message_list(self, info, **kwargs):
        return resolve_message_list(info, **kwargs)


class Mutations(ObjectType):
    insert_update_assistant = InsertUpdateAssistant.Field()
    delete_assistant = DeleteAssistant.Field()
    insert_update_thread = InsertUpdateThread.Field()
    delete_thread = DeleteThread.Field()
    insert_update_message = InsertUpdateMessage.Field()
    delete_message = DeleteMessage.Field()
