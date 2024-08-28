#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import time
from typing import Any, Dict
from typing import List as Typing_List

from graphene import Boolean, Field, Int, List, ObjectType, ResolveInfo, String
from silvaengine_utility import JSON

from .mutations import (
    DeleteAssistant,
    DeleteFile,
    DeleteFineTuningMessage,
    DeleteMessage,
    DeleteThread,
    DeleteToolCall,
    InsertFile,
    InsertUpdateAssistant,
    InsertUpdateFineTuningMessage,
    InsertUpdateMessage,
    InsertUpdateThread,
    InsertUpdateToolCall,
)
from .queries import (
    resolve_ask_open_ai,
    resolve_assistant,
    resolve_assistant_list,
    resolve_current_run,
    resolve_file,
    resolve_files,
    resolve_fine_tuning_message,
    resolve_fine_tuning_message_list,
    resolve_last_message,
    resolve_live_messages,
    resolve_message,
    resolve_message_list,
    resolve_thread,
    resolve_thread_list,
    resolve_tool_call,
    resolve_tool_call_list,
)
from .types import (
    AskOpenAIType,
    AssistantListType,
    AssistantType,
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


def type_class():
    return [
        AskOpenAIType,
        LiveMessageType,
        OpenAIFileType,
        CurrentRunType,
        AssistantType,
        AssistantListType,
        ThreadType,
        ThreadListType,
        MessageType,
        MessageListType,
        ToolCallType,
        ToolCallListType,
        FineTuningMessageType,
        FineTuningMessageListType,
    ]


class Query(ObjectType):
    ping = String()
    ask_open_ai = Field(
        AskOpenAIType,
        required=True,
        assistant_type=String(required=True),
        assistant_id=String(required=True),
        instructions=String(required=False),
        attachments=List(JSON, required=False),
        tool_resources=JSON(required=False),
        thread_metadata=JSON(required=False),
        message_metadata=JSON(required=False),
        user_query=String(required=True),
        updated_by=String(required=True),
        thread_id=String(),
    )

    live_messages = List(
        LiveMessageType,
        required=True,
        thread_id=String(required=True),
        roles=List(String, required=False),
        order=String(required=False),
    )

    file = Field(
        OpenAIFileType,
        required=True,
        file_id=String(required=True),
    )

    files = List(
        OpenAIFileType,
        purpose=String(),
    )

    last_message = Field(
        LiveMessageType,
        required=True,
        assistant_id=String(required=False),
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
        assistant_types=List(String),
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
        thread_id=String(),
        roles=List(String),
        message=String(),
    )

    tool_call = Field(
        ToolCallType,
        required=True,
        run_id=String(required=True),
        tool_call_id=String(required=True),
    )

    tool_call_list = Field(
        ToolCallListType,
        page_number=Int(),
        limit=Int(),
        run_id=String(),
        tool_types=List(String),
        name=String(),
    )

    fine_tuning_message = Field(
        FineTuningMessageType,
        required=True,
        model=String(required=True),
        timestamp=String(required=True),
    )

    fine_tuning_message_list = Field(
        FineTuningMessageListType,
        page_number=Int(),
        limit=Int(),
        model=String(),
        assistant_id=String(),
        roles=List(String),
        trained=Boolean(),
    )

    def resolve_ping(self, info: ResolveInfo) -> str:
        return f"Hello at {time.strftime('%X')}!!"

    def resolve_ask_open_ai(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> AskOpenAIType:
        return resolve_ask_open_ai(info, **kwargs)

    def resolve_live_messages(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> Typing_List[LiveMessageType]:
        return resolve_live_messages(info, **kwargs)

    def resolve_file(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> OpenAIFileType:
        return resolve_file(info, **kwargs)

    def resolve_files(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> Typing_List[OpenAIFileType]:
        return resolve_files(info, **kwargs)

    def resolve_last_message(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> LiveMessageType:
        return resolve_last_message(info, **kwargs)

    def resolve_current_run(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> CurrentRunType:
        return resolve_current_run(info, **kwargs)

    def resolve_assistant(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> AssistantType:
        return resolve_assistant(info, **kwargs)

    def resolve_assistant_list(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> AssistantListType:
        return resolve_assistant_list(info, **kwargs)

    def resolve_thread(self, info: ResolveInfo, **kwargs: Dict[str, Any]) -> ThreadType:
        return resolve_thread(info, **kwargs)

    def resolve_thread_list(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> ThreadListType:
        return resolve_thread_list(info, **kwargs)

    def resolve_message(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> MessageType:
        return resolve_message(info, **kwargs)

    def resolve_message_list(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> MessageListType:
        return resolve_message_list(info, **kwargs)

    def resolve_tool_call(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> ToolCallType:
        return resolve_tool_call(info, **kwargs)

    def resolve_tool_call_list(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> ToolCallListType:
        return resolve_tool_call_list(info, **kwargs)

    def resolve_fine_tuning_message(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> FineTuningMessageType:
        return resolve_fine_tuning_message(info, **kwargs)

    def resolve_fine_tuning_message_list(
        self, info: ResolveInfo, **kwargs: Dict[str, Any]
    ) -> FineTuningMessageListType:
        return resolve_fine_tuning_message_list(info, **kwargs)


class Mutations(ObjectType):
    insert_file = InsertFile.Field()
    delete_file = DeleteFile.Field()
    insert_update_assistant = InsertUpdateAssistant.Field()
    delete_assistant = DeleteAssistant.Field()
    insert_update_thread = InsertUpdateThread.Field()
    delete_thread = DeleteThread.Field()
    insert_update_message = InsertUpdateMessage.Field()
    delete_message = DeleteMessage.Field()
    insert_update_tool_call = InsertUpdateToolCall.Field()
    delete_tool_call = DeleteToolCall.Field()
    insert_update_fine_tuning_message = InsertUpdateFineTuningMessage.Field()
    delete_fine_tuning_message = DeleteFineTuningMessage.Field()
