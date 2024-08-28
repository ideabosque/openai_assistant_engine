#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from graphene import Boolean, DateTime, Float, Int, List, ObjectType, String
from silvaengine_dynamodb_base import ListObjectType
from silvaengine_utility import JSON


class AskOpenAIType(ObjectType):
    assistant_id = String()
    thread_id = String()
    user_query = String()
    current_run_id = String()


class LiveMessageType(ObjectType):
    thread_id = String()
    run_id = String()
    message_id = String()
    role = String()
    message = String()
    created_at = DateTime()


class CurrentRunType(ObjectType):
    thread_id = String()
    run_id = String()
    status = String()
    usage = JSON()


class OpenAIFileType(ObjectType):
    id = String()
    object = String()
    bytes = Int()
    created_at = DateTime()
    filename = String()
    purpose = String()
    encoded_content = String()


class AssistantType(ObjectType):
    assistant_type = String()
    assistant_id = String()
    assistant_name = String()
    description = String()
    model = String()
    instructions = String()
    metadata = JSON()
    temperature = Float()
    top_p = Float()
    response_format = String()
    tools = List(JSON)
    tool_resources = JSON()
    configuration = JSON()
    functions = List(JSON)
    updated_by = String()
    created_at = DateTime()
    updated_at = DateTime()


class ThreadType(ObjectType):
    assistant = JSON()
    thread_id = String()
    runs = List(JSON)
    updated_by = String()
    created_at = DateTime()
    updated_at = DateTime()


class MessageType(ObjectType):
    thread_id = String()
    run_id = String()
    message_id = String()
    role = String()
    message = String()
    created_at = DateTime()


class ToolCallType(ObjectType):
    run_id = String()
    tool_call_id = String()
    tool_type = String()
    name = String()
    arguments = JSON()
    content = String()
    created_at = DateTime()


class FineTuningMessageType(ObjectType):
    model = String()
    timestamp = String()
    assistant = JSON()
    role = String()
    tool_calls = List(JSON)
    tool_call_id = String()
    content = String()
    weight = Float()
    trained = Boolean()


class AssistantListType(ListObjectType):
    assistant_list = List(AssistantType)


class ThreadListType(ListObjectType):
    thread_list = List(ThreadType)


class MessageListType(ListObjectType):
    message_list = List(MessageType)


class ToolCallListType(ListObjectType):
    tool_call_list = List(ToolCallType)


class FineTuningMessageListType(ListObjectType):
    fine_tuning_message_list = List(FineTuningMessageType)
