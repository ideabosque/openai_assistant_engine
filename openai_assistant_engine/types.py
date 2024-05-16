#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from graphene import (
    ObjectType,
    Field,
    List,
    String,
    Int,
    Decimal,
    DateTime,
    Boolean,
)
from silvaengine_utility import JSON
from silvaengine_dynamodb_base import ListObjectType
from datetime import datetime


class AskOpenAIType(ObjectType):
    assistant_id: str = String()
    thread_id: str = String()
    question: str = String()
    current_run_id: str = String()


class LastMessageType(ObjectType):
    thread_id: str = String()
    run_id: str = String()
    message_id: str = String()
    role: str = String()
    message: str = String()
    created_at: datetime = DateTime()


class CurrentRunType(ObjectType):
    thread_id: str = String()
    run_id: str = String()
    status: str = String()


class AssistantType(ObjectType):
    assistant_type: str = String()
    assistant_id: str = String()
    assistant_name: str = String()
    functions: list = List(JSON)
    updated_by: str = String()
    created_at: datetime = DateTime()
    updated_at: datetime = DateTime()


class ThreadType(ObjectType):
    assistant_id: str = String()
    thread_id: str = String()
    assistant_type: str = String()
    run_ids: list = List(String)
    updated_by: str = String()
    created_at: datetime = DateTime()
    updated_at: datetime = DateTime()


class MessageType(ObjectType):
    thread_id: str = String()
    run_id: str = String()
    message_id: str = String()
    role: str = String()
    message: str = String()
    created_at: datetime = DateTime()


class AssistantListType(ListObjectType):
    assistant_list: list = List(AssistantType)


class ThreadListType(ListObjectType):
    thread_list: list = List(ThreadType)


class MessageListType(ListObjectType):
    message_list: list = List(MessageType)
