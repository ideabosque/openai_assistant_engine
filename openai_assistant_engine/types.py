#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from graphene import Boolean, DateTime, List, ObjectType, String

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


class AssistantType(ObjectType):
    assistant_type = String()
    assistant_id = String()
    assistant_name = String()
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


class AssistantListType(ListObjectType):
    assistant_list = List(AssistantType)


class ThreadListType(ListObjectType):
    thread_list = List(ThreadType)


class MessageListType(ListObjectType):
    message_list = List(MessageType)
