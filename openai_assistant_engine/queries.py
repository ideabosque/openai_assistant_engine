#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from .handlers import (
    ask_open_ai_handler,
    last_message_handler,
    current_run_handler,
    resolve_assistant_handler,
    resolve_assistant_list_handler,
    resolve_thread_handler,
    resolve_thread_list_handler,
    resolve_message_handler,
    resolve_message_list_handler,
)
from graphene import ResolveInfo
from typing import Dict, Any
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


def resolve_ask_open_ai(info: ResolveInfo, **kwargs: Dict[str, Any]) -> AskOpenAIType:
    return ask_open_ai_handler(info, **kwargs)


def resolve_last_message(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> LastMessageType:
    return last_message_handler(info, **kwargs)


def resolve_current_run(info: ResolveInfo, **kwargs: Dict[str, Any]) -> CurrentRunType:
    return current_run_handler(info, **kwargs)


def resolve_assistant(info: ResolveInfo, **kwargs: Dict[str, Any]) -> AssistantType:
    return resolve_assistant_handler(info, **kwargs)


def resolve_assistant_list(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> AssistantListType:
    return resolve_assistant_list_handler(info, **kwargs)


def resolve_thread(info: ResolveInfo, **kwargs: Dict[str, Any]) -> ThreadType:
    return resolve_thread_handler(info, **kwargs)


def resolve_thread_list(info: ResolveInfo, **kwargs: Dict[str, Any]) -> ThreadListType:
    return resolve_thread_list_handler(info, **kwargs)


def resolve_message(info: ResolveInfo, **kwargs: Dict[str, Any]) -> MessageType:
    return resolve_message_handler(info, **kwargs)


def resolve_message_list(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> MessageListType:
    return resolve_message_list_handler(info, **kwargs)
