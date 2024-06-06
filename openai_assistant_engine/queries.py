#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from typing import Any, Dict, List

from graphene import ResolveInfo

from .handlers import (
    resolve_ask_open_ai_handler,
    resolve_assistant_handler,
    resolve_assistant_list_handler,
    resolve_current_run_handler,
    resolve_last_message_handler,
    resolve_live_messages_handler,
    resolve_message_handler,
    resolve_message_list_handler,
    resolve_thread_handler,
    resolve_thread_list_handler,
)
from .types import (
    AskOpenAIType,
    AssistantListType,
    AssistantType,
    CurrentRunType,
    LiveMessageType,
    MessageListType,
    MessageType,
    ThreadListType,
    ThreadType,
)


def resolve_ask_open_ai(info: ResolveInfo, **kwargs: Dict[str, Any]) -> AskOpenAIType:
    return resolve_ask_open_ai_handler(info, **kwargs)


def resolve_live_messages(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> List[LiveMessageType]:
    return resolve_live_messages_handler(info, **kwargs)


def resolve_last_message(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> LiveMessageType:
    return resolve_last_message_handler(info, **kwargs)


def resolve_last_message(
    info: ResolveInfo, **kwargs: Dict[str, Any]
) -> LiveMessageType:
    return resolve_last_message_handler(info, **kwargs)


def resolve_current_run(info: ResolveInfo, **kwargs: Dict[str, Any]) -> CurrentRunType:
    return resolve_current_run_handler(info, **kwargs)


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
