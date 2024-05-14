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


def resolve_ask_open_ai(info, **kwargs):
    return ask_open_ai_handler(info, **kwargs)


def resolve_last_message(info, **kwargs):
    return last_message_handler(info, **kwargs)


def resolve_current_run(info, **kwargs):
    return current_run_handler(info, **kwargs)


def resolve_assistant(info, **kwargs):
    return resolve_assistant_handler(info, **kwargs)


def resolve_assistant_list(info, **kwargs):
    return resolve_assistant_list_handler(info, **kwargs)


def resolve_thread(info, **kwargs):
    return resolve_thread_handler(info, **kwargs)


def resolve_thread_list(info, **kwargs):
    return resolve_thread_list_handler(info, **kwargs)


def resolve_message(info, **kwargs):
    return resolve_message_handler(info, **kwargs)


def resolve_message_list(info, **kwargs):
    return resolve_message_list_handler(info, **kwargs)
