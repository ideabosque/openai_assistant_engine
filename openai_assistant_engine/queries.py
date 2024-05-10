#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from .handlers import ask_open_ai_handler, last_message_handler


def resolve_ask_open_ai(info, **kwargs):
    return ask_open_ai_handler(info, **kwargs)


def resolve_last_message(info, **kwargs):
    return last_message_handler(info, **kwargs)
