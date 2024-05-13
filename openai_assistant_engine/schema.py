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
)
from .types import AskOpenAIType, LastMessageType, CurrentRunType


def type_class():
    return [AskOpenAIType, LastMessageType, CurrentRunType]


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

    def resolve_ping(self, info):
        return f"Hello at {time.strftime('%X')}!!"

    def resolve_ask_open_ai(self, info, **kwargs):
        return resolve_ask_open_ai(info, **kwargs)

    def resolve_last_message(self, info, **kwargs):
        return resolve_last_message(info, **kwargs)

    def resolve_current_run(self, info, **kwargs):
        return resolve_current_run(info, **kwargs)


class Mutations(ObjectType):
    pass
