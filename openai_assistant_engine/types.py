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


class AskOpenAIType(ObjectType):
    assistant_id = String()
    thread_id = String()
    question = String()


class LastMessageType(ObjectType):
    thread_id = String()
    role = String()
    message = String()
