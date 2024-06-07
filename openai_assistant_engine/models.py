#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from pynamodb.attributes import (
    BooleanAttribute,
    ListAttribute,
    MapAttribute,
    UnicodeAttribute,
    UTCDateTimeAttribute,
)
from silvaengine_dynamodb_base import BaseModel


class AssistantModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-assistants"

    assistant_type = UnicodeAttribute(hash_key=True)
    assistant_id = UnicodeAttribute(range_key=True)
    assistant_name = UnicodeAttribute()
    functions = ListAttribute()
    updated_by = UnicodeAttribute()
    created_at = UTCDateTimeAttribute()
    updated_at = UTCDateTimeAttribute()


class ThreadModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-threads"

    assistant_id = UnicodeAttribute(hash_key=True)
    thread_id = UnicodeAttribute(range_key=True)
    assistant_type = UnicodeAttribute()
    is_voice = BooleanAttribute()
    runs = ListAttribute(of=MapAttribute)
    updated_by = UnicodeAttribute()
    created_at = UTCDateTimeAttribute()
    updated_at = UTCDateTimeAttribute()


class MessageModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-messages"

    thread_id = UnicodeAttribute(hash_key=True)
    message_id = UnicodeAttribute(range_key=True)
    run_id = UnicodeAttribute(null=True)
    role = UnicodeAttribute()
    message = UnicodeAttribute()
    created_at = UTCDateTimeAttribute()
