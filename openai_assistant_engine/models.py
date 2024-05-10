#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from pynamodb.attributes import (
    ListAttribute,
    MapAttribute,
    NumberAttribute,
    UnicodeAttribute,
    UTCDateTimeAttribute,
    BooleanAttribute,
)
from pynamodb.indexes import GlobalSecondaryIndex, LocalSecondaryIndex, AllProjection
from silvaengine_dynamodb_base import BaseModel


class AssistantModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-assistants"

    type = UnicodeAttribute(hash_key=True)
    assistant_id = UnicodeAttribute(range_key=True)
    assistant_name = UnicodeAttribute()
    functoins = ListAttribute(default=[])
    updated_by = UnicodeAttribute()
    created_at = UTCDateTimeAttribute()
    updated_at = UTCDateTimeAttribute()


class ThreadModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-threads"

    assistant_id = UnicodeAttribute(hash_key=True)
    thread_id = UnicodeAttribute(range_key=True)
    type = UnicodeAttribute()
    status = UnicodeAttribute()
    log = UnicodeAttribute()
    updated_by = UnicodeAttribute()
    created_at = UTCDateTimeAttribute()
    updated_at = UTCDateTimeAttribute()


class MessageModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-messages"

    thread_id = UnicodeAttribute(hash_key=True)
    message_id = UnicodeAttribute(range_key=True)
    role = UnicodeAttribute()
    message = UnicodeAttribute()
    updated_by = UnicodeAttribute()
    created_at = UTCDateTimeAttribute()
    updated_at = UTCDateTimeAttribute()
