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
from typing import List, Optional
from datetime import datetime


class AssistantModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-assistants"

    assistant_type: str = UnicodeAttribute(hash_key=True)
    assistant_id: str = UnicodeAttribute(range_key=True)
    assistant_name: str = UnicodeAttribute()
    functions: List[str] = ListAttribute()
    updated_by: str = UnicodeAttribute()
    created_at: datetime = UTCDateTimeAttribute()
    updated_at: datetime = UTCDateTimeAttribute()


class ThreadModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-threads"

    assistant_id: str = UnicodeAttribute(hash_key=True)
    thread_id: str = UnicodeAttribute(range_key=True)
    assistant_type: str = UnicodeAttribute()
    run_ids: List[str] = ListAttribute()
    updated_by: str = UnicodeAttribute()
    created_at: datetime = UTCDateTimeAttribute()
    updated_at: datetime = UTCDateTimeAttribute()


class MessageModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-messages"

    thread_id: str = UnicodeAttribute(hash_key=True)
    message_id: str = UnicodeAttribute(range_key=True)
    run_id: Optional[str] = UnicodeAttribute(null=True)
    role: str = UnicodeAttribute()
    message: str = UnicodeAttribute()
    created_at: datetime = UTCDateTimeAttribute()
