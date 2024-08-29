#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from pynamodb.attributes import (
    BooleanAttribute,
    ListAttribute,
    MapAttribute,
    NumberAttribute,
    UnicodeAttribute,
    UTCDateTimeAttribute,
)
from pynamodb.indexes import AllProjection, LocalSecondaryIndex
from silvaengine_dynamodb_base import BaseModel


class AssistantModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-assistants"

    assistant_type = UnicodeAttribute(hash_key=True)
    assistant_id = UnicodeAttribute(range_key=True)
    assistant_name = UnicodeAttribute()
    configuration = MapAttribute()
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


class ToolCallModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-tool_calls"

    run_id = UnicodeAttribute(hash_key=True)
    tool_call_id = UnicodeAttribute(range_key=True)
    tool_type = UnicodeAttribute()
    name = UnicodeAttribute()
    arguments = MapAttribute()
    content = UnicodeAttribute(null=True)
    created_at = UTCDateTimeAttribute()


class ThreadIdIndex(LocalSecondaryIndex):
    """
    This class represents a local secondary index
    """

    class Meta:
        billing_mode = "PAY_PER_REQUEST"
        # All attributes are projected
        projection = AllProjection()
        index_name = "thread_id-index"

    assistant_id = UnicodeAttribute(hash_key=True)
    thread_id = UnicodeAttribute(range_key=True)


class TimestampIndex(LocalSecondaryIndex):
    """
    This class represents a local secondary index
    """

    class Meta:
        billing_mode = "PAY_PER_REQUEST"
        # All attributes are projected
        projection = AllProjection()
        index_name = "timestamp-index"

    assistant_id = UnicodeAttribute(hash_key=True)
    timestamp = NumberAttribute(range_key=True)


class FineTuningMessageModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-fine_tuning_messages"

    assistant_id = UnicodeAttribute(hash_key=True)
    message_uuid = UnicodeAttribute(range_key=True)
    thread_id = UnicodeAttribute()
    timestamp = NumberAttribute()
    role = UnicodeAttribute()
    tool_calls = ListAttribute(of=MapAttribute, null=True)
    tool_call_id = UnicodeAttribute(null=True)
    content = UnicodeAttribute(null=True)
    weight = NumberAttribute(null=True)
    trained = BooleanAttribute(default=False)
    thread_id_index = ThreadIdIndex()
    timestamp_index = TimestampIndex()


class AsyncTaskModel(BaseModel):
    class Meta(BaseModel.Meta):
        table_name = "oae-async_tasks"

    function_name = UnicodeAttribute(hash_key=True)
    task_uuid = UnicodeAttribute(range_key=True)
    arguments = MapAttribute()
    status = UnicodeAttribute()
    results = MapAttribute(null=True)
    log = UnicodeAttribute(null=True)
    created_at = UTCDateTimeAttribute()
    updated_at = UTCDateTimeAttribute()
