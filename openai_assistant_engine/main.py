#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import logging
from typing import Any, Dict, List

from graphene import Schema
from silvaengine_dynamodb_base import SilvaEngineDynamoDBBase

from .handlers import handlers_init
from .schema import Mutations, Query, type_class


# Hook function applied to deployment
def deploy() -> List:
    return []


class OpenaiAssistantEngine(SilvaEngineDynamoDBBase):
    def __init__(self, logger: logging.Logger, **setting: Dict[str, Any]) -> None:
        handlers_init(logger, **setting)

        self.logger = logger
        self.setting = setting

        SilvaEngineDynamoDBBase.__init__(self, logger, **setting)

    def open_assistant_graphql(self, **params: Dict[str, Any]) -> Any:
        schema = Schema(
            query=Query,
            mutation=Mutations,
            types=type_class(),
        )
        return self.graphql_execute(schema, **params)
