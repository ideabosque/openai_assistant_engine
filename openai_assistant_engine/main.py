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
    return [
        {
            "service": "OpenAI Assistant Engine",
            "class": "OpenaiAssistantEngine",
            "functions": {
                "open_assistant_graphql": {
                    "is_static": False,
                    "label": "OpenAI Assistant GraphQL",
                    "query": [
                        {
                            "action": "askOpenAi",
                            "label": "Ask Open AI",
                        },
                        {
                            "action": "liveMessages",
                            "label": "View Live Messages",
                        },
                        {
                            "action": "lastMessage",
                            "label": "View Last Message",
                        },
                        {
                            "action": "currentRun",
                            "label": "View Current Run",
                        },
                        {
                            "action": "assistant",
                            "label": "View Assistant",
                        },
                        {
                            "action": "assistantList",
                            "label": "View Assistant List",
                        },
                        {
                            "action": "thread",
                            "label": "View Thread",
                        },
                        {
                            "action": "threadList",
                            "label": "View Thread List",
                        },
                        {
                            "action": "message",
                            "label": "View Message",
                        },
                        {
                            "action": "messageList",
                            "label": "View Message List",
                        },
                    ],
                    "mutation": [
                        {
                            "action": "InsertUpdateAssistant",
                            "label": "Create Update Assistant",
                        },
                        {
                            "action": "deleteAssistant",
                            "label": "Delete Assistant",
                        },
                        {
                            "action": "insertUpdateThread",
                            "label": "Create Update Thread",
                        },
                        {
                            "action": "deleteThread",
                            "label": "Delete Thread",
                        },
                        {
                            "action": "insertUpdateMessage",
                            "label": "Create Update Message",
                        },
                        {
                            "action": "deleteMessage",
                            "label": "Delete Message",
                        },
                    ],
                    "type": "RequestResponse",
                    "support_methods": ["POST"],
                    "is_auth_required": False,
                    "is_graphql": True,
                    "settings": "open_assistant_engine",
                    "disabled_in_resources": True,  # Ignore adding to resource list.
                },
            },
        }
    ]


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
