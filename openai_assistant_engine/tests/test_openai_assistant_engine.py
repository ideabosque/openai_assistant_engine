#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import json
import logging
import os
import sys
import time
import unittest
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "openai_api_key": os.getenv("openai_api_key"),
}

document = Path(
    os.path.join(os.path.dirname(__file__), "openai_assistant_engine.graphql")
).read_text()
sys.path.insert(0, "C:/Users/bibo7/gitrepo/silvaengine/openai_assistant_engine")
sys.path.insert(1, "C:/Users/bibo7/gitrepo/silvaengine/redis_stack_connector")
# sys.path.insert(0, "/var/www/projects/openai_assistant_engine")
# sys.path.insert(1, "/var/www/projects/redis_stack_connector")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

from openai_assistant_engine import OpenaiAssistantEngine


class OpenaiAssistantEngineTest(unittest.TestCase):
    def setUp(self):
        self.openai_assistant_engine = OpenaiAssistantEngine(logger, **setting)
        logger.info("Initiate OpenaiAssistantEngineTest ...")

    def tearDown(self):
        logger.info("Destory OpenaiAssistantEngineTest ...")

    @unittest.skip("demonstrating skipping")
    def test_graphql_ping(self):
        payload = {
            "query": document,
            "variables": {},
            "operation_name": "ping",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    # @unittest.skip("demonstrating skipping")
    def test_conversation_search(self):
        logger.info("Start test_conversation_search ...")
        print("Hello! I am an AI assistant. How can I help you today?")
        assistant_id = "asst_jUzZKojROaz6HACC1uzaqR5x"
        thread_id = None
        while True:
            user_input = input("You: ").strip().lower()

            if user_input == "exit":
                print("Exiting the program.")
                break

            payload = {
                "query": document,
                "variables": {
                    "userQuery": user_input,
                    "assistantId": assistant_id,
                    "assistantType": "conversation",
                    "threadId": thread_id,
                    "updatedBy": "Use XYZ",
                },
                "operation_name": "askOpenAi",
            }
            response = json.loads(
                self.openai_assistant_engine.open_assistant_graphql(**payload)
            )
            # logger.info(response)
            thread_id = response["data"]["askOpenAi"]["threadId"]
            current_run_id = response["data"]["askOpenAi"]["currentRunId"]

            while True:
                payload = {
                    "query": document,
                    "variables": {
                        "assistantId": assistant_id,
                        "threadId": thread_id,
                        "runId": current_run_id,
                        "updatedBy": "Use XYZ",
                    },
                    "operation_name": "getCurrentRun",
                }
                response = json.loads(
                    self.openai_assistant_engine.open_assistant_graphql(**payload)
                )
                # logger.info(response)
                if response["data"]["currentRun"]["status"] == "completed":
                    break

                time.sleep(5)

            payload = {
                "query": document,
                "variables": {"threadId": thread_id, "role": "assistant"},
                "operation_name": "getLastMessage",
            }
            response = json.loads(
                self.openai_assistant_engine.open_assistant_graphql(**payload)
            )
            # logger.info(response)
            last_message = response["data"]["lastMessage"]["message"]

            print(
                "AI:",
                last_message,
            )

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_assistant(self):
        variables = {
            "assistantType": "agent",
            "assistantId": "123456",
            "assistantName": "Agent ABC",
            "functions": {},
            "updatedBy": "user abc",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertUpdateAssistant",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_delete_assistant(self):
        variables = {
            "assistantType": "agent",
            "assistantId": "123456",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "deleteAssistant",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_assistant(self):
        variables = {
            "assistantType": "agent",
            "assistantId": "123456",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getAssistant",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_assistant_list(self):
        variables = {
            "assistantType": "agent",
            "assistantName": "Agent ABC",
            "pageNumber": 1,
            "limit": 100,
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getAssistantList",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_thread(self):
        variables = {
            "assistantId": "123456",
            "threadId": "XXXXXX",
            "assistantType": "agent",
            "run": {"run_id": "yyyyy", "usage": {"a": "x"}},
            "updatedBy": "user abc",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertUpdateThread",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_delete_thread(self):
        variables = {
            "assistantId": "123456",
            "threadId": "XXXXXX",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "deleteThread",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_thread(self):
        variables = {
            "assistantId": "123456",
            "threadId": "XXXXXX",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getThread",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_thread_list(self):
        variables = {
            "assistantId": "123456",
            "assistantTypes": ["agent"],
            "pageNumber": 1,
            "limit": 100,
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getThreadList",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_message(self):
        variables = {
            "threadId": "XXXXXX",
            "messageId": "123456",
            "runId": "XXXXXX",
            "role": "assistant",
            "message": "Hello, how are you?",
            "createdAt": "2024-05-13T23:23:32.000000+0000",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertUpdateMessage",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_delete_message(self):
        variables = {
            "threadId": "XXXXXX",
            "messageId": "123456",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "deleteMessage",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_message(self):
        variables = {
            "threadId": "XXXXXX",
            "messageId": "123456",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getMessage",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_message_list(self):
        variables = {
            # "threadId": "XXXXXX",
            "roles": ["user", "assistant"],
            # "message": "Hello, how are you?",
            "pageNumber": 1,
            "limit": 100,
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getMessageList",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)


if __name__ == "__main__":
    unittest.main()
