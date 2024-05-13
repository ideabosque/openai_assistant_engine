#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import logging, sys, unittest, os, json, time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "openai_api_key": os.getenv("openai_api_key"),
    # "assistant_functions": {
    #     "asst_jUzZKojROaz6HACC1uzaqR5x": {
    #         "inquiry_data": {
    #             "module_name": "redis_search",
    #             "class_name": "RedisSearch",
    #             "configuration": {
    #                 "openai_api_key": os.getenv("openai_api_key"),
    #                 "EMBEDDING_MODEL": os.getenv(
    #                     "embedding_model", "text-embedding-3-small"
    #                 ),
    #                 "REDIS_HOST": "localhost",
    #                 "REDIS_PORT": 6379,
    #                 "REDIS_PASSWORD": "",  # default for passwordless Redis},
    #             },
    #         }
    #     }
    # },
}

document = Path(
    os.path.join(os.path.dirname(__file__), "openai_assistant_engine.graphql")
).read_text()
sys.path.insert(0, "C:/Users/bibo7/gitrepo/silvaengine/openai_assistant_engine")
sys.path.insert(1, "C:/Users/bibo7/gitrepo/silvaengine/redis_search")

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
                    "question": user_input,
                    "assistantId": assistant_id,
                    "assistantType": "conversation",
                    "threadId": thread_id,
                },
                "operation_name": "askOpenAi",
            }
            response = json.loads(
                self.openai_assistant_engine.open_assistant_graphql(**payload)
            )
            logger.info(response)
            thread_id = response["data"]["askOpenAi"]["threadId"]
            current_run_id = response["data"]["askOpenAi"]["currentRunId"]

            while True:
                payload = {
                    "query": document,
                    "variables": {
                        "assistantId": assistant_id,
                        "threadId": thread_id,
                        "runId": current_run_id,
                    },
                    "operation_name": "getCurrentRun",
                }
                response = json.loads(
                    self.openai_assistant_engine.open_assistant_graphql(**payload)
                )
                logger.info(response)
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


if __name__ == "__main__":
    unittest.main()
