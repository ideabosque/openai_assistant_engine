#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import logging, sys, unittest, os, json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
setting = {
    "openai_api_key": os.getenv("openai_api_key"),
    "assistant_id": os.getenv("assistant_id"),
}

sys.path.insert(0, "/var/www/projects/openai_assistant_engine")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

from openai_assistant_engine import OpenaiAssistantEngine


class OpenaiAssistantEngineTest(unittest.TestCase):
    def setUp(self):
        self.openai_assistant_engine = OpenaiAssistantEngine(logger, **setting)
        logger.info("Initiate OpenaiAssistantEngineTest ...")

    def tearDown(self):
        logger.info("Destory OpenaiAssistantEngineTest ...")

    # @unittest.skip("demonstrating skipping")
    def test_conversation(self):
        logger.info("Start test_conversation ...")
        thread_id = self.openai_assistant_engine.create_thread()
        logger.info(f"thread_id: {thread_id}")

        # Load JSON data from file
        json_file_path = "formula_#83020.json"  # Replace with the actual file path
        with open(json_file_path, "r") as file:
            formula_json = json.load(file)

        message = f"Get healthcare keywords (e.g. Digestive health, Immune system support) by the formula: \n{formula_json}\n"
        +"Present the response as specified below without any explanation: \n"
        +"{ formula: #xxxxx, keywords: { <keyword-x>: { explanation: <explanation>, category: <category-x>, ingredients: [ <ingredient-x> ] } } }\n\n"


if __name__ == "__main__":
    unittest.main()
