#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import time
from openai import OpenAI

client = None
assistant_id = None


def handlers_init(logger, **setting):
    global client, assistant_id
    try:
        client = OpenAI(
            api_key=setting["openai_api_key"],
        )
        assistant_id = setting["assistant_id"]
    except Exception as e:
        logger.error(e)
        raise e


def create_thread(logger):
    try:
        thread = client.beta.threads.create()
        return thread.id
    except Exception as e:
        logger.error(e)
        raise e


def submit_message(logger, thread_id, user_message):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=user_message
        )
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        return run.id
    except Exception as e:
        logger.error(e)
        raise e


def wait_on_run(logger, thread_id, run_id):
    try:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
        )

        while run.status == "queued" or run.status == "in_progress":

            if run.status == "requires_action":
                # Extract single tool call
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    # arguments = json.loads(tool_call.function.arguments)

                    # logger.info(f"Function Name:, {name}")
                    # logger.info(f"Function Arguments: {arguments}")
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=[
                            {
                                "tool_call_id": tool_call.id,
                                "output": "Submit",
                                "output": eval(f"{function_name}_handler")(
                                    logger, **arguments
                                ),
                            }
                        ],
                    )

            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id,
            )
        return run.id
    except Exception as e:
        logger.error(e)
        raise e


def get_response(logger, thread_id):
    try:
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        return messages
    except Exception as e:
        logger.error(e)
        raise e
