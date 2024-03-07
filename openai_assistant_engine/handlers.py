#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import time
from openai import OpenAI
from silvaengine_utility import Utility

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


# Define your handler functions here
def function_name_handler(logger, **arguments):
    # Your handler logic here
    return "Sample output for function_name"


# Define more handler functions as needed

# Mapping of function names to their respective handler functions
function_handlers = {
    "function_name": function_name_handler,
    # Add more mappings as necessary
}


def wait_on_run(logger, thread_id, run_id):
    try:
        # Initial retrieval of the run information
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
        )

        # Check the status of the run and wait until it is not in 'queued' or 'in_progress'
        while run.status in ["queued", "in_progress"]:
            # Pause to prevent excessive requests
            time.sleep(0.5)

            # Retrieve the latest status of the run
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id,  # Ensure we're getting updates for the correct run
            )

            # Handle the case where the run requires user action
            if run.status == "requires_action":
                # Logging for debug purposes
                logger.info(f"Run requires action for thread {thread_id}, run {run_id}")

                # Iterate over all required tool calls
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    function_name = tool_call.function.name
                    arguments = Utility.json_loads(
                        tool_call.function.arguments
                    )  # Ensure arguments are properly loaded

                    # Log the information about the function
                    logger.info(f"Function Name: {function_name}")
                    logger.info(f"Function Arguments: {arguments}")

                    # Check if the function name has a corresponding handler
                    if function_name in function_handlers:
                        # Get the corresponding handler function
                        handler = function_handlers[function_name]
                        # Call the handler and get the output
                        output = handler(logger, **arguments)
                    else:
                        # Log error if no handler found
                        # logger.error(f"No handler for function: {function_name}")
                        # output = "Error: Handler not found"
                        output = "submit"

                    # Submit the output for the tool call
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=[
                            {
                                "tool_call_id": tool_call.id,
                                "output": output,
                            }
                        ],
                    )

        # Return the final run id after the loop exits
        return run.id

    except Exception as e:
        # Log any exceptions that occur
        logger.error(f"An error occurred: {e}")
        raise  # Rethrow the exception for further handling


def get_response(logger, thread_id):
    try:
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        return messages
    except Exception as e:
        logger.error(e)
        raise e


# Pretty printing helper
def pretty_log(logger, messages, roles=["user", "assistant"]):
    try:
        logger.info("# Messages")
        for m in messages:
            if m.role not in roles:
                continue
            logger.info(f"{m.role}: {m.content[0].text.value}")
    except Exception as e:
        logger.error(e)
        raise e
