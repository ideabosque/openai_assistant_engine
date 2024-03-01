#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

from .handlers import (
    handlers_init,
    create_thread,
    submit_message,
    wait_on_run,
    get_response,
)


# Hook function applied to deployment
def deploy() -> list:
    return []


class OpenaiAssistantEngine(object):
    def __init__(self, logger, **setting):
        handlers_init(logger, **setting)

        self.logger = logger
        self.setting = setting

    def create_thread(self):
        return create_thread(self.logger)

    def submit_message(self, thread_id, user_message):
        return submit_message(self.logger, thread_id, user_message)

    def wait_on_run(self, thread_id, run_id):
        return wait_on_run(self.logger, thread_id, run_id)

    def get_response(self, thread_id):
        return get_response(self.logger, thread_id)
