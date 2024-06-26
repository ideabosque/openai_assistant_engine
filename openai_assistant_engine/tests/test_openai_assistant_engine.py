#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import base64
import json
import logging
import os
import sys
import threading
import time
import unittest
import wave
from io import BytesIO
from pathlib import Path

import keyboard
import pyaudio
from dotenv import load_dotenv
from pydub import AudioSegment

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
from openai_assistant_engine.handlers import (
    encode_audio_to_base64,
    text_to_base64_speech,
)


def record_audio(frames, stream, chunk):
    print("Recording... Press Space to stop.")
    while recording:
        data = stream.read(chunk)
        frames.append(data)
    print("Finished recording.")


def play_base64_audio(encoded_audio):
    audio_data = base64.b64decode(encoded_audio)
    audio_buffer = BytesIO(audio_data)
    audio_buffer.seek(0)

    # Ensure the audio is recognized as an MP3 file
    audio = AudioSegment.from_file(audio_buffer)

    playback = pyaudio.PyAudio()
    stream = playback.open(
        format=playback.get_format_from_width(audio.sample_width),
        channels=audio.channels,
        rate=audio.frame_rate,
        output=True,
    )

    data = audio.raw_data
    stream.write(data)

    stream.stop_stream()
    stream.close()
    playback.terminate()


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

    @unittest.skip("demonstrating skipping")
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

    # @unittest.skip("demonstrating skipping")
    def test_conversation_search_by_voice(self):
        global recording
        recording = False
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        chunk = 1024
        assistant_id = "asst_jUzZKojROaz6HACC1uzaqR5x"
        thread_id = None

        def start_recording():
            global recording
            if not recording:
                recording = True
                recording_thread = threading.Thread(
                    target=record_audio, args=(frames, stream, chunk)
                )
                recording_thread.start()

        def stop_recording():
            global recording
            if recording:
                recording = False

        logger.info("Start test_conversation_search_by_voice ...")
        initial_greeting = "Hello! I am an AI assistant. How can I help you today?"
        print(initial_greeting)
        response_encoded_audio = text_to_base64_speech(initial_greeting)
        play_base64_audio(response_encoded_audio)

        while True:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )

            frames = []

            print("Press Enter to start recording.")
            keyboard.wait("enter")
            start_recording()

            print("Press Space to stop recording.")
            keyboard.wait("space")
            stop_recording()

            stream.stop_stream()
            stream.close()
            audio.terminate()

            audio_buffer = BytesIO()
            wf = wave.open(audio_buffer, "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            audio_buffer.seek(0)

            encoded_audio = encode_audio_to_base64(audio_buffer)
            print("Base64 Encoded Audio:")
            print(encoded_audio)

            payload = {
                "query": document,
                "variables": {
                    "userQuery": encoded_audio,
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
                "variables": {
                    "assistantId": assistant_id,
                    "threadId": thread_id,
                    "role": "assistant",
                },
                "operation_name": "getLastMessage",
            }
            response = json.loads(
                self.openai_assistant_engine.open_assistant_graphql(**payload)
            )
            # logger.info(response)
            last_message = response["data"]["lastMessage"]["message"]
            play_base64_audio(last_message)

            print("Press 'q' to quit or any other key to continue.")
            if keyboard.read_event().name == "q":
                break

    @unittest.skip("demonstrating skipping")
    def test_graphql_live_messages(self):
        variables = {
            "threadId": "thread_3vIC9xKok5xNaPe0O4uHawef",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getLiveMessages",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

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
            "assistantId": "asst_jUzZKojROaz6HACC1uzaqR5x",
            "threadId": "thread_q6QCVXplJEhQCKmaVYrDkTzG",
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
            "threadId": "thread_3vIC9xKok5xNaPe0O4uHawef",
            "messageId": "msg_trZHt7Q8XrTd02JCgW23Vz0T",
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
