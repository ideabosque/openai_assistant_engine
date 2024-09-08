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
from openai import OpenAI
from pydub import AudioSegment
from silvaengine_utility import Utility

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "openai_api_key": os.getenv("openai_api_key"),
    "whisper_model": os.getenv("whisper_model"),
    "tts_model": os.getenv("tts_model"),
    "assistant_voice": os.getenv("assistant_voice"),
}
client = OpenAI(
    api_key=setting["openai_api_key"],
)
document = Path(
    os.path.join(os.path.dirname(__file__), "openai_assistant_engine.graphql")
).read_text()
sys.path.insert(0, "C:/Users/bibo7/gitrepo/silvaengine/openai_assistant_engine")
sys.path.insert(1, "C:/Users/bibo7/gitrepo/silvaengine/openai_funct_base")
sys.path.insert(2, "C:/Users/bibo7/gitrepo/silvaengine/silvaengine_dynamodb_base")
sys.path.insert(3, "C:/Users/bibo7/gitrepo/silvaengine/io_network_funct")
sys.path.insert(4, "C:/Users/bibo7/gitrepo/silvaengine/marketing_collection_funct")
sys.path.insert(5, "C:/Users/bibo7/gitrepo/silvaengine/price_inquiry_funct")
# sys.path.insert(0, "/var/www/projects/openai_assistant_engine")
# sys.path.insert(1, "/var/www/projects/openai_funct_base")
# sys.path.insert(2, "/var/www/projects/silvaengine_dynamodb_base")


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Specify the format for the log messages
    datefmt="%Y-%m-%d %H:%M:%S",  # Specify the date format
)
logger = logging.getLogger()

from openai_assistant_engine import OpenaiAssistantEngine


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


# Convert to base64
def encode_audio_to_base64(audio_buffer):
    encoded_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")
    return encoded_audio


def convert_base64_audio_to_text(encoded_audio):
    audio_data = base64.b64decode(encoded_audio)
    audio_buffer = BytesIO(audio_data)
    audio_buffer.seek(0)

    # Ensure the audio is recognized as an MP3 file
    audio = AudioSegment.from_file(audio_buffer)
    mp3_buffer = BytesIO()
    audio.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)

    # Send the BytesIO object directly to the transcription API
    mp3_buffer.name = "audio.mp3"  # Assign a name attribute to mimic a file
    transcript = client.audio.transcriptions.create(
        model=setting["whisper_model"], file=mp3_buffer
    )

    return transcript.text


def text_to_base64_speech(text):
    # Create the speech response with streaming
    response = client.audio.speech.create(
        model=setting["tts_model"], voice=setting["assistant_voice"], input=text
    )

    # Stream the response content into BytesIO
    audio_buffer = BytesIO()
    for chunk in response.iter_bytes():
        audio_buffer.write(chunk)

    # Reset the buffer position to the beginning
    audio_buffer.seek(0)

    return encode_audio_to_base64(audio_buffer)


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
        # print("Hello! I am an AI assistant. Please provide your detail location?")
        # assistant_id = "asst_jUzZKojROaz6HACC1uzaqR5x"
        # assistant_id = "asst_0tCDxNsScVvEVekbjSqxBThi"
        # assistant_id = "asst_tyXJ4FnLLUAD76umXFuNoXv4"
        # assistant_id = "asst_Xrt7Ls4Arhj4QV71mtxJcYqm"
        assistant_id = "asst_esIGKrZY4ikA6imyfsjvjMz3"
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
                    # "instructions": "You are the greatful assistant.",
                    "threadId": thread_id,
                    "updatedBy": "Use XYZ",
                },
                "operation_name": "askOpenAi",
            }
            response = json.loads(
                self.openai_assistant_engine.open_assistant_graphql(**payload)
            )
            logger.info(response)
            if response.get("errors"):
                raise Exception(response["errors"])

            function_name = response["data"]["askOpenAi"]["functionName"]
            task_uuid = response["data"]["askOpenAi"]["taskUuid"]
            thread_id = response["data"]["askOpenAi"]["threadId"]
            current_run_id = response["data"]["askOpenAi"]["currentRunId"]

            while True:
                payload = {
                    "query": document,
                    "variables": {
                        "functionName": function_name,
                        "taskUuid": task_uuid,
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
                logger.info(response)
                if response.get("errors"):
                    raise Exception(response["errors"])

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
            logger.info(response)
            if response.get("errors"):
                raise Exception(response["errors"])

            last_message = response["data"]["lastMessage"]["message"]

            print(
                "AI:",
                last_message,
            )

    @unittest.skip("demonstrating skipping")
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
                    "userQuery": convert_base64_audio_to_text(encoded_audio),
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
            if response.get("errors"):
                raise Exception(response["errors"])

            function_name = response["data"]["askOpenAi"]["functionName"]
            task_uuid = response["data"]["askOpenAi"]["taskUuid"]
            thread_id = response["data"]["askOpenAi"]["threadId"]
            current_run_id = response["data"]["askOpenAi"]["currentRunId"]

            while True:
                payload = {
                    "query": document,
                    "variables": {
                        "functionName": function_name,
                        "taskUuid": task_uuid,
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
                logger.info(response)
                if response.get("errors"):
                    raise Exception(response["errors"])

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
            logger.info(response)
            if response.get("errors"):
                raise Exception(response["errors"])

            last_message = response["data"]["lastMessage"]["message"]
            print("AI:", last_message)
            play_base64_audio(text_to_base64_speech(last_message))

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
    def test_graphql_insert_file(self):
        # Path to the local file
        file_path = "C:/Users/bibo7/gitrepo/silvaengine/openai_assistant_engine/openai_assistant_engine/tests/openai_assistant_engine.graphql"

        # Extract the filename
        filename = os.path.basename(file_path)

        # Read the file and encode it in Base64
        encoded_content = None
        with open(file_path, "rb") as file:
            file_content = file.read()
            encoded_content = base64.b64encode(file_content).decode("utf-8")

        variables = {
            "filename": filename,
            "encodedContent": encoded_content,
            "purpose": "fine-tune",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertFile",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_delete_file(self):
        variables = {
            "fileId": "file-pIgGUzBEPKX68tylWU3vZa8j",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "deleteFile",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_file(self):
        variables = {
            "fileId": "file-vYF15hO1FJZNUNLtTcw4LEye",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getFile",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_file_content(self):
        variables = {
            "fileId": "file-smYkW2yh3WbIpgg8ZvdJy0sK",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getFileContent",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)
        response = json.loads(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_files(self):
        # variables = {"purpose": "assistants"}
        variables = {}
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getFiles",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_assistant(self):
        variables = {
            "assistantType": "conversation",
            "assistantId": "asst_rhGhCdlTpQNv3ClPqMIxP7kn",
            "assistantName": "Conversation ABC",
            "model": "gpt-4o-2024-05-13",
            "instructions": "You are a helpful assistant.",
            "configuration": {"endpoint_id": "api"},
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
            "assistantType": "conversation",
            "assistantId": "asst_rhGhCdlTpQNv3ClPqMIxP7kn",
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
            "assistantType": "conversation",
            "assistantId": "asst_jUzZKojROaz6HACC1uzaqR5x",
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
            "assistantType": "conversation",
            # "assistantName": "Agent ABC",
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

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_tool_call(self):
        variables = {
            "runId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "toolCallId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "toolType": "code_interpreter",
            "name": "my_tool",
            "arguments": {"input": "print('Hello, World!')"},
            "content": "print('Hello, World!')",
            "createdAt": "2024-05-13T23:23:32.000000+0000",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertUpdateToolCall",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_delete_tool_call(self):
        variables = {
            "runId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "toolCallId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "deleteToolCall",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_tool_call(self):
        variables = {
            "runId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "toolCallId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getToolCall",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_tool_call_list(self):
        variables = {
            "runId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "toolTypes": ["code_interpreter"],
            "pageNumber": 1,
            "limit": 100,
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getToolCallList",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_fine_tuning_messages(self):
        variables = {
            "assistantType": "conversation",
            "assistantId": "asst_esIGKrZY4ikA6imyfsjvjMz3",
            "retrain": True,
            # "trainedMessageUuids": [
            #     "6666396519121752559",
            #     "6695032152710910447",
            #     "6695032157005877743",
            # ],
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertUpdateFineTuningMessages",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_fine_tuning_message(self):
        variables = {
            "assistantId": "asst_BUoN3ONJdMzqVpvi0wrplGPj",
            "messageUuid": "7947670243734852079",
            "threadId": "thread_3c2YnLa2aWz4o1ZOxZd28Y0k",
            "timestamp": "2024-05-13T23:23:32.000000+0000",
            "role": "assistant",
            "toolCalls": [
                {
                    "toolType": "code_interpreter",
                    "name": "my_tool",
                    "arguments": {"input": "print('Hello, World!')"},
                }
            ],
            "toolCallId": "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "content": "print('Hello, World!')",
            "weight": 1.0,
            "trained": True,
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertUpdateFineTuningMessage",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_delete_fine_tuning_message(self):
        variables = {
            "model": "gpt-4o-2024-05-13",
            "timestamp": "2024-05-13T23:23:32.000000+0000",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "deleteFineTuningMessage",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_fine_tuning_message(self):
        variables = {
            "assistantId": "asst_BUoN3ONJdMzqVpvi0wrplGPj",
            "messageUuid": "7947670243734852079",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getFineTuningMessage",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    # @unittest.skip("demonstrating skipping")
    def test_graphql_fine_tuning_message_list(self):
        variables = {
            "assistantId": "asst_esIGKrZY4ikA6imyfsjvjMz3",
            "fromDate": "2024-05-13T23:23:32.000000+0800",
            "roles": ["user", "assistant", "tool"],
            # "roles": ["system"],
            "pageNumber": 1,
            "limit": 10,
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getFineTuningMessageList",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_insert_update_async_task(self):
        variables = {
            "functionName": "async_openai_assistant_stream",
            "taskUuid": "XXXXXXXXXXXXXXXXXXX",
            # "arguments": {},
            "status": "completed",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "insertUpdateAsyncTask",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_delete_async_task(self):
        variables = {
            "functionName": "async_openai_assistant_stream",
            "taskUuid": "XXXXXXXXXXXXXXXXXXX",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "deleteAsyncTask",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_async_task(self):
        variables = {
            "functionName": "async_openai_assistant_stream",
            "taskUuid": "XXXXXXXXXXXXXXXXXXX",
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getAsyncTask",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)

    @unittest.skip("demonstrating skipping")
    def test_graphql_async_task_list(self):
        variables = {
            "functionName": "async_openai_assistant_stream",
            "statuses": ["completed"],
            "pageNumber": 1,
            "limit": 100,
        }
        payload = {
            "query": document,
            "variables": variables,
            "operation_name": "getAsyncTaskList",
        }
        response = self.openai_assistant_engine.open_assistant_graphql(**payload)
        logger.info(response)


if __name__ == "__main__":
    unittest.main()
