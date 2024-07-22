# openai_assistant_engine

## **Introduction**

The module harnesses the comprehensive capabilities of the OpenAI Assistant API to perform various functions. It handles a wide range of tasks and meticulously records usage and conversation data. This ensures that every interaction is captured and stored for further analysis and continuous improvement. By maintaining detailed records, the module allows for a deeper understanding of user interactions, enhancing the system's performance over time. 

In addition, it possesses the capability to manage, configure, and seamlessly integrate with OpenAI Assistant's "[function calling](https://platform.openai.com/docs/assistants/tools/function-calling/quickstart)". This feature allows it to interact with external data sources and make API calls to outside systems. By leveraging this functionality, it can access and utilize a wide range of external information and resources efficiently.

Furthermore, the module is designed to be installed with the SilverEngine AWS Serverless framework. This integration facilitates the module to act as a proxy (OpenAI Assistant API Proxy), effectively managing and executing the required functionalities. The combination of these advanced features and seamless integration ensures a robust and efficient system capable of meeting diverse operational requirements. 

### Key Features

1. **Comprehensive Capabilities Utilization:**
    - Harnesses the full potential of the OpenAI Assistant API.
    - Performs a wide range of tasks efficiently.
2. **Usage and Conversation Data Recording:**
    - Meticulously records all interactions.
    - Stores usage and conversation data for further analysis.
    - Ensures continuous improvement by understanding user interactions.
3. **Function Calling Management:**
    - Manages and configures function calling within the OpenAI Assistant API.
    - Enables interaction with external data sources.
    - Facilitates API calls to outside systems for accessing and utilizing external information and resources.
4. **Integration with SilverEngine AWS Serverless Framework:**
    - Designed to be installed with the SilverEngine AWS Serverless framework.
    - Acts as a proxy for the OpenAI Assistant API.
    - Manages and executes required functionalities seamlessly.
5. **Operational Efficiency and Robustness:**
    - Ensures a robust system capable of meeting diverse operational requirements.
    - Enhances system performance over time through detailed record-keeping and analysis.

## Installation

To easily install the OpenAI Assistant Engine using pip and Git, execute the following command in your terminal:

```bash
$ python -m pip install git+ssh://git@github.com/ideabosque/silvaengine_utility.git@main#egg=silvaengine_utility
$ python -m pip install git+ssh://git@github.com/ideabosque/silvaengine_dynamodb_base.git@main#egg=silvaengine_dynamodb_base
$ python -m pip install git+ssh://git@github.com/ideabosque/openai_assistant_engine.git@main#egg=openai_assistant_engine
```

## Configuration
Configuring the OpenAI Assistant Engine requires setting up specific files and environment variables. Follow these steps to ensure proper configuration:

### AWS DynamoDB Tables

To enable efficient caching and data management, create the following AWS DynamoDB tables:

1. **`oae-assistants`**: This table uses a partition key (`assistant_type`) and a sort key (`assistant_id`).

2. **`oae-threads`**: Configure this table with a partition key (`assistant_id`) and a sort key (`thread_id`).

3. **`oae-messages`**: Configure this table with a partition key (`thread_id`) and a sort key (`message_id`).

### .env File

Create a `.env` file in your project directory with the following content:

```plaintext
region_name=YOUR_AWS_REGION
aws_access_key_id=YOUR_AWS_ACCESS_KEY_ID
aws_secret_access_key=YOUR_AWS_SECRET_ACCESS_KEY
openai_api_key=OPENAI_API_KEY
embedding_model=EMBEDDING_MODEL
```

Replace the placeholders (`YOUR_AWS_REGION`, `YOUR_AWS_ACCESS_KEY_ID`, `YOUR_AWS_SECRET_ACCESS_KEY`, `OPENAI_API_KEY`, and `EMBEDDING_MODEL`) with your actual AWS region, AWS Access Key ID, AWS Secret Access Key, OpenAI API Key, and Embedding Model.

### Assistant Configuration Setup

The following JSON structure is utilized to define and configure an assistant within the **`oae-assistants`** table. This structure specifies the assistant's type, unique identifier, name, and the various functions it can perform.

```json
{
    "assistant_type": <ASSISTANT_TYPE>,
    "assistant_id": <ASSISTANT_ID>,
    "assistant_name": <ASSISTANT_NAME>,
    "functions": [
        {
            "module_name": <MODULE_NAME>,
            "class_name": <CLASS_NAME>,
            "function_name": <FUNCTION_NAME>,
            "configuration": {
                ...<CONFIGURATION>
            }
        }
    ]
}
```

- `assistant_type`: Defines the category of the assistant, such as a conversational agent or a task-specific agent.
- `assistant_id`: A unique identifier assigned to the assistant by OpenAI, ensuring its distinct recognition within the system.
- `assistant_name`: The designated name for the assistant, used for display and reference purposes.
- `functions`: An array of functions that the assistant can perform. Each function is detailed as follows:
  - `module_name`: The name of the module containing the function.
  - `class_name`: The name of the class where the function is defined.
  - `function_name`: The specific function to be executed.
  - `configuration`: Configuration details for the function, which include various settings and parameters necessary for its operation.

## Usage

Utilizing the OpenAI Assistant Engine is straightforward. Below, you'll find examples that illustrate how to construct GraphQL queries and mutations for seamless interaction with OpenAI Assistant.

### Call GraphQL API

Here is a detailed example of how to call a GraphQL API using Python. This method leverages the `requests` library to send a POST request to the API endpoint, with the necessary query and variables provided as parameters.

**Parameters:**
- `AWS API Gateway:invoke URL`: The endpoint URL for invoking the API.
- `AWS API Gateway:API Key`: The API key required for accessing the endpoint.
- `query`: The GraphQL query to be executed.
- `variables`: The variables needed for the GraphQL query.

```python
import requests
import json

def call_graphql_api(api_url, api_key, query, variables):
    """
    Call the GraphQL API with the provided query and variables.
    
    Parameters:
    - api_url (str): The endpoint URL for invoking the API.
    - api_key (str): The API key required for accessing the endpoint.
    - query (str): The GraphQL query to be executed.
    - variables (dict): The variables needed for the GraphQL query.
    
    Returns:
    - response (str): The response text from the API.
    """
    
    # Combine the query and variables into a single payload
    payload = {
        "query": query,
        "variables": variables
    }

    # Convert the payload to a JSON string
    payload_json = json.dumps(payload)

    # Define the headers for the HTTP request
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }

    # Send the HTTP POST request to the API endpoint
    response = requests.post(api_url, headers=headers, data=payload_json)

    # Return the response text
    return response.text

# Define the API endpoint and API key
api_url = <AWS API Gateway:invoke URL>
api_key = <AWS API Gateway:API Key>

# Define the GraphQL query
query = <query>

# Define the variables for the GraphQL query
variables = <variables>

# Call the function and print the response
response_text = call_graphql_api(api_url, api_key, query, variables)
print(response_text)
```

In this example, the `call_graphql_api` function takes the `api_url`, `api_key`, `query`, and `variables` as parameters, allowing for a flexible and reusable way to interact with the GraphQL API. The function combines the query and variables into a payload, sends an HTTP POST request to the API endpoint, and returns the response text.

You can replace the `query` and `variables` with your specific GraphQL query and corresponding variables to customize this script for different use cases.

### Example: Querying 'ask_openai'

Let's delve into a comprehensive example of querying the `ask_openai` operation. This involves leveraging the OpenAI assistant to execute a prompt and receive a detailed response. By integrating this capability, you can harness the power of advanced AI to process user queries, facilitating enhanced interaction and information retrieval.

**Parameters:**
- `user_query`: The search query provided by the user.
- `assistant_id`: The unique identifier for the assistant, generated from the OpenAI console.
- `assistant_type`: The category of assistant being utilized.
- `thread_id`: The identifier for the conversation thread. This can be `None` for the initial prompt of a conversation. For a continuing conversation, the thread id should match the previous interaction's thread id.
- `updated_by`: The identifier of the user or system that last modified the query.

```python
.....
# Define the GraphQL query and variables
query = """
    fragment AskOpenAIInfo on AskOpenAIType {
        assistantId
        threadId
        userQuery
        currentRunId
    }

    query askOpenAi(
        $assistantType: String!,
        $assistantId: String!,
        $userQuery: String!,
        $updatedBy: String!,
        $threadId: String
    ) {
        askOpenAi(
            assistantType: $assistantType,
            assistantId: $assistantId,
            userQuery: $userQuery,
            updatedBy: $updatedBy,
            threadId: $threadId
        ) {
            ...AskOpenAIInfo
        }
    }
"""

# Define the variables for the GraphQL query
variables = {
    "userQuery": <user_query>,
    "assistantId": <assistant_id>,
    "assistantType": <assistant_type>,
    "threadId": <thread_id>,
    "updatedBy": <updated_by>
}
.....
```

This example illustrates the seamless integration of GraphQL with the OpenAI assistant API, enabling sophisticated interaction with AI-driven query processing. By utilizing this approach, developers can create robust applications that leverage the advanced capabilities of the OpenAI assistant, providing users with accurate and timely information based on their queries.

### Example: Querying 'current_run'

The following example demonstrates how to retrieve information about the current run. This can be used to check whether the run has been completed or is still in progress.

**Parameters:**
- `assistant_id`: A unique identifier for the assistant, generated from the OpenAI console.
- `thread_id`: The identifier for the conversation thread. For an initial prompt, this can be `None`. For ongoing conversations, it should match the thread id from the previous interaction.
- `run_id`: The identifier for the current run.
- `updated_by`: The identifier of the user or system that last modified the query.

```python
# Define the GraphQL query and variables
query = """
    fragment CurrentRunInfo on CurrentRunType {
        threadId
        runId
        status
        usage
    }

    query getCurrentRun(
        $assistantId: String!,
        $threadId: String!,
        $runId: String!,
        $updatedBy: String!
    ) {
        currentRun(
            assistantId: $assistantId,
            threadId: $threadId,
            runId: $runId,
            updatedBy: $updatedBy
        ) {
            ...CurrentRunInfo
        }
    }
"""

# Define the variables for the GraphQL query
variables = {
    "assistantId": <assistant_id>,
    "threadId": <thread_id>,
    "runId": <run_id>,
    "updatedBy": <updated_by>,
}
```

This query retrieves essential details about the current run, such as its status and usage, helping you monitor the progress and completion status efficiently.


### Example: Querying 'last_message'

The following example demonstrates how to retrieve the last message in a conversation thread. This can be used to fetch the most recent message along with its details.

**Parameters:**
- `assistant_id`: A unique identifier for the assistant, generated from the OpenAI console. This can be optional in this query.
- `thread_id`: The identifier for the conversation thread. It is required for identifying the specific conversation.
- `role`: The role of the message sender (e.g., user or assistant). This helps filter the messages based on the sender's role.

```python
# Define the GraphQL query and variables
query = """
    fragment LiveMessageInfo on LiveMessageType {
        threadId
        runId
        messageId
        role
        message
        createdAt
    }

    query getLastMessage(
        $assistantId: String,
        $threadId: String!,
        $role: String!
    ) {
        lastMessage(
            assistantId: $assistantId,
            threadId: $threadId,
            role: $role
        ) {
            ...LiveMessageInfo
        }
    }
"""

# Define the variables for the GraphQL query
variables = {
    "assistantId": <assistant_id>,
    "threadId": <thread_id>, 
    "role": <role>
}
```

This query retrieves the latest message in a specified conversation thread, providing details such as the message content, sender's role, and the timestamp of when the message was created.

### Complete Integration: A Chatbot with External Redis Vector Search Database

The following script demonstrates a chatbot that leverages external functions to query data from a Redis vector search database and summarizes the results for the user. This example uses the OpenAI API for interaction.

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "bibow"

import logging
import requests
import json
import time
import sys
import os
from dotenv import load_dotenv

"""
This script facilitates interactions with an AI assistant using the OpenAI API.
Users can submit queries, which are processed by the API, and receive responses from the AI assistant.
The program operates in a loop, allowing continuous user interaction until "exit" is typed.
"""

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define the API endpoint and headers
API_URL = os.getenv("API_URL")
HEADERS = {
    'x-api-key': os.getenv("API_KEY"),
    'Content-Type': 'application/json'
}
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# Main function
def main():
    logger.info("Starting AI assistant interaction...")
    print("Hello! I am an AI assistant. How can I help you today?")
    thread_id = None

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Exiting the program.")
            break

        payload = json.dumps({
            "query": """
                fragment AskOpenAIInfo on AskOpenAIType {
                    assistantId
                    threadId
                    userQuery
                    currentRunId
                }

                query askOpenAi(
                    $assistantType: String!,
                    $assistantId: String!,
                    $userQuery: String!,
                    $updatedBy: String!,
                    $threadId: String
                ) {
                    askOpenAi(
                        assistantType: $assistantType,
                        assistantId: $assistantId,
                        userQuery: $userQuery,
                        updatedBy: $updatedBy,
                        threadId: $threadId
                    ) {
                        ...AskOpenAIInfo
                    }
                }
            """,
            "variables": {
                "userQuery": user_input,
                "assistantId": ASSISTANT_ID,
                "assistantType": "conversation",
                "threadId": thread_id,
                "updatedBy": "Use XYZ"
            }
        })

        response = requests.post(API_URL, headers=HEADERS, data=payload).json()
        logger.info(response)
        thread_id = response["data"]["askOpenAi"]["threadId"]
        current_run_id = response["data"]["askOpenAi"]["currentRunId"]

        while True:
            payload = json.dumps({
                "query": """
                    fragment CurrentRunInfo on CurrentRunType {
                        threadId
                        runId
                        status
                        usage
                    }

                    query getCurrentRun(
                        $assistantId: String!,
                        $threadId: String!,
                        $runId: String!,
                        $updatedBy: String!
                    ) {
                        currentRun(
                            assistantId: $assistantId,
                            threadId: $threadId,
                            runId: $runId,
                            updatedBy: $updatedBy
                        ){
                            ...CurrentRunInfo
                        }
                    }
                """,
                "variables": {
                    "assistantId": ASSISTANT_ID,
                    "threadId": thread_id,
                    "runId": current_run_id,
                    "updatedBy": "Use XYZ"
                }
            })

            response = requests.post(API_URL, headers=HEADERS, data=payload).json()
            logger.info(response)
            if response["data"]["currentRun"]["status"] == "completed":
                break

            time.sleep(5)

        payload = json.dumps({
            "query": """
                fragment LiveMessageInfo on LiveMessageType {
                    threadId
                    runId
                    messageId
                    role
                    message
                    createdAt
                }

                query getLastMessage(
                    $assistantId: String,
                    $threadId: String!,
                    $role: String!
                ) {
                    lastMessage(
                        assistantId: $assistantId,
                        threadId: $threadId,
                        role: $role
                    ){
                        ...LiveMessageInfo
                    }
                }
            """,
            "variables": {
                "assistantId": ASSISTANT_ID,
                "threadId": thread_id,
                "role": "assistant"
            }
        })

        response = requests.post(API_URL, headers=HEADERS, data=payload).json()
        logger.info(response)
        last_message = response["data"]["lastMessage"]["message"]

        print("AI:", last_message)

if __name__ == "__main__":
    main()
```

### Detailed Steps and Setup

0. **Configuration for the Function on the API**: Set up the function and the module in the `oae-assistants` table.
   
   Example function setup:
   ```json
   {
     "assistant_type": "conversation",
     "assistant_id": "asst_XXXXXXXXXXXXXXXXX",
     "assistant_name": "Data Inquiry Assistant",
     "functions": [
       {
         "class_name": "OpenAIFunctBase",
         "configuration": {
           "endpoint_id": "api"
         },
         "function_name": "inquiry_data",
         "module_name": "openai_funct_base"
       }
     ]
   }
   ```

1. **Setup Logging**: The script begins by setting up logging to record information and debug messages.

2. **Load Environment Variables**: Using `dotenv`, the script loads necessary environment variables from a `.env` file.

3. **Define API Endpoint and Headers**: The script sets up the API URL and headers, including an API key for authentication.

4. **Main Function**:
   - **Initialization**: Logs the start of the AI assistant interaction and prints a welcome message.
   - **User Input Loop**: Continuously prompts the user for input until "exit" is typed.
   - **API Request for User Query**: Constructs and sends a GraphQL query to the API with the user’s input, retrieves the response, and extracts the `thread_id` and `current_run_id`.
   - **Check Query Status**: Continuously checks the status of the current run until it is completed.
   - **Retrieve Last Message**: Fetches the last message from the AI assistant and prints it.

### Required Variables and `.env` Setup

The script requires the following variables, which should be defined in a `.env` file:

- **API_URL**: The endpoint URL for the API.
- **API_KEY**: The API key used for authentication.
- **ASSISTANT_ID**: The unique identifier for the AI assistant.

Example `.env` file:

```plaintext
API_URL=https://api.yourservice.com/v1/graphql
API_KEY=your_api_key_here
ASSISTANT_ID=your_assistant_id_here
```

### How to Run

1. **Install Dependencies**: Ensure you have the required Python packages:
   ```bash
   pip install requests python-dotenv
   ```

2. **Create `.env` File**: In the same directory as your script, create a `.env` file with the necessary variables.

3. **Run the Script**: Execute the script:
   ```bash
   python your_script_name.py
   ```

This setup will allow the chatbot to interact with users, query an external Redis vector search database, and provide summarized responses based on the user's queries.

### Example Video Demonstration

In this video demonstration, you will see the chatbot script in action. This demonstration will help you visualize the interaction process and understand how the chatbot handles user queries in real-time.