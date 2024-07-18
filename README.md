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

## Usage

This JSON structure is used to define and configure an assistant, specifying its type, unique identifier, name, and the functions it can perform.

```plaintext
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
        },
        ...
    ]
}
```

### Parameters:
- `assistant_type`: Specifies the type of assistant, such as chat, voice, or hybrid.
- `assistant_id`: A unique identifier for the assistant, ensuring it can be distinctly recognized within a system.
- `assistant_name`: The name assigned to the assistant, which can be used for display or reference purposes.
- `functions`: A list of functions the assistant can perform. Each function is described with:
  - `module_name`: The name of the module where the function resides.
  - `class_name`: The name of the class containing the function.
  - `function_name`: The specific function to be utilized.
  - `configuration`: Configuration details for the function, which can include various settings and parameters required for its operation.