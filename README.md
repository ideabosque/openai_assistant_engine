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

### Loading the GraphQL Schema

Begin by loading the GraphQL schema into a document parameter. This schema defines the structure of your queries and mutations.

```graphql
fragment AskOpenAIInfo on AskOpenAIType {
    assistantId
    threadId
    userQuery
    currentRunId
}

fragment LiveMessageInfo on LiveMessageType {
    threadId
    runId
    messageId
    role
    message
    createdAt
}

fragment CurrentRunInfo on CurrentRunType {
    threadId
    runId
    status
    usage
}

fragment AssistantInfo on AssistantType {
    assistantType
    assistantId
    assistantName
    functions
    updatedBy
    createdAt
    updatedAt
}

fragment AssistantListInfo on AssistantListType {
    assistantList{
        ...AssistantInfo
    }
    pageSize
    pageNumber
    total
}

fragment ThreadInfo on ThreadType {
    assistant
    threadId
    isVoice
    runs
    updatedBy
    createdAt
    updatedBy
}

fragment ThreadListInfo on ThreadListType {
    threadList{
        ...ThreadInfo
    }
    pageSize
    pageNumber
    total
}

fragment MessageInfo on MessageType {
    threadId
    runId
    messageId
    role
    message
    createdAt
}

fragment MessageListInfo on MessageListType {
    messageList{
        ...MessageInfo
    }
    pageSize
    pageNumber
    total
}

query ping {
    ping
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

query getLiveMessages(
    $threadId: String!,
    $roles: [String],
    $order: String
) {
    liveMessages(
        threadId: $threadId,
        roles: $roles,
        order: $order
    ){
        ...LiveMessageInfo
    }
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

query getAssistant(
    $assistantType: String!,
    $assistantId: String!
) {
    assistant(
        assistantType: $assistantType,
        assistantId: $assistantId
    ) {
        ...AssistantInfo
    }
}

query getAssistantList(
    $pageNumber: Int, 
    $limit: Int,
    $assistantType: String,
    $assistantName: String
) {
    assistantList(
        pageNumber: $pageNumber,
        limit: $limit,
        assistantType: $assistantType,
        assistantName: $assistantName
    ) {
        ...AssistantListInfo
    }
}

mutation insertUpdateAssistant(
    $assistantType: String!,
    $assistantId: String!,
    $assistantName: String!,
    $functions: [JSON]!,
    $updatedBy: String!
) {
    insertUpdateAssistant(
        assistantType: $assistantType,
        assistantId: $assistantId,
        assistantName: $assistantName,
        functions: $functions,
        updatedBy: $updatedBy
    ) {
        assistant{
            ...AssistantInfo
        }
    }
}

mutation deleteAssistant(
    $assistantType: String!,
    $assistantId: String!
) {
    deleteAssistant(
        assistantType: $assistantType,
        assistantId: $assistantId
    ) {
        ok
    }
}

query getThread(
    $assistantId: String!,
    $threadId: String!
) {
    thread(
        assistantId: $assistantId,
        threadId: $threadId
    ) {
        ...ThreadInfo
    }
}

query getThreadList(
    $pageNumber: Int, 
    $limit: Int,
    $assistantId: String,
    $assistantTypes: [String]
) {
    threadList(
        pageNumber: $pageNumber,
        limit: $limit,
        assistantId: $assistantId,
        assistantTypes: $assistantTypes
    ) {
        ...ThreadListInfo
    }
}

mutation insertUpdateThread(
    $assistantId: String!,
    $threadId: String!,
    $assistantType: String!,
    $run: JSON,
    $updatedBy: String!
) {
    insertUpdateThread(
        assistantId: $assistantId,
        threadId: $threadId,
        assistantType: $assistantType,
        run: $run,
        updatedBy: $updatedBy
    ) {
        thread{
            ...ThreadInfo
        }
    }
}

mutation deleteThread(
    $assistantId: String!,
    $threadId: String!
) {
    deleteThread(
        assistantId: $assistantId,
        threadId: $threadId
    ) {
        ok
    }
}

query getMessage(
    $threadId: String!,
    $messageId: String!
) {
    message(
        threadId: $threadId,
        messageId: $messageId
    ) {
        ...MessageInfo
    }
}

query getMessageList(
    $pageNumber: Int, 
    $limit: Int,
    $threadId: String,
    $roles: [String],
    $message: String
) {
    messageList(
        pageNumber: $pageNumber,
        limit: $limit,
        threadId: $threadId,
        roles: $roles,
        message: $message
    ) {
        ...MessageListInfo
    }
}

mutation insertUpdateMessage(
    $threadId: String!,
    $messageId: String!,
    $runId: String!,
    $role: String!,
    $message: String!,
    $createdAt: DateTime!
) {
    insertUpdateMessage(
        threadId: $threadId,
        messageId: $messageId,
        runId: $runId,
        role: $role,
        message: $message,
        createdAt: $createdAt
    ) {
        message{
            ...MessageInfo
        }
    }
}

mutation deleteMessage(
    $threadId: String!,
    $messageId: String!
) {
    deleteMessage(
        threadId: $threadId,
        messageId: $messageId
    ) {
        ok
    }
}
```

This GraphQL schema provides you with the tools to construct queries and mutations to interact with OpenAI assistant API and DynamoDB tables for future usage. Each query or mutation is named and can accept variables for customization.

### Using the Payload for Querying or Mutating Data

You can use the following payload structure to execute GraphQL queries or mutations programmatically:

```python
payload = {
    "query": document,
    "variables": variables,
    "operation_name": "getSelectValues",
}
```

Parameters:
- `query`: The GraphQL query or mutation document you've loaded earlier.
- `variables`: Any variables needed for your GraphQL query or mutation.
- `operation_name`: The name of the operation to be executed, corresponding to a named query or mutation in your GraphQL schema.

These parameters allow you to customize your GraphQL requests as needed, making your interactions with NetSuite data highly flexible.