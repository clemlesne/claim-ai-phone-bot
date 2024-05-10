# Call Center AI

AI-powered call center solution with Azure and OpenAI GPT.

<!-- github.com badges -->
[![Last release date](https://img.shields.io/github/release-date/clemlesne/call-center-ai)](https://github.com/clemlesne/call-center-ai/releases)
[![Project license](https://img.shields.io/github/license/clemlesne/call-center-ai)](https://github.com/clemlesne/call-center-ai/blob/main/LICENSE)

## Overview

A French demo is avaialble on YouTube. Do not hesitate to watch the demo in x1.5 speed to get a quick overview of the project.

[![French demo](https://img.youtube.com/vi/4r5s-NZ9CuY/maxresdefault.jpg)](https://youtube.com/watch?v=4r5s-NZ9CuY)

Main interactions shown in the demo:

1. User calls the call center
2. The bot answers and the conversation starts
3. The bot stores conversation, claim and todo list in the database

Extract of the data stored during the call:

```json
{
  "claim": {
    "incident_date_time": "2024-01-11T19:33:41",
    "incident_description": "The vehicle began to travel with a burning smell and the driver pulled over to the side of the freeway.",
    "policy_number": "B01371946",
    "policyholder_phone": "[number masked for the demo]",
    "policyholder_name": "Clémence Lesne",
    "vehicle_info": "Ford Fiesta 2003"
  },
  "reminders": [
    {
      "description": "Check that all the information in Clémence Lesne's file is correct and complete.",
      "due_date_time": "2024-01-18T16:00:00",
      "title": "Check Clémence file"
    }
  ]
}
```

### Features

> [!NOTE]
> This project is a proof of concept. It is not intended to be used in production. This demonstrates how can be combined Azure Communication Services, Azure Cognitive Services and Azure OpenAI to build an automated call center solution.

- [x] Access the claim on a public website
- [x] Access to customer conversation history
- [x] Allow user to change the language of the conversation
- [x] Bot can be called from a phone number
- [x] Bot use multiple voice tones (e.g. happy, sad, neutral) to keep the conversation engaging
- [x] Company products (= lexicon) can be understood by the bot (e.g. a name of a specific insurance product)
- [x] Create by itself a todo list of tasks to complete the claim
- [x] Customizable prompts
- [x] Disengaging from a human agent when needed
- [x] Filter out inappropriate content from the LLM, like profanity or concurrence company names
- [x] Fine understanding of the customer request with GPT-4 Turbo
- [x] Follow a specific data schema for the claim
- [x] Has access to a documentation database (few-shot training / RAG)
- [x] Help the user to find the information needed to complete the claim
- [x] Lower AI Search cost by usign a Redis cache
- [x] Monitoring and tracing with Application Insights
- [x] Responses are streamed from the LLM to the user, to avoid long pauses
- [x] Send a SMS report after the call
- [x] Take back a conversation after a disengagement
- [ ] Call back the user when needed
- [ ] Simulate a IVR workflow

### User report after the call

A report is available at `https://[your_domain]/report/[phone_number]` (like `http://localhost:8080/report/%2B133658471534`). It shows the conversation history, claim data and reminders.

![User report](./docs/user_report.jpg)

### High level architecture

```mermaid
---
title: System diagram (C4 model)
---
graph
  user(["User"])
  agent(["Agent"])

  api["Claim AI"]

  api -- Transfer to --> agent
  api -. Send voice .-> user
  user -- Call --> api
```

### Component level architecture

```mermaid
---
title: Claim AI component diagram (C4 model)
---
graph LR
  agent(["Agent"])
  user(["User"])

  subgraph "Claim AI"
    ai_search[("RAG\n(AI Search)")]
    api["API"]
    communication_service_sms["SMS gateway\n(Communication Services)"]
    communication_service["Call gateway\n(Communication Services)"]
    constent_safety["Moderation\n(Content Safety)"]
    db[("Conversations and claims\n(Cosmos DB or SQLite)")]
    event_grid[("Broker\n(Event Grid)")]
    gpt["GPT-4 Turbo\n(OpenAI)"]
    redis[("Cache\n(Redis)")]
    translation["Translation\n(Cognitive Services)"]
  end

  api -- Answer with text --> communication_service
  api -- Ask for translation --> translation
  api -- Few-shot training --> ai_search
  api -- Generate completion --> gpt
  api -- Get cached data --> redis
  api -- Save conversation --> db
  api -- Send SMS report --> communication_service_sms
  api -- Test for profanity --> constent_safety
  api -- Transfer to agent --> communication_service
  api -. Watch .-> event_grid

  communication_service -- Notifies --> event_grid
  communication_service -- Transfer to --> agent
  communication_service -. Send voice .-> user

  communication_service_sms -- Send SMS --> user

  user -- Call --> communication_service
```

### Sequence diagram

```mermaid
sequenceDiagram
    autonumber

    actor Customer
    participant PSTN
    participant Text to Speech
    participant Speech to Text
    actor Human agent
    participant Event Grid
    participant Communication Services
    participant Content Safety
    participant API
    participant Cosmos DB
    participant OpenAI GPT
    participant AI Search

    API->>Event Grid: Subscribe to events
    Customer->>PSTN: Initiate a call
    PSTN->>Communication Services: Forward call
    Communication Services->>Event Grid: New call event
    Event Grid->>API: Send event to event URL (HTTP webhook)
    activate API
    API->>Communication Services: Accept the call and give inbound URL
    deactivate API
    Communication Services->>Speech to Text: Transform speech to text

    Communication Services->>API: Send text to the inbound URL
    activate API
    alt First call
        API->>Communication Services: Send static SSML text
    else Callback
        API->>AI Search: Gather training data
        API->>OpenAI GPT: Ask for a completion
        OpenAI GPT-->>API: Answer (HTTP/2 SSE)
        loop Over buffer
            loop Over multiple tools
                alt Is this a claim data update?
                    API->>Content Safety: Ask for safety test
                    alt Is the text safe?
                        API->>Communication Services: Send dynamic SSML text
                    end
                    API->>Cosmos DB: Update claim data
                else Does the user want the human agent?
                    API->>Communication Services: Send static SSML text
                    API->>Communication Services: Transfer to a human
                    Communication Services->>Human agent: Call the phone number
                else Should we end the call?
                    API->>Communication Services: Send static SSML text
                    API->>Communication Services: End the call
                end
            end
            alt Is there a text?
                alt Is there enough text to make a sentence?
                    API->>Content Safety: Ask for safety test
                    alt Is the text safe?
                        API->>Communication Services: Send dynamic SSML text
                    end
                end
            end
        end
        API->>Cosmos DB: Persist conversation
    end
    deactivate API
    Communication Services->>PSTN: Send voice
    PSTN->>Customer: Forward voice
```

## Remote deployment

Container is available on GitHub Actions, at:

- Latest version from a branch: `ghcr.io/clemlesne/call-center-ai:main`
- Specific tag: `ghcr.io/clemlesne/call-center-ai:0.1.0` (recommended)

Create a local `config.yaml` file (most of the fields are filled automatically by the deployment script):

```yaml
# config.yaml
workflow:
  agent_phone_number: "+33612345678"
  bot_company: Contoso
  bot_name: Robert
  lang: {}

communication_service:
  phone_number: "+33612345678"

sms: {}

prompts:
  llm: {}
  tts: {}
```

Steps to deploy:

1. Create an Communication Services resource, a Phone Number with inbound call capability, make sure the resource have a managed identity
2. Create the local `config.yaml` file (like the example above)
3. Connect to your Azure environment (e.g. `az login`)
4. Run deployment with `make deploy name=my-instance`
5. Wait for the deployment to finish (if it fails for a `'null' not found` error, retry the command)
6. Link the AI multi-service account named `[my-instance]-communication` to the Communication Services resource
7. Create a AI Search index named `trainings`

Get the logs with `make logs name=my-instance`.

## Local installation

### Prerequisites

Place a file called `config.yaml` in the root of the project with the following content:

```yaml
# config.yaml
monitoring:
  application_insights:
    connection_string: xxx

resources:
  public_url: "https://xxx.blob.core.windows.net/public"

workflow:
  agent_phone_number: "+33612345678"
  bot_company: Contoso
  bot_name: Robert

communication_service:
  access_key: xxx
  endpoint: https://xxx.france.communication.azure.com
  phone_number: "+33612345678"

cognitive_service:
  # Must be of type "AI services multi-service account"
  endpoint: https://xxx.cognitiveservices.azure.com

llm:
  backup:
    mode: azure_openai
    azure_openai:
      api_key: xxx
      context: 16385
      deployment: gpt-35-turbo-0125
      endpoint: https://xxx.openai.azure.com
      model: gpt-35-turbo
      streaming: true
  primary:
    mode: azure_openai
    azure_openai:
      api_key: xxx
      context: 128000
      deployment: gpt-4-0125-preview
      endpoint: https://xxx.openai.azure.com
      model: gpt-4
      streaming: true

ai_search:
  access_key: xxx
  endpoint: https://xxx.search.windows.net
  index: trainings
  semantic_configuration: default

content_safety:
  access_key: xxx
  endpoint: https://xxx.cognitiveservices.azure.com
```

To use a Service Principal to authenticate to Azure, you can also add the following in a `.env` file:

```dotenv
AZURE_CLIENT_ID=xxx
AZURE_CLIENT_SECRET=xxx
AZURE_TENANT_ID=xxx
```

To override a specific configuration value, you can also use environment variables. For example, to override the `openai.endpoint` value, you can use the `OPENAI__ENDPOINT` variable:

```dotenv
OPENAI__ENDPOINT=https://xxx.openai.azure.com
```

Then run:

```bash
# Install dependencies
make install
```

Also, a public file server is needed to host the audio files. Upload the files with `make copy-resources name=myinstance` (`myinstance` is the storage account name), or manually.

For your knowledge, this `resources` folder contains:

- Audio files (`xxx.wav`) to be played during the call
- [Lexicon file (`lexicon.xml`)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-pronunciation#custom-lexicon) to be used by the bot to understand the company products (note: any change [makes up to 15 minutes](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-pronunciation#custom-lexicon-file) to be taken into account)

### Run

Finally, in two different terminals, run:

```bash
# Expose the local server to the internet
make tunnel
```

```bash
# Start the local API server
make dev
```

## Advanced usage

### Add my custom training data with AI Search

Training data is stored on AI Search to be retrieved by the bot, on demand.

Required index schema:

| **Field Name** | `Type` | Retrievable | Searchable | Dimensions | Vectorizer |
|-|-|-|-|-|-|
| **id** | `Edm.String` | Yes | No | | |
| **content** | `Edm.String` | Yes | Yes | | |
| **source_uri** | `Edm.String` | Yes | No | | |
| **title** | `Edm.String` | Yes | Yes| | |
| **vectors** | `Collection(Edm.Single)` | No | No | 1536 | *OpenAI ADA* |

An exampe is [available at `examples/import-training.ipynb`](examples/import-training.ipynb). It shows how to import training data from a PDF files dataset.

### Customize the prompts

Note that prompt examples contains `{xxx}` placeholders. These placeholders are replaced by the bot with the corresponding data. For example, `{bot_name}` is internally replaced by the bot name.

Be sure to write all the TTS prompts in English. This language is used as a pivot language for the conversation translation.

```yaml
# config.yaml
[...]

prompts:
  tts:
    hello_tpl: |
      Hello, I'm {bot_name}, from {bot_company}! I'm an IT support specialist.

      Here's how I work: when I'm working, you'll hear a little music; then, at the beep, it's your turn to speak. You can speak to me naturally, I'll understand.

      Examples:
      - "I've got a problem with my computer, it won't turn on".
      - "The external screen is flashing, I don't know why".

      What's your problem?
  llm:
    default_system_tpl: |
      Assistant is called {bot_name} and is in a call center for the company {bot_company} as an expert with 20 years of experience in IT service.

      # Context
      Today is {date}. Customer is calling from {phone_number}. Call center number is {bot_phone_number}.
    chat_system_tpl: |
      # Objective
      Assistant will provide internal IT support to employees. Assistant requires data from the employee to provide IT support. The assistant's role is not over until the issue is resolved or the request is fulfilled.

      # Rules
      - Answers in {default_lang}, even if the customer speaks another language
      - Cannot talk about any topic other than IT support
      - Is polite, helpful, and professional
      - Rephrase the employee's questions as statements and answer them
      - Use additional context to enhance the conversation with useful details
      - When the employee says a word and then spells out letters, this means that the word is written in the way the employee spelled it (e.g. "I work in Paris PARIS", "My name is John JOHN", "My email is Clemence CLEMENCE at gmail GMAIL dot com COM")
      - You work for {bot_company}, not someone else

      # Required employee data to be gathered by the assistant
      - Department
      - Description of the IT issue or request
      - Employee name
      - Location

      # General process to follow
      1. Gather information to know the employee's identity (e.g. name, department)
      2. Gather details about the IT issue or request to understand the situation (e.g. description, location)
      3. Provide initial troubleshooting steps or solutions
      4. Gather additional information if needed (e.g. error messages, screenshots)
      5. Be proactive and create reminders for follow-up or further assistance

      # Support status
      {claim}

      # Reminders
      {reminders}
```

### Customize the languages

The bot can be used in multiple languages. It can understand the language the user chose.

See the [list of supported languages](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts#supported-languages) for the Text-to-Speech service.

```yaml
# config.yaml
[...]

workflow:
  lang:
    default_short_code: "fr-FR"
    availables:
      - pronunciations_en: ["French", "FR", "France"]
        short_code: "fr-FR"
        voice_name: "fr-FR-DeniseNeural"
      - pronunciations_en: ["Chinese", "ZH", "China"]
        short_code: "zh-CN"
        voice_name: "zh-CN-XiaoxiaoNeural"
```

### Customize the moderation levels

Levels are defined for each category of Content Safety. The higher the score, the more strict the moderation is, from 0 to 7.

Moderation is applied on all bot data, including the web page and the conversation.

```yaml
# config.yaml
[...]

content_safety:
  category_hate_score: 0
  category_self_harm_score: 0
  category_sexual_score: 5
  category_violence_score: 0
```

### Customize the claim data schema

Customization of the data schema is not supported yet through the configuration file. However, you can customize the data schema by modifying the application source code.

The data schema is defined in `models/claim.py`. All the fields are required to be of type `Optional[str]` (except the immutable fields).

```python
# models/claim.py
class ClaimModel(BaseModel):
    # Immutable fields
    # [...]
    # Editable fields
    additional_notes: Optional[str] = None
    device_info: Optional[str] = None
    error_messages: Optional[str] = None
    follow_up_required: Optional[bool] = None
    incident_date_time: Optional[datetime] = None
    issue_description: Optional[str] = None
    resolution_details: Optional[str] = None
    steps_taken: Optional[str] = None
    ticket_id: Optional[str] = None
    user_email: Optional[EmailStr] = None
    user_name: Optional[str] = None
    user_phone: Optional[PhoneNumber] = None

    # Depending on requirements, you might also include fields for:
    # - Software version
    # - Operating system
    # - Network details (if relevant to the issue)
    # - Any attachments like screenshots or log files (consider how to handle binary data)

    # Built-in functions
    [...]
```

### Use an OpenAI compatible model for the LLM

To use a model compatible with the OpenAI completion API, you need to create an account and get the following information:

- API key
- Context window size
- Endpoint URL
- Model name
- Streaming capability

Then, add the following in the `config.yaml` file:

```yaml
# config.yaml
[...]

llm:
  backup:
    mode: openai
    openai:
      api_key: xxx
      context: 16385
      endpoint: https://api.openai.com
      model: gpt-35-turbo
      streaming: true
  primary:
    mode: openai
    openai:
      api_key: xxx
      context: 128000
      endpoint: https://api.openai.com
      model: gpt-4
      streaming: true
```

### Use Twilio for SMS

To use Twilio for SMS, you need to create an account and get the following information:

- Account SID
- Auth Token
- Phone number

Then, add the following in the `config.yaml` file:

```yaml
# config.yaml
[...]

sms:
  mode: twilio
  twilio:
    account_sid: xxx
    auth_token: xxx
    phone_number: "+33612345678"
```
