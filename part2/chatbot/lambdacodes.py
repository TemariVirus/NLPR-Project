import base64
import hashlib
import os
import re
from datetime import datetime

import boto3
import openai
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from llama_index import (
    PromptHelper,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llm_predictor import LLMPredictor


def should_fulfill(event: dict[str, any]) -> bool:
    return event["invocationSource"] == "FulfillmentCodeHook"


def get_intent_name(event: dict[str, any]) -> str:
    return event["sessionState"]["intent"]["name"]


def format_response(
    event: dict[str, any], session_state: dict[str, any], messages: [str] = []
) -> dict[str, any]:
    return {
        "sessionState": session_state,
        "sessionId": event["sessionId"],
        "messages": [
            {"contentType": "PlainText", "content": message} for message in messages
        ],
        "requestAttributes": event.get("requestAttributes", {}),
    }


def fulfill(event: dict[str, any], messages: [str] = []) -> dict[str, any]:
    session_state = event["sessionState"]
    session_state["dialogAction"] = {"type": "Close"}
    session_state["intent"]["state"] = "Fulfilled"
    return format_response(event, session_state, messages)


def delegate(event: dict[str, any]) -> dict[str, any]:
    session_state = event["sessionState"]
    session_state["dialogAction"] = {"type": "Delegate"}
    return format_response(event, session_state)


def elicit_slot(
    event: dict[str, any], slot_name: str, messages: [str] = []
) -> dict[str, any]:
    session_state = event["sessionState"]
    session_state["dialogAction"] = {"type": "ElicitSlot", "slotToElicit": slot_name}
    session_state["intent"]["state"] = "InProgress"
    return format_response(event, session_state, messages)


def set_slot_value(slots, slot_name, value):
    slots[slot_name] = {
        "value": {
            "originalValue": value,
            "interpretedValue": value,
            "resolvedValues": [value],
        }
    }


def is_valid_username(username: str) -> bool:
    return re.match(r"^[a-zA-Z ]+$", username)


def is_valid_password(password: str) -> bool:
    return re.match(r"^[a-zA-Z0-9!@#$%^&*()_+]{8,}$", password)


def get_user(username: str) -> bool:
    response = user_table.get_item(Key={"username": username})
    return response.get("Item", {})


def hash_password(password: str) -> str:
    return base64.b64encode(
        hashlib.pbkdf2_hmac("sha512", password.encode(), b"", 100_000)
    ).decode("utf-8")


def check_password(username: str, hashed_password: str) -> bool:
    response = user_table.get_item(Key={"username": username})
    return response.get("Item", {}).get("password", "") == hashed_password


def parse_history(history: str) -> [str]:
    return [h.strip() for h in history.split(",") if h.strip() != ""]


def pack_symptom(symptom: str) -> dict[str, any]:
    return {
        "time": int(round(datetime.now().timestamp())),
        "symptom": symptom,
    }


def unpack_symptoms(symptoms: [dict[str, any]]) -> [str]:
    return "\n".join(
        [
            f"    {datetime.fromtimestamp(int(s['time'])).strftime('%Y-%m-%d %H:%M:%S')}: {s['symptom']}"
            for s in symptoms
        ]
    )


def fulfill_get_symptoms(event: dict[str, any]) -> dict[str, any]:
    slots = event["sessionState"]["intent"]["slots"]

    username = slots["name"]["value"]["originalValue"]
    user = get_user(username)

    gender = user["gender"]
    age = user["age"]
    history = "\n".join(["    " + s for s in user["history"]])
    symptoms = unpack_symptoms(user["symptoms"])

    return fulfill(
        event,
        [
            "Here is your information:",
            f"Name: {username}",
            f"Gender: {gender}",
            f"Age: {age}",
            f"History:\n{history if history else '    No known medial history'}",
            f"Past symptoms:\n{symptoms}",
        ],
    )


def get_symptoms(event: dict[str, any]) -> dict[str, any]:
    if should_fulfill(event):
        return fulfill_get_symptoms(event)

    slots = event["sessionState"]["intent"]["slots"]
    attributes = event["sessionState"]["sessionAttributes"]
    user = None

    # Validate name
    if slots.get("name", ""):
        username = slots["name"]["value"]["originalValue"]
        if not is_valid_username(username):
            return elicit_slot(event, "name", ["Please enter a valid name."])
        user = get_user(username)
        if not user:
            return fulfill(event, ["You have not recorded any symptoms yet."])
        else:
            attributes["name"] = username
            if not slots.get("password", ""):
                # Delegate does not display messages, thus elict_slot is used here
                return elicit_slot(
                    event,
                    "password",
                    [
                        f"Welcome back, {username}!",
                        "Please enter your password.",
                    ],
                )
    elif attributes.get("name", ""):
        # Fill in name if possible
        set_slot_value(slots, "name", attributes["name"])

    # Validate password
    if slots.get("password", ""):
        password = slots["password"]["value"]["originalValue"]
        if not is_valid_password(password):
            return elicit_slot(
                event,
                "password",
                ["A password must be at least 8 characters long. Please try again."],
            )

        hashed = hash_password(password)
        if user and not check_password(username, hashed):
            return elicit_slot(
                event,
                "password",
                ["The password you entered is incorrect. Please try again."],
            )

        attributes["password"] = password
    elif attributes.get("password", ""):
        # Fill in password if possible
        set_slot_value(slots, "password", attributes["password"])

    return delegate(event)


def fulfill_post_symptoms(event: dict[str, any]) -> dict[str, any]:
    slots = event["sessionState"]["intent"]["slots"]

    username = slots["name"]["value"]["originalValue"]
    password = slots["password"]["value"]["originalValue"]
    gender = slots["gender"]["value"]["interpretedValue"]
    age = slots["age"]["value"]["interpretedValue"]
    history = parse_history(slots["history"]["value"]["originalValue"])
    symptom = pack_symptom(slots["symptoms"]["value"]["originalValue"])

    if get_user(username):
        user_table.update_item(
            Key={"username": username},
            UpdateExpression="SET symptoms = list_append(symptoms, :s)",
            ExpressionAttributeValues={":s": [symptom]},
        )
    else:
        user_table.put_item(
            Item={
                "username": username,
                "password": hash_password(password),
                "gender": gender,
                "age": age,
                "history": history,
                "symptoms": [symptom],
            }
        )
    return fulfill(event)


def post_symptoms(event: dict[str, any]) -> dict[str, any]:
    if should_fulfill(event):
        return fulfill_post_symptoms(event)

    slots = event["sessionState"]["intent"]["slots"]
    attributes = event["sessionState"]["sessionAttributes"]
    user = None

    # Validate name
    if slots.get("name", ""):
        username = slots["name"]["value"]["originalValue"]
        if not is_valid_username(username):
            return elicit_slot(event, "name", ["Please enter a valid name."])
        user = get_user(username)
        attributes["name"] = username

        if not slots.get("password", ""):
            # Delegate does not display messages, thus elict_slot is used here
            return elicit_slot(
                event,
                "password",
                (
                    [
                        f"Welcome back, {username}!",
                        "Please enter your password.",
                    ]
                    if user
                    else [
                        f"Hello, {username}! It looks like your using our system for the first time, so we'll need some additional information.",
                        "Please set your password below.",
                    ]
                ),
            )
    elif attributes.get("name", ""):
        # Fill in name if possible
        set_slot_value(slots, "name", attributes["name"])

    # Validate password
    if slots.get("password", ""):
        password = slots["password"]["value"]["originalValue"]
        if not is_valid_password(password):
            return elicit_slot(
                event,
                "password",
                ["A password must be at least 8 characters long. Please try again."],
            )

        hashed = hash_password(password)
        if user and not check_password(username, hashed):
            return elicit_slot(
                event,
                "password",
                ["The password you entered is incorrect. Please try again."],
            )

        attributes["password"] = password
    elif attributes.get("password", ""):
        # Fill in password if possible
        set_slot_value(slots, "password", attributes["password"])

    # Fill in gender if possible
    if not slots.get("gender", "") and user:
        set_slot_value(slots, "gender", user["gender"])

    # Validate age
    if slots.get("age", ""):
        age = slots["age"]["value"]["originalValue"]
        try:
            age = int(age)
        except:
            return elicit_slot(
                event, "age", ["Please enter a whole number for your age."]
            )
        if age < 0:
            return elicit_slot(event, "age", ["Please enter a positive age."])
        if age > 200:
            return elicit_slot(event, "age", ["Please enter a realistic age."])
    elif user:
        # Fill in age if possible
        set_slot_value(slots, "age", user["age"])

    # Fill in history_exists if possible
    history_exists = None
    if slots.get("history_exists", ""):
        history_exists = slots["history_exists"]["value"]["interpretedValue"]
    elif user:
        history_exists = "yes"
        set_slot_value(slots, "history_exists", history_exists)

    # Fill in history if possible
    if not slots.get("history", ""):
        if user:
            set_slot_value(slots, "history", ",".join(user["history"]))
        elif history_exists == "no":
            # Fill with empty array if no history exists
            set_slot_value(slots, "history", "")

    return delegate(event)


def prepare_prompt(name: str, password: str, prompt: str) -> str:
    prompt = (
        'You are a healthcare expert system called "Health Buddy", created to answer my questions about the new context or about myself. Here is my question: '
        + prompt.strip()
    )
    unauthorized_prompt = (
        "If you require my personal information, you should request that I log in. "
        + prompt
    )

    if not name or not password:
        return unauthorized_prompt

    user = get_user(name)
    if not user:
        return unauthorized_prompt

    hashed = hash_password(password)
    if user.get("password", "") != hashed:
        return unauthorized_prompt

    # Inject user info into prompt if they are authenticated
    gender = user["gender"]
    age = user["age"]
    history = ",".join([s for s in user["history"]]) if user["history"] else "none"
    symptoms = (
        ",".join([s["symptom"] for s in user["symptoms"]])
        if user["symptoms"]
        else "none"
    )
    return f"Hello, I am {name}. My gender is {gender} and I am {age} years old. This is my medical history: ({history}). These are the symptoms I am experiencing: ({symptoms}). {prompt}"


def fallback(event: dict[str, any]) -> dict[str, any]:
    attributes = event["sessionState"]["sessionAttributes"]
    name = attributes.get("name", None)
    password = attributes.get("password", None)
    prompt = prepare_prompt(name, password, event["inputTranscript"])

    # Fallback to GPT-3.5 to answer queries
    response = query_engine.query(prompt).response
    return fulfill(event, [response])


def init_llm():
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("OPENAI_API_VERSION")
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Hyperparameters
    TEMPERATURE = 0.9
    MAX_OUTPUT_LEN = 1024
    CONTEXT_WINDOW = 4096
    CHUNK_OVERLAP_RATIO = 0.1
    LLM_NAME = "gpt-35-turbo"  # Fixed to deployed model
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Fixed to deployed model

    # Create model and embeddings
    llm = LLMPredictor(
        llm=AzureChatOpenAI(
            model=LLM_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_LEN,
        )
    )
    embed_model = LangchainEmbedding(AzureOpenAIEmbeddings(model=EMBEDDING_MODEL))
    prompt_helper = PromptHelper(CONTEXT_WINDOW, MAX_OUTPUT_LEN, CHUNK_OVERLAP_RATIO)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm, embed_model=embed_model, prompt_helper=prompt_helper
    )

    # Load index
    global query_engine
    storage_context = StorageContext.from_defaults(persist_dir="./gpt_store")
    llm_index = load_index_from_storage(
        storage_context, service_context=service_context
    )
    query_engine = llm_index.as_query_engine()


def init_db():
    global user_table
    dynamodb = boto3.resource("dynamodb")
    user_table = dynamodb.Table(os.getenv("USER_TABLE"))


def main(event: dict[str, any], _) -> dict[str, any]:
    print("event:", event)

    intent = get_intent_name(event)
    match intent:
        case "getSymptoms":
            response = get_symptoms(event)
        case "postSymptoms":
            response = post_symptoms(event)
        case "FallbackIntent":
            response = fallback(event)
        case _:
            raise ValueError(f"Unknown intent: {intent}")

    print("response:", response)
    return response


init_llm()
init_db()
