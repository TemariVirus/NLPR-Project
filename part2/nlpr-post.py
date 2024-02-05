import json
import os

import boto3

BOT_ID = os.environ["BOT_ID"]
BOT_ALIAS_ID = os.environ["BOT_ALIAS_ID"]

lex = boto3.client("lexv2-runtime")


def format_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    print("event:", event)

    user_text = event["body"]
    session_id = event["queryStringParameters"]["session_id"]

    response = lex.recognize_text(
        botId=BOT_ID,
        botAliasId=BOT_ALIAS_ID,
        localeId="en_US",
        sessionId=session_id,
        text=user_text,
    )

    messages = [m["content"] for m in response["messages"]]
    return format_response(200, messages)
