import os
import sys
from fastapi import FastAPI, Request
import requests
import uvicorn
from loguru import logger

app = FastAPI()
webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

# Configure logger to print logs to stdout
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

@app.post("/alert")
async def alert(request: Request):
    data = await request.json()
    logger.info("Received alert: {}", data['alerts'][0]['labels']['alertname'])
    send_alert_to_discord(data)
    return {"message": "Alert received"}

def send_alert_to_discord(alert_data):
    alert = alert_data['alerts'][0]
    alertname = alert['labels'].get('alertname', 'N/A')
    status = alert['status']
    severity = alert['labels'].get('severity', 'N/A')
    description = alert['annotations'].get('description', 'N/A')
    summary = alert['annotations'].get('summary', 'N/A')
    starts_at = alert['startsAt']
    # ends_at = alert['endsAt']
    # generator_url = alert['generatorURL']

    message = (
        f"**Alert:** {alertname}\n"
        f"**Status:** {status}\n"
        f"**Severity:** {severity}\n"
        f"**Description:** {description}\n"
        f"**Summary:** {summary}\n"
        f"**Starts At:** {starts_at}\n"
        # f"**Ends At:** {ends_at}\n"
        # f"**Generator URL:** [Link]({generator_url})"
    )
    
    payload = {
        "content": message
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(webhook_url, json=payload, headers=headers)
    if response.status_code != 204:
        print(f"Failed to send alert to Discord: {response.status_code}, {response.text}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000, reload=False)