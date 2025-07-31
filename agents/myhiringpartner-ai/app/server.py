# app/server.py
import asyncio
import json
from fastapi import FastAPI, BackgroundTasks
from google.cloud import pubsub_v1
from app.agents.coordinator import CoordinatorAgent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
coordinator = CoordinatorAgent()

# Pub/Sub subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(
    "saas-agent-d", "hiring-emails-subscription"
)

@app.post("/process-pubsub-message")
async def process_pubsub_message(message_data: dict, background_tasks: BackgroundTasks):
    """Process Pub/Sub message containing email data"""
    
    # Decode message data
    email_data = json.loads(message_data['data'])
    
    # Add background task to process email
    background_tasks.add_task(process_email_async, email_data)
    
    return {"status": "received", "message_id": email_data.get('message_id')}

async def process_email_async(email_data: dict):
    """Async task to process email through coordinator"""
    try:
        result = await coordinator.route_email(email_data)
        print(f"Processing result: {result}")
    except Exception as e:
        print(f"Error processing email: {e}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)