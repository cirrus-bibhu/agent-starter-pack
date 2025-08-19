# app/server.py
import asyncio
import json
from fastapi import FastAPI, BackgroundTasks, HTTPException
from google.cloud import pubsub_v1
from app.agents.coordinator import CoordinatorAgent
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()
coordinator = CoordinatorAgent()

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

# Instantiate the coordinator agent at module level (singleton pattern)
coordinator_agent = CoordinatorAgent()

from typing import Dict, Any
from fastapi import HTTPException

@app.post("/api/receive-email")
def receive_email(email_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Endpoint to receive email data and invoke the Coordinator Agent as a background task.
    """
    if not email_data:
        raise HTTPException(status_code=400, detail="No email data received")

    # Add the long-running agent task to the background
    background_tasks.add_task(coordinator_agent.run, email_json=email_data)
    
    # Immediately return a response to the client
    return {"status": "success", "message": "Email received and is being processed by the agent."}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)