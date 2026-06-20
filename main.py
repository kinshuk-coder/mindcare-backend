from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import uuid
from pymongo import MongoClient
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
import requests



CRISIS_TRIGGERS = [
    "suicide", "kill myself", "end it all", "want to die", 
    "hurt myself", "self harm", "no reason to live", "better off dead","end my life"
]

#function to detect crisis 
def detect_crisis(user_input):
    text = user_input.lower()
    for TRIGGER in CRISIS_TRIGGERS:
        if TRIGGER in text:
            return True
    return False    

#api key is added through load_dotenv()
load_dotenv()

#intiailizing AsyncOpenAI object 
client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1")

#initializing mongo db client
mc = MongoClient(os.getenv("MONGODB_URI"))
history_collection = mc["mindcare_db"]
history = history_collection["chat_sessions"]


#initializing pineconde client
pc = Pinecone(os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("mindcare-memory")


#initializing the vector embedder
def vector_embedder(s : str):
    hf_api_key = os.getenv("HF_API_KEY")
    api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    headers = {"Authorization":f"Bearer {hf_api_key}"}

    response = requests.post(api_url,headers=headers,json={"input":s})

    if(response.status_code==200):
        return response.json()
    else:
        raise Exception(f"Failed to connect to hugging face:{response.text}")

#intializing FastAPI object
app = FastAPI(title="MindCare API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_headers = ["*"],
    allow_methods = ["*"],

)


#schema of both user and Ai using pydantic BaseModel
class UserQuery(BaseModel):
    user_message : str
    session_id : str = "default_session"

class AiResponse(BaseModel):
    ai_message : str = Field(description="the reply ")
    intent : str = Field(description="What is the intent of user?")
    confidence_score : float = Field(description="Confidence Score between 0.0 and 1.0")



#FastAPI post endpoint
@app.post("/chat")
async def chat_endpoint(user_query : UserQuery):

    
    try:
        session = user_query.session_id

        doc = history.find_one({"session_id":session})
        if(doc): session_history = doc["messages"]
        else: session_history = []

        last_10_messages = session_history[-10:]
        
        short_term_history = " | ".join([f"{msg['role']}: {msg['content']}" for msg in last_10_messages])

        if(detect_crisis(user_query.user_message)):
            emergency_reply = "I am an AI, and it sounds like you are going through an incredibly difficult time right now. Your safety is the most important thing. Please reach out to a human who can help. In India, you can call the Kiran Helpline at 1800-599-0019 or AASRA at 9820466726. You do not have to go through this alone."

            interaction = f"user:{user_query.user_message} | assistant:{emergency_reply}"
            
            session_history.append({"role": "user","content":user_query.user_message})
            session_history.append({"role":"assistant","content":emergency_reply,"intent":"CRISIS_EMERGENCY","confidence_score":1.00})

            
            history.update_one({
                "session_id":session},
                {"$set":{"messages":session_history}},
                upsert = True
            )

            #return the correct schema
            return {
                "ai_message" : emergency_reply,
                "intent" : "CRISIS_EMERGENCY",
                "confidence_score" : 1.00

            }
        
        query_vector = vector_embedder(user_query.user_message)
        
        res = pinecone_index.query(vector=query_vector,
                                    top_k = 10,
                                    filter = {"session_id":session},
                                    include_metadata = True)
        

        #adding the related messags to the variable chat_history
        chat_history = ""
        
        for match in res["matches"]:
            chat_history += match["metadata"]["text"] + ' | '

        #calling client.beta.chat.completions.parse method from pydantic to force the AI to generate the response in strict json schema
        #using await so that other processes does not stop till the response is recieved
        completion = await client.beta.chat.completions.parse(
            model ="openai/gpt-oss-120b",
            messages = [ {"role":"system","content": f"You are an AI mental health support assistant, if you do not know the exact answer just say 'I don't know' instead of guessing the answer. These are the previous chat history regarding the current user message, these contains the messages that are semantically related to current message of user, it is provided to you so that you can understand the context of the user's current message : {chat_history} | \n\n These are the last few messages of the user, also use these to understand the context of the recent conversation: {short_term_history}"
            },{"role" : "user","content":user_query.user_message} ],
            temperature = 0.2,
            response_format = AiResponse
        )
        #parse the json file to python dictionary
        ai_data = completion.choices[0].message.parsed

        #save and add the interaction to the database for short term as well as long term purposes
        interaction = f"user:{user_query.user_message} | assistant:{ai_data.ai_message}"
        interaction_vector = vector_embedder(interaction)

        pinecone_index.upsert(vectors=[
            {"id":str(uuid.uuid4()),
                "values": interaction_vector,
                "metadata": {"session_id":session,"text":interaction}
            }
        ])

        session_history.append({"role": "user","content":user_query.user_message})
        session_history.append({"role":"assistant","content":ai_data.ai_message,"intent":ai_data.intent,"confidence_score":ai_data.confidence_score})

        history.update_one(
                {"session_id":session},
                {"$set":{"messages":session_history}},
                upsert = True
            )

        #return the correct schema
        return {
            "ai_message" : ai_data.ai_message,
            "intent" : ai_data.intent,
            "confidence_score" : ai_data.confidence_score
        }
    except Exception as e:
        print(f"Critical Error:{str(e)}")
        raise HTTPException(status_code=500,detail=f"Ai processing error:{str(e)}")
    
    

@app.get("/history/{session_id}")
def get_chat_history(session_id: str):
    
    doc = history.find_one({"session_id":session_id})
    if(doc): return doc["messages"]
    else: return []
    

