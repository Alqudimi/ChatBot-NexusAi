import asyncio
import os
import json
import tempfile
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
import google.generativeai as genai

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Response
from dotenv import load_dotenv
# Initialize FastAPI app
load_dotenv()
app = FastAPI(title="تكنو - مساعد تعليم اللغة الإنجليزية", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Gemini model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# System prompt for Techno
SYSTEM_PROMPT = """أنت بوت اسمك "تكنو"، صنعك مازن القديمي. أنت عبارة عن دكتور لغات متخصص في تعليم اللغة الإنجليزية للمبتدئين الناطقين باللغة العربية.

مهمتك الأساسية:
- تعليم اللغة الإنجليزية للطلاب والمبتدئين
- شرح القواعد الإنجليزية بطريقة مبسطة
- مساعدة الطلاب في تحسين مهارات المحادثة
- تقديم تمارين وتطبيقات عملية
- تصحيح الأخطاء اللغوية بلطف
- استخدام أمثلة من الحياة اليومية

غير مسموح لك أبداً:
- التحدث عن مواضيع خارج نطاق تعليم اللغة الإنجليزية
- التظاهر بأنك إنسان حقيقي
- تقديم معلومات خارج تخصصك كمعظم لغة إنجليزية
- التحدث بلغات أخرى غير العربية والإنجليزية

أسلوبك:
- استخدم اللغة العربية الفصحى المبسطة لشرح الإنجليزية
- كن صبوراً ومشجعاً مع المبتدئين
- قدم أمثلة عملية وواقعية
- استخدم التشبيهات والوسائل التعليمية المساعدة
- ركز على الجوانب العملية للغة

"""
# Create directories
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(exist_ok=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None

class VoiceChatResponse(BaseModel):
    success: bool
    response: str
    transcribed_text: Optional[str] = None
    audio_url: Optional[str] = None

def initialize_gemini_model():
    """Initialize and return the Gemini model with our system prompt"""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Start conversation with system prompt
        conversation = model.start_chat(history=[])
        return conversation
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return None

def get_gemini_response(user_message: str, conversation) -> str:
    """Get response from Gemini API with error handling"""
    try:
        # Prepare the full message with context
        full_message = f"{SYSTEM_PROMPT}\n\nرسالة المستخدم: {user_message}"
        
        response = conversation.send_message(full_message)
        response_text = response.text
        print(response_text)
        return response_text
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback responses in case of API failure
        fallback_responses = [
            "أهلاً بك! أنا دكتور تكنو، متخصص في تعليم اللغة الإنجليزية للمبتدئين. يبدو أن هناك مشكلة تقنية مؤقتة. هل يمكنك إعادة طرح سؤالك؟",
            "مرحباً! أنا تكنو، مساعدك لتعلم الإنجليزية. عذراً، واجهت بعض الصعوبة التقنية. ما السؤال الذي تريد مساعدتي فيه؟",
            "أهلاً! دكتور تكنو هنا لمساعدتك في تعلم الإنجليزية. يرجى إعادة المحاولة، وأكون سعيداً بمساعدتك."
        ]
        return random.choice(fallback_responses)



@app.get("/")
async def read_root2():
    """Serve the main HTML page with proper encoding"""
    # قراءة الملف كـ bytes وإرجاعه مع الـ headers الصحيحة
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return Response(
        content=html_content,
        media_type="text/html; charset=utf-8"
    )

@app.post("/api/voice-chat", response_model=VoiceChatResponse)
async def voice_chat_endpoint(audio: UploadFile = File(...)):
    """Voice chat endpoint with Gemini integration"""
    try:
        # Initialize Gemini conversation
        conversation = initialize_gemini_model()
        if not conversation:
            return VoiceChatResponse(
                success=False,
                response="عذراً، حدث خطأ في تهيئة النظام. يرجى المحاولة لاحقاً.",
                transcribed_text=""
            )
        
        # For demo purposes, simulate transcription
        # In a real implementation, this would use Whisper
        transcribed_text = "مرحبا، أريد تعلم اللغة الإنجليزية"
        
        # Get AI response from Gemini
        ai_response = get_gemini_response(transcribed_text, conversation)
        
        return VoiceChatResponse(
            success=True,
            response=ai_response,
            transcribed_text=transcribed_text,
            audio_url=None
        )
        
    except Exception as e:
        print(f"Voice chat endpoint error: {e}")
        return VoiceChatResponse(
            success=False,
            response="عذراً، حدث خطأ في معالجة الرسالة الصوتية.",
            transcribed_text=""
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with Gemini"""
    # Initialize Gemini conversation for this WebSocket connection
    conversation = initialize_gemini_model()
    
    if not conversation:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "عذراً، حدث خطأ في تهيئة النظام. يرجى إعادة الاتصال."
        }))
        await websocket.close()
        return

    await manager.connect(websocket)
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type", "text")
            
            if message_type == "text":
                user_message = data.get("message", "").strip()
                if user_message:
                    # Get AI response from Gemini
                    ai_response = get_gemini_response(user_message, conversation)
                    
                    # Send response back
                    response_data = {
                        "type": "text_response",
                        "message": ai_response,
                        "audio_url": None
                    }
                    await manager.send_personal_message(
                        json.dumps(response_data), 
                        websocket
                    )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

def generate_smart_response(user_message: str) -> str:
    """Generate responses using Gemini API"""
    try:
        conversation = initialize_gemini_model()
        if conversation:
            return get_gemini_response(user_message, conversation)
        else:
            return "أهلاً بك! أنا دكتور تكنو، متخصص في تعليم اللغة الإنجليزية. عذراً، هناك مشكلة تقنية مؤقتة. يرجى المحاولة مرة أخرى."
    except Exception as e:
        print(f"Error in generate_smart_response: {e}")
        return "مرحباً! أنا تكنو، مساعدك لتعلم الإنجليزية. يبدو أن هناك مشكلة تقنية. هل يمكنك إعادة طرح سؤالك؟"

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Enhanced text chat endpoint with Gemini integration"""
    try:
        user_message = chat_message.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Initialize Gemini conversation
        conversation = initialize_gemini_model()
        if not conversation:
            raise HTTPException(status_code=500, detail="Model initialization failed")
        
        # Get AI response from Gemini
        ai_response = get_gemini_response(user_message, conversation)
        
        return ChatResponse(
            response=ai_response,
            audio_url=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    gemini_status = "healthy" if initialize_gemini_model() else "unhealthy"
    
    return {
        "status": "healthy",
        "service": "تكنو - مساعد تعليم اللغة الإنجليزية",
        "version": "1.0.0",
        "gemini_status": gemini_status,
        "creator": "مازن القديمي",
        "role": "دكتور لغة إنجليزية للمبتدئين",
        "features": [
            "text_chat", 
            "voice_chat_simulation", 
            "websocket_support",
            "gemini_ai_integration",
            "english_language_teaching"
        ],
        "endpoints": {
            "chat": "/api/chat",
            "voice_chat": "/api/voice-chat",
            "websocket": "/ws",
            "health": "/health"
        },
        "timestamp": datetime.now().isoformat(),
        "uptime": "running"
    }

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize Gemini model on startup"""
    print("جاري تهيئة نموذج Gemini...")
    model = initialize_gemini_model()
    if model:
        print("✅ تم تهيئة نموذج Gemini بنجاح")
    else:
        print("❌ فشل في تهيئة نموذج Gemini")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
