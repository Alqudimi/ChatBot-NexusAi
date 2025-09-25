import asyncio
import os
import json
import tempfile
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Response

# Initialize FastAPI app
app = FastAPI(title="تكنو - مساعد تعليم اللغة الإنجليزية", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نموذج Hugging Face المحلي
MODEL_NAME = "ethzanalytics/distilgpt2-tiny-conversational"
MODEL_DIR = "./distilgpt2_tiny_conversational"

# تهيئة النموذج و Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    print("✅ تم تحميل النموذج و Tokenizer بنجاح")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")
    tokenizer = None
    model = None

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
- تقديم معلومات خارج تخصصك كمعظم ل��ة إنجليزية
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

def get_huggingface_response(user_message: str, max_length: int = 300) -> str:
    """Get response from local Hugging Face model with error handling"""
    if model is None or tokenizer is None:
        return "عذراً، النموذج غير متاح حالياً. يرجى المحاولة لاحقاً."
    
    try:
        # إعداد النص مع السياق
        prompt = f"{SYSTEM_PROMPT}\n\nالمستخدم: {user_message}\nتكنو:"
        
        # ترميز النص المدخل
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # توليد الرد
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # فك ترميز الناتج
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # استخراج جزء الرد فقط (بعد آخر "تكنو:")
        if "تكنو:" in response:
            response = response.split("تكنو:")[-1].strip()
        
        # تنظيف الرد وإزالة النص الزائد
        response = response.split("\n")[0].split("المستخدم:")[0].strip()
        
        # إذا كان الرد قصيراً جداً، نستخدم رداً افتراضياً
        if len(response) < 10:
            response = "أهلاً بك! أنا دكتور تكنو، متخصص في تعليم اللغة الإنجليزية. كيف يمكنني مساعدتك اليوم؟"
        
        return response
        
    except Exception as e:
        print(f"Hugging Face model error: {e}")
        # ردود احتياطية في حالة فشل النموذج
        fallback_responses = [
            "أهلاً بك! أنا دكتور تكنو، متخصص في تعليم اللغة الإنجليزية للمبتدئين. كيف يمكنني مساعدتك في تعلم الإنجليزية اليوم؟",
            "مرحباً! أنا تكنو، مساعدك لتعلم الإنجليزية. ما السؤال الذي تريد مساعدتي فيه؟",
            "أهلاً! دكتور تكنو هنا لمساعدتك في تعلم الإنجليزية. هل لديك سؤال عن القواعد أو المحادثة الإنجليزية؟"
        ]
        return random.choice(fallback_responses)

@app.get("/")
async def read_root2():
    """Serve the main HTML page with proper encoding"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return Response(
            content=html_content,
            media_type="text/html; charset=utf-8"
        )
    except FileNotFoundError:
        # صفحة افتراضية في حالة عدم وجود الملف
        html_content = """
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>تكنو - مساعد تعليم اللغة الإنجليزية</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                h1 { color: #2c3e50; }
                .status { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>تكنو - مساعد تعليم اللغة الإنجليزية</h1>
            <p>تم تطوير هذا النظام بواسطة مازن القديمي</p>
            <p class="status">✅ الخدمة تعمل بشكل طبيعي</p>
            <p>استخدم نقاط النهاية API للتواصل مع المساعد</p>
        </body>
        </html>
        """
        return Response(content=html_content, media_type="text/html; charset=utf-8")

@app.post("/api/voice-chat", response_model=VoiceChatResponse)
async def voice_chat_endpoint(audio: UploadFile = File(...)):
    """Voice chat endpoint with Hugging Face integration"""
    try:
        # For demo purposes, simulate transcription
        transcribed_text = "مرحبا، أريد تعلم اللغة الإنجليزية"
        
        # Get AI response from Hugging Face model
        ai_response = get_huggingface_response(transcribed_text)
        
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
    """WebSocket endpoint for real-time chat with Hugging Face model"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type", "text")
            
            if message_type == "text":
                user_message = data.get("message", "").strip()
                if user_message:
                    # Get AI response from Hugging Face model
                    ai_response = get_huggingface_response(user_message)
                    
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Enhanced text chat endpoint with Hugging Face integration"""
    try:
        user_message = chat_message.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get AI response from Hugging Face model
        ai_response = get_huggingface_response(user_message)
        
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
    model_status = "healthy" if model is not None and tokenizer is not None else "unhealthy"
    
    return {
        "status": "healthy" if model_status == "healthy" else "degraded",
        "service": "تكنو - مساعد تعليم اللغة الإنجليزية",
        "version": "1.0.0",
        "model_status": model_status,
        "model_name": MODEL_NAME,
        "creator": "مازن القديمي",
        "role": "دكتور لغة إنجليزية للمبتدئين",
        "features": [
            "text_chat", 
            "voice_chat_simulation", 
            "websocket_support",
            "huggingface_local_model",
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
    """Initialize Hugging Face model on startup"""
    print("جاري التحقق من نموذج Hugging Face...")
    if model is not None and tokenizer is not None:
        print("✅ نموذج Hugging Face جاهز للاستخدام")
        
        # اختبار النموذج
        try:
            test_response = get_huggingface_response("مرحبا")
            print(f"✅ اختبار النموذج ناجح: {test_response[:50]}...")
        except Exception as e:
            print(f"⚠️  اختبار النموذج به مشكلة: {e}")
    else:
        print("❌ نموذج Hugging Face غير متوفر")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)