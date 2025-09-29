import asyncio
import json
import logging
import os
import random
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    quality: str = "high"
    speed: float = 1.0
    pitch: float = 1.0

class TTSResponse(BaseModel):
    success: bool
    audio_url: str
    file_name: str
    duration: str = None
    file_size: str = None
    message: str = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    audio_files_count: int

# إعدادات التطبيق
AUDIO_DIR = "audio_files"
MAX_TEXT_LENGTH = 5000
SUPPORTED_LANGUAGES = ['en']
os.makedirs(AUDIO_DIR, exist_ok=True)
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

async def periodic_audio_cleanup():
    
    while True:
        try:
            await asyncio.sleep(0.5 * 60 * 60)  
            
            logger.info("بدء المسح الدوري للملفات الصوتية...")
            
            current_time = datetime.now()
            deleted_files = 0
            
            # مسح جميع الملفات الصوتية في المجلد
            for file_path in Path(AUDIO_DIR).glob("*.mp3"):
                try:
                    os.remove(file_path)
                    deleted_files += 1
                    logger.info(f"تم حذف الملف: {file_path.name}")
                except Exception as e:
                    logger.error(f"خطأ في حذف الملف {file_path.name}: {e}")
            
            logger.info(f"اكتمل المسح الدوري. تم حذف {deleted_files} ملف")
            
        except Exception as e:
            logger.error(f"خطأ في المهمة الدورية: {e}")
            await asyncio.sleep(60)

def remove_markdown_formatting(text: str) -> str:
    """
    إزالة تنسيق Markdown من النص وإرجاع نص عادي
    """
    if not text:
        return text
    
    # إزالة النجمة المزدوجة **text** → text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # إزالة النجمة المفردة *text* → text
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # إزالة التسطير __text__ → text
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # إزالة التسطير _text_ → text
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # إزالة الرموز الأخرى مثل ~~text~~ → text
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    
    # إزالة الرموز الخاصة الأخرى
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # تنظيف المسافات الزائدة الناتجة عن الإزالة
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def contains_english_text(text: str) -> bool:
    """التحقق من أن النص يحتوي على أحرف إنجليزية"""
    import re
    # البحث عن أي حرف إنجليزي (a-z, A-Z)
    english_pattern = re.compile(r'[a-zA-Z]')
    return bool(english_pattern.search(text))

def format_file_size(size_bytes: int) -> str:
    """تنسيق حجم الملف إلى صيغة مقروءة"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
        
    return f"{size_bytes:.2f} {size_names[i]}"

def estimate_audio_duration(text: str, speed: float = 1.0) -> str:
    """تقدير مدة الملف الصوتي (تقريبي)"""
    # متوسط سرعة الكلام: 150 كلمة في الدقيقة
    words = len(text.split())
    base_duration_seconds = (words / 150) * 60  # تحويل إلى ثواني
    
    # تعديل المدة بناءً على السرعة
    adjusted_duration = base_duration_seconds / speed
    
    # تنسيق المدة
    minutes = int(adjusted_duration // 60)
    seconds = int(adjusted_duration % 60)
    
    if minutes > 0:
        return f"{minutes}:{seconds:02d}"
    else:
        return f"0:{seconds:02d}"

def cleanup_old_files(hours_old: int = 24):
    """تنظيف الملفات القديمة (اختياري)"""
    try:
        current_time = datetime.now().timestamp()
        for file_path in Path(AUDIO_DIR).glob("*.mp3"):
            file_age = current_time - file_path.stat().st_mtime
            if file_age > hours_old * 3600:  # تحويل الساعات إلى ثواني
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def get_server_uptime() -> str:
    """الحصول على مدة تشغيل الخادم (مبسط)"""
    # في تطبيق حقيقي، يمكنك استخدام psutil أو حفظ وقت البدء
    return "غير متاح في وضع التطوير"

# معالجة الأخطاء العامة
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "حدث خطأ داخلي في الخادم"}
    )


def get_gemini_response(user_message: str, conversation) -> str:
    """Get response from Gemini API with error handling"""
    try:
        # Prepare the full message with context
        full_message = f"{SYSTEM_PROMPT}\n\nرسالة المستخدم: {user_message}"
        
        response = conversation.send_message(full_message)
        response_text = response.text
        response_text = response.text
        print("النص الأصلي من Gemini:", response_text)
        
        # إزالة تنسيق Markdown
        clean_text = remove_markdown_formatting(response_text)
        print("النص بعد التنظيف:", clean_text)
        
        return clean_text
        
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
    
    
@app.get("/text-to-speec")
async def text_to_speec():
    
    with open("text-to-speec.html", "r", encoding="utf-8") as f:
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

@app.post("/api/tts", response_model=TTSResponse)
async def convert_text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """تحويل النص الإنجليزي إلى صوت"""
    
    # التحقق من صحة النص
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="النص لا يمكن أن يكون فارغاً")
    
    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"النص طويل جداً. الحد الأقصى هو {MAX_TEXT_LENGTH} حرف"
        )
    
    # التحقق من أن النص باللغة الإنجليزية
    if not contains_english_text(request.text):
        raise HTTPException(
            status_code=400, 
            detail="يجب أن يحتوي النص على أحرف إنجليزية"
        )
    
    try:
        # إنشاء اسم فريد للملف
        file_id = str(uuid.uuid4())
        filename = f"tts_{file_id}.mp3"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # إعداد معلمات gTTS
        tts_params = {
            'text': request.text,
            'lang': 'en',
            'slow': False
        }
        
        # ضبط السرعة (إذا كانت مدعومة)
        # ملاحظة: gTTS لا يدعم ضبط السرعة مباشرة، لكننا نضبطها في الواجهة
        if request.speed < 0.8:
            tts_params['slow'] = True
        
        logger.info(f"Converting text to speech: {request.text[:100]}...")
        
        # إنشاء كائن gTTS وتحويل النص إلى صوت
        tts = gTTS(**tts_params)
        
        # حفظ الملف الصوتي
        tts.save(file_path)
        
        # الحصول على معلومات الملف
        file_size = os.path.getsize(file_path)
        file_size_str = format_file_size(file_size)
        
        # تقدير المدة (تقريبي)
        duration_estimate = estimate_audio_duration(request.text, request.speed)
        
        # إنشاء رابط للوصول إلى الملف
        audio_url = f"/audio/{filename}"
        
        # جدولة تنظيف الملف بعد فترة (اختياري)
        # background_tasks.add_task(cleanup_old_files)
        
        logger.info(f"Successfully converted text to speech: {filename}")
        
        return TTSResponse(
            success=True,
            audio_url=audio_url,
            file_name=filename,
            duration=duration_estimate,
            file_size=file_size_str,
            message="تم تحويل النص إلى صوت بنجاح"
        )
        
    except Exception as e:
        logger.error(f"Error converting text to speech: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"حدث خطأ أثناء تحويل النص إلى صوت: {str(e)}"
        )

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """خدمة للحصول على الملفات الصوتية"""
    
    # التحقق من صحة اسم الملف
    if not filename.endswith('.mp3'):
        raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    
    file_path = os.path.join(AUDIO_DIR, filename)
    
    # التحقق من وجود الملف
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    
    # التحقق من أن الملف ضمن المجلد المسموح به
    print(file_path)
    
    return FileResponse(
        path=file_path,
        media_type='audio/mpeg',
        filename=f"converted_{filename}"
    )

@app.delete("/api/audio/{filename}")
async def delete_audio_file(filename: str):
    """حذف ملف صوتي"""
    
    file_path = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    
    try:
        os.remove(file_path)
        logger.info(f"Deleted audio file: {filename}")
        return {"success": True, "message": "تم حذف الملف بنجاح"}
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء حذف الملف")


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
    asyncio.create_task(periodic_audio_cleanup())
    logger.info("تم بدء مهمة المسح الدوري للملفات الصوتية (كل 12 ساعة)")
    if model:
        print("✅ تم تهيئة نموذج Gemini بنجاح")
    else:
        print("❌ فشل في تهيئة نموذج Gemini")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
