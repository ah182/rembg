import os
import io
from fastapi import FastAPI, UploadFile, File, Response
from rembg import remove, new_session
from PIL import Image, ImageEnhance

# -------------------------------
# إعداد البيئة والكاش
# -------------------------------
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.makedirs("/tmp/numba_cache", exist_ok=True)
os.environ["NUMBA_DISABLE_JIT"] = "1"  # تعطيل الـ JIT لتجنب RuntimeError

# أماكن حفظ موديلات rembg
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["U2NET_HOME"] = "/app/.u2net"
os.makedirs("/app/.cache/huggingface", exist_ok=True)
os.makedirs("/app/.u2net", exist_ok=True)

# -------------------------------
# تهيئة التطبيق والموديل
# -------------------------------
app = FastAPI(title="Enhanced Background Remover API 🚀")

# تحميل الموديل مرة واحدة
session = new_session("u2netp")

# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server running successfully 🚀"}

# -------------------------------
# API endpoint: تحسين الصورة ثم إزالة الخلفية
# -------------------------------
@app.post("/api/remove")
async def enhance_then_remove_bg(file: UploadFile = File(...)):
    # قراءة الصورة
    image_data = await file.read()

    # فتح الصورة وتحسينها أولاً
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    # تحسينات تشبه الماسح الضوئي (قبل إزالة الخلفية)
    img = ImageEnhance.Sharpness(img).enhance(2.0)      # وضوح أعلى
    img = ImageEnhance.Contrast(img).enhance(1.4)       # تباين أقوى
    img = ImageEnhance.Brightness(img).enhance(1.15)    # إضاءة محسّنة

    # تحويل الصورة بعد التحسين إلى bytes
    enhanced_bytes = io.BytesIO()
    img.save(enhanced_bytes, format="PNG")
    enhanced_bytes.seek(0)

    # إزالة الخلفية من الصورة المحسنة
    result = remove(enhanced_bytes.getvalue(), session=session)

    # إرجاع الصورة الناتجة
    return Response(content=result, media_type="image/png")
