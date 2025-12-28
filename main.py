import os
import io
from fastapi import FastAPI, UploadFile, File, Response
from rembg import remove, new_session
from PIL import Image, ImageEnhance

# ---------------------------------------------------------
# 1. إعداد المسارات العالمية (للتوافق مع سيرفرات Render)
# ---------------------------------------------------------
# نستخدم /tmp لأنه المجلد الوحيد المضمون فيه صلاحيات الكتابة (Write Access)
HOME_DIR = "/tmp"

os.environ["NUMBA_CACHE_DIR"] = os.path.join(HOME_DIR, "numba_cache")
os.environ["U2NET_HOME"] = os.path.join(HOME_DIR, ".u2net")
os.environ["HF_HOME"] = os.path.join(HOME_DIR, ".cache/huggingface")

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["U2NET_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# تعطيل الـ JIT لتفادي أخطاء التوافق مع إصدارات بايثون الجديدة ورامات Render المحدودة
os.environ["NUMBA_DISABLE_JIT"] = "1"

# ---------------------------------------------------------
# 2. تهيئة التطبيق والموديل
# ---------------------------------------------------------
app = FastAPI(title="Medical Products BG Remover 🚀")

# تحميل الموديل الخفيف u2netp (أفضل للرامات 512MB وسريع جداً لعلب الأدوية)
print("Loading model...")
session = new_session("u2netp")
print("Model loaded successfully!")

# ---------------------------------------------------------
# 3. الروابط (Endpoints)
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "u2netp",
        "environment": "Render.com",
        "message": "API is ready for medical products processing!"
    }

@app.post("/api/remove")
async def enhance_then_remove_bg(file: UploadFile = File(...)):
    # 1. قراءة البيانات المرفوعة
    image_data = await file.read()

    # 2. تحويل البيانات لصورة PIL ومعالجتها
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    # --- تحسينات مخصصة لصور المنتجات الطبية والعلب ---
    # زيادة الوضوح لتحديد حواف العلبة بدقة
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    # زيادة التباين لفصل المنتج عن الخلفية (خاصة لو الخلفية فاتحة)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    # تحسين بسيط في الإضاءة
    img = ImageEnhance.Brightness(img).enhance(1.15)

    # 3. تحويل الصورة المحسنة إلى Bytes لإرسالها لـ rembg
    enhanced_io = io.BytesIO()
    img.save(enhanced_io, format="PNG")
    enhanced_bytes = enhanced_io.getvalue()

    # 4. إزالة الخلفية باستخدام الموديل المحمل مسبقاً
    # تم تفعيل alpha_matting لضمان نعومة الحواف حول علب الأدوية
    result = remove(
        enhanced_bytes, 
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )

    # 5. إرجاع الصورة النهائية كـ Response مباشر
    return Response(content=result, media_type="image/png")

# ---------------------------------------------------------
# تشغيل التطبيق (محلياً للاختبار)
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
