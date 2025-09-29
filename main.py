from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from rembg import remove
import io

# إنشاء تطبيق FastAPI
app = FastAPI(title="Background Remover API")

# Endpoint بسيط للترحيب والتأكد من أن الخدمة تعمل
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}

# Endpoint الأساسي الذي يستقبل الصورة ويعالجها
@app.post("/remove-background")
async def process_image(file: UploadFile = File(...)):
    # قراءة محتوى الصورة التي تم رفعها
    image_bytes = await file.read()

    # استخدام مكتبة rembg لإزالة الخلفية
    # New line specifying a smaller model
    processed_image_bytes = remove(image_bytes, model_name="u2netp")
    # إرجاع الصورة الجديدة كاستجابة مباشرة
    # نستخدم StreamingResponse لإرسال بيانات الصورة (bytes) مباشرة
    return StreamingResponse(content=io.BytesIO(processed_image_bytes), media_type="image/png")
