import os, torch, re, io
from PIL import Image
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# تحميل النماذج مرة واحدة عند بدء التشغيل
blip_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(
    blip_model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

trans_model_name = "facebook/m2m100_418M"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)

def load_and_compress_image(file, max_side=512, quality=80):
    img = Image.open(file).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def dedup_keywords(kws):
    cleaned, seen = [], set()
    for k in kws:
        k = k.strip().strip("،,;.- ").replace("  ", " ")
        if k and k.lower() not in seen:
            seen.add(k.lower())
            cleaned.append(k)
    return cleaned

def translate_to_arabic(text_en):
    inputs = trans_tokenizer(text_en, return_tensors="pt", truncation=True)
    forced_bos_token_id = trans_tokenizer.get_lang_id("ar")
    with torch.no_grad():
        outputs = trans_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=80
        )
    return trans_tokenizer.decode(outputs[0], skip_special_tokens=True)

def refine_arabic_text(text_ar):
    text_ar = re.sub(r"\s+", " ", text_ar)
    text_ar = text_ar.replace(" ,", ",").replace(" .", ".")
    return text_ar.strip()

def analyze_image_bilingual(file):
    image = load_and_compress_image(file)
    inputs = blip_processor(images=image, return_tensors="pt")
    out_ids = blip_model.generate(**inputs, max_new_tokens=50)
    caption_en = blip_processor.decode(out_ids[0], skip_special_tokens=True)
    caption_ar = refine_arabic_text(translate_to_arabic(caption_en))
    return {
        "arabic_description": caption_ar,
        "arabic_keywords": dedup_keywords(caption_ar.split()),
        "english_keywords": dedup_keywords(caption_en.split())
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        result = analyze_image_bilingual(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Image Captioning API is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render يحدد المنفذ هنا
    app.run(host="0.0.0.0", port=port)

