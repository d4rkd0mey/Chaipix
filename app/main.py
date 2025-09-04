from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import uuid
import aiofiles
from ai_service import photo_ai  # Import our AI brain!

# Create your app
app = FastAPI(title="PhotoAI - AI-Powered Food Enhancement")

# Add support for HTML pages and static files
app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../templates")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", {
        "request": request,
        "title": "PhotoAI - AI Photo Enhancement",
        "hero_title": "Turn Ordinary Food Photos Into Professional Menu Images",
        "hero_subtitle": "Our AI analyzes and enhances your food photos in 30 seconds. Perfect for restaurants, food blogs, and social media.",
        "stats": {
            "photos_enhanced": "2,847+",
            "average_time": "30 Sec",
            "price": "$0.50"
        }
    })


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": "Upload Your Photo - PhotoAI"
    })


# 🤖 THE MAGIC HAPPENS HERE! AI-powered upload processing
@app.post("/upload", response_class=HTMLResponse)
async def upload_and_enhance_with_ai(request: Request, file: UploadFile = File(...)):
    """Upload photo and enhance it with AI magic!"""

    print(f"🚀 New photo upload: {file.filename}")

    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        return templates.TemplateResponse("error.html", {
            "request": request,
            "title": "Upload Error",
            "error_title": "Invalid File Type",
            "error_message": "Please upload a valid image file (JPG, PNG, WEBP)",
            "back_link": "/upload"
        })

    # Check file size
    content = await file.read()
    file_size = len(content)

    if file_size > 10 * 1024 * 1024:  # 10MB limit
        return templates.TemplateResponse("error.html", {
            "request": request,
            "title": "File Too Large",
            "error_title": "File Size Limit Exceeded",
            "error_message": "Please upload an image smaller than 10MB",
            "back_link": "/upload"
        })

    # Save temp file for AI processing
    temp_dir = "../temp"
    os.makedirs(temp_dir, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'jpg'
    temp_filename = f"{file_id}.{file_extension}"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        # Save uploaded file
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)

        print(f"💾 Saved temp file: {temp_filename}")

        # 🤖 AI ENHANCEMENT PIPELINE!
        print("🧠 Starting AI enhancement...")
        ai_result = await photo_ai.process_photo_complete(temp_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("🗑️ Cleaned up temp file")

        if ai_result["success"]:
            # Success! Show beautiful AI results
            return templates.TemplateResponse("ai_result.html", {
                "request": request,
                "title": "✨ AI Enhancement Complete!",
                "original_filename": file.filename,
                "original_size": f"{file_size // 1024} KB",
                "food_item": ai_result["food_item"],
                "analysis": ai_result["original_analysis"],
                "enhanced_image_url": ai_result["enhanced_image_url"],
                "processing_steps": ai_result["processing_steps"],
                "enhancement_prompt": ai_result["enhancement_prompt"],
                "ai_cost": ai_result.get("ai_cost", "$0.05"),
                "processing_time": ai_result.get("processing_time", "30 seconds")
            })
        else:
            # AI processing failed
            return templates.TemplateResponse("error.html", {
                "request": request,
                "title": "AI Processing Failed",
                "error_title": "Enhancement Error",
                "error_message": ai_result["error"],
                "back_link": "/upload"
            })

    except Exception as e:
        # Clean up on any error
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"❌ Error processing photo: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "title": "Processing Error",
            "error_title": "Something Went Wrong",
            "error_message": f"Error processing your photo: {str(e)}",
            "back_link": "/upload"
        })


# Keep your existing API endpoint (JSON response for developers)
@app.post("/api/upload")
async def api_upload_photo(file: UploadFile = File(...)):
    """API endpoint - returns JSON (for developers)"""

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    file_size = len(await file.read())

    return {
        "success": "✅ Got your photo!",
        "photo_name": file.filename,
        "photo_size": f"{file_size // 1024} KB",
        "ai_enhancement": "Now available! Use /upload for full AI processing",
        "status": "Photo received - AI enhancement ready! 🤖"
    }


@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {
        "status": "✅ PhotoAI with AI enhancement is running!",
        "version": "3.0.0 - AI Powered",
        "features": ["GPT-4 Vision Analysis", "DALL-E 3 Enhancement", "Professional Results"],
        "ai_status": "🤖 Ready for photo enhancement"
    }


# Test endpoint to check AI connection
@app.get("/test-ai")
async def test_ai_connection():
    """Test if OpenAI connection is working"""
    try:
        # Simple test to see if OpenAI API key works
        response = photo_ai.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'AI connection working!'"}],
            max_tokens=10
        )

        return {
            "ai_status": "✅ Connected to OpenAI!",
            "test_response": response.choices[0].message.content,
            "models_available": ["GPT-4 Vision", "DALL-E 3", "GPT-4o-mini"]
        }
    except Exception as e:
        return {
            "ai_status": "❌ OpenAI connection failed",
            "error": str(e),
            "fix": "Check your OPENAI_API_KEY in .env file"
        }


if __name__ == "__main__":
    print("🚀 Starting PhotoAI with AI Enhancement...")
    print("🏠 Homepage: http://localhost:8000")
    print("🤖 AI Upload: http://localhost:8000/upload")
    print("🧪 Test AI: http://localhost:8000/test-ai")
    print("📚 API Docs: http://localhost:8000/docs")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)