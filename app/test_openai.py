import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv("../.env")

print("🔍 Checking OpenAI API key...")
print(f"Current directory: {os.getcwd()}")

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ Found API key: {api_key[:10]}...")

    # Test OpenAI connection
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        print("🤖 OpenAI client created successfully!")
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
else:
    print("❌ No API key found!")
    print("💡 Make sure your .env file contains:")
    print("   OPENAI_API_KEY=sk-your-key-here")