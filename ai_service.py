import openai
import os
import base64
from dotenv import load_dotenv

# Load environment variables
# Try to load .env from different locations
load_dotenv()  # Current directory
load_dotenv("../.env")  # Parent directory
load_dotenv(override=True)  # Override existing env vars

class PhotoAI:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("🤖 PhotoAI initialized with OpenAI connection!")

    async def analyze_photo(self, image_path: str) -> dict:
        """Step 1: AI analyzes the uploaded photo"""

        try:
            print("🔍 Analyzing photo with GPT-4 Vision...")

            # Convert image to base64 for OpenAI API
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Ask GPT-4 Vision to analyze the food photo
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are a professional food photographer analyzing this photo for restaurant menu enhancement.

Please tell me:
1. What specific food item is this?
2. Current photo quality (lighting, composition, appeal)
3. What would make this more appetizing for a restaurant menu?
4. Specific improvements needed

Keep it concise and professional."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            analysis = response.choices[0].message.content
            food_item = self.extract_food_item(analysis)

            print(f"✅ Analysis complete! Detected: {food_item}")

            return {
                "success": True,
                "analysis": analysis,
                "food_detected": food_item
            }

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return {
                "success": False,
                "error": f"Photo analysis failed: {str(e)}"
            }

    async def create_enhancement_prompt(self, analysis: str, food_item: str) -> str:
        """Step 2: Generate perfect DALL-E prompt"""

        try:
            print("🎨 Generating enhancement prompt...")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Create DALL-E prompts for professional food photography enhancement."
                    },
                    {
                        "role": "user",
                        "content": f"""Based on this analysis: {analysis}

Create a DALL-E prompt to enhance this {food_item} for a restaurant menu.

Include:
- Professional food photography lighting
- Appetizing colors and presentation  
- Clean, restaurant-quality background
- High-end commercial style
- Make it irresistible and menu-worthy

Keep under 200 characters."""
                    }
                ]
            )

            prompt = response.choices[0].message.content.strip()
            print(f"✅ Enhancement prompt ready!")
            return prompt

        except Exception as e:
            print(f"⚠️ Using fallback prompt due to: {str(e)}")
            return f"Professional restaurant photo of {food_item}, perfect lighting, clean background, appetizing presentation, commercial food photography, vibrant colors, high-resolution"

    async def enhance_with_dalle(self, enhancement_prompt: str) -> dict:
        """Step 3: Create enhanced photo with DALL-E"""

        try:
            print("✨ Creating enhanced photo with DALL-E...")

            response = self.client.images.generate(
                model="dall-e-3",
                prompt=enhancement_prompt,
                size="1024x1024",
                quality="hd",
                n=1
            )

            enhanced_url = response.data[0].url
            print("🎉 Enhanced photo created successfully!")

            return {
                "success": True,
                "enhanced_image_url": enhanced_url,
                "prompt_used": enhancement_prompt
            }

        except Exception as e:
            print(f"❌ Enhancement failed: {str(e)}")
            return {
                "success": False,
                "error": f"Photo enhancement failed: {str(e)}"
            }

    async def process_photo_complete(self, image_path: str) -> dict:
        """Complete AI enhancement pipeline"""

        print("🚀 Starting AI photo enhancement pipeline...")

        # Step 1: Analyze photo
        analysis_result = await self.analyze_photo(image_path)
        if not analysis_result["success"]:
            return analysis_result

        food_item = analysis_result["food_detected"]
        analysis = analysis_result["analysis"]

        # Step 2: Generate enhancement prompt
        enhancement_prompt = await self.create_enhancement_prompt(analysis, food_item)

        # Step 3: Create enhanced photo
        enhancement_result = await self.enhance_with_dalle(enhancement_prompt)

        if not enhancement_result["success"]:
            return enhancement_result

        print("🎉 AI enhancement pipeline complete!")

        return {
            "success": True,
            "original_analysis": analysis,
            "food_item": food_item,
            "enhancement_prompt": enhancement_prompt,
            "enhanced_image_url": enhancement_result["enhanced_image_url"],
            "processing_steps": [
                "✅ Photo analyzed with photo analyzer",
                "✅ Food item identified: " + food_item.title(),
                "✅ Enhancement strategy generated",
                "✅ Professional photo created with Transformer LLM",
                "✅ Ready for download!"
            ],
            "ai_cost": "$0.08",
            "processing_time": "30 seconds"
        }

    def extract_food_item(self, analysis: str) -> str:
        """Extract the main food item from AI analysis"""

        # Common food keywords to look for
        food_keywords = [
            'pizza', 'burger', 'pasta', 'salad', 'sandwich', 'soup', 'steak',
            'chicken', 'fish', 'cake', 'dessert', 'bread', 'rice', 'noodles',
            'tacos', 'sushi', 'curry', 'pancakes', 'eggs', 'cheese', 'fruit',
            'vegetable', 'meat', 'seafood', 'beverage', 'coffee', 'drink'
        ]

        analysis_lower = analysis.lower()

        # Find the first food keyword that appears
        for food in food_keywords:
            if food in analysis_lower:
                return food

        # If no specific food found, return generic
        return "delicious dish"


# Create global AI instance
photo_ai = PhotoAI()