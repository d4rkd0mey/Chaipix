import os
import base64
import time
import asyncio
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()


class PhotoAI:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def analyze_food_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze the uploaded food image to understand what it contains"""
        try:
            # Read and encode image for GPT-4 Vision
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 Vision for analysis
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this food image and provide:
1. Main food item/dish name
2. Cuisine type (southeast_asian, western, asian, mediterranean, etc.)
3. Current lighting and composition quality (1-10 scale)
4. Specific improvements needed for professional food photography
5. Authentic cultural context for plating style

Respond in this exact format:
FOOD_ITEM: [dish name]
CUISINE_TYPE: [cuisine_type]
QUALITY_SCORE: [1-10]
IMPROVEMENTS: [specific issues to fix]
CULTURAL_CONTEXT: [authentic plating traditions]"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            analysis = response.choices[0].message.content

            # Parse the structured response
            lines = analysis.split('\n')
            result = {}

            for line in lines:
                if line.startswith('FOOD_ITEM:'):
                    result['food_item'] = line.replace('FOOD_ITEM:', '').strip()
                elif line.startswith('CUISINE_TYPE:'):
                    result['cuisine_type'] = line.replace('CUISINE_TYPE:', '').strip()
                elif line.startswith('QUALITY_SCORE:'):
                    result['quality_score'] = line.replace('QUALITY_SCORE:', '').strip()
                elif line.startswith('IMPROVEMENTS:'):
                    result['improvements'] = line.replace('IMPROVEMENTS:', '').strip()
                elif line.startswith('CULTURAL_CONTEXT:'):
                    result['cultural_context'] = line.replace('CULTURAL_CONTEXT:', '').strip()

            return result

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return {
                'food_item': 'Unknown Food Item',
                'cuisine_type': 'unknown',
                'quality_score': '5',
                'improvements': 'General enhancement needed',
                'cultural_context': 'Standard presentation'
            }

    def build_enhancement_prompt(self, analysis: Dict[str, Any], style_options: Dict[str, str]) -> str:
        """Build a concise prompt for image enhancement (under 1000 chars for OpenAI limit)"""

        food_item = analysis.get('food_item', 'food')
        cuisine_type = analysis.get('cuisine_type', 'unknown')

        # Start with core enhancement
        prompt_parts = [f"Transform {food_item} into professional food photography"]

        # Lighting (shortened)
        lighting_map = {
            'natural': 'natural window light',
            'studio': 'studio lighting',
            'soft': 'soft diffused light'
        }
        lighting = style_options.get('lighting', 'natural')
        prompt_parts.append(lighting_map.get(lighting, 'natural light'))

        # Camera angle (shortened)
        angle_map = {
            '45_degree_overhead': '45° overhead angle',
            'straight_overhead': 'flat lay overhead',
            'eye_level': 'eye level view',
            'slight_angle': 'slight angle'
        }
        angle = style_options.get('angle', '45_degree_overhead')
        prompt_parts.append(angle_map.get(angle, '45° overhead'))

        # Surface/Background (auto-detect or user choice)
        surface = style_options.get('surface', 'auto')
        if surface == 'auto':
            if 'asian' in cuisine_type.lower():
                surface_text = 'banana leaf/dark wood'
            else:
                surface_text = 'marble/wood surface'
        else:
            surface_map = {
                'wooden_table': 'wood table',
                'marble_surface': 'marble surface',
                'ceramic_plate': 'white ceramic',
                'bamboo_mat': 'bamboo mat',
                'dark_slate': 'dark slate'
            }
            surface_text = surface_map.get(surface, 'clean surface')

        prompt_parts.append(surface_text)

        # Color temperature (shortened)
        color_temp_map = {
            'warm': 'warm tones',
            'neutral': 'neutral colors',
            'cool': 'cool tones'
        }
        color_temp = style_options.get('color_temperature', 'warm')
        prompt_parts.append(color_temp_map.get(color_temp, 'warm tones'))

        # Plating style (shortened)
        plating_map = {
            'authentic': 'authentic plating',
            'clean': 'clean precise plating',
            'rustic': 'rustic plating'
        }
        plating = style_options.get('plating', 'authentic')
        prompt_parts.append(plating_map.get(plating, 'authentic plating'))

        # Garnish (shortened)
        garnish_map = {
            'minimal': 'minimal garnish',
            'moderate': 'moderate garnish',
            'elaborate': 'elaborate garnish'
        }
        garnish = style_options.get('garnish', 'moderate')
        prompt_parts.append(garnish_map.get(garnish, 'moderate garnish'))

        # Quality and finish
        quality = style_options.get('quality', 'hd')
        if quality == 'hd':
            prompt_parts.append('DSLR quality, sharp focus')

        # Essential enhancements
        prompt_parts.extend([
            'enhanced textures',
            'appetizing appearance',
            'remove clutter',
            'professional commercial quality'
        ])

        # Join and ensure under 1000 chars
        full_prompt = ', '.join(prompt_parts) + '.'

        # Truncate if too long (safety check)
        if len(full_prompt) > 1000:
            full_prompt = full_prompt[:997] + '...'

        return full_prompt

    async def enhance_image_with_editing(self, image_path: str, prompt: str, quality: str = 'hd') -> Dict[str, Any]:
        """Use OpenAI image editing to enhance the food photo"""
        png_path = None
        file_handle = None

        try:
            print(f"Starting image enhancement for: {image_path}")

            # Determine image size based on quality (gpt-image-1 supported sizes)
            size = "1024x1024" if quality == 'hd' else "auto"
            print(f"Using size: {size}")

            # Convert image to PNG with RGBA (required by OpenAI image editing API)
            print("Converting image to PNG with alpha channel...")
            with Image.open(image_path) as img:
                print(f"Original image mode: {img.mode}, size: {img.size}")

                # Convert to RGBA (add alpha channel if missing)
                if img.mode != 'RGBA':
                    print(f"Converting from {img.mode} to RGBA")
                    img = img.convert('RGBA')

                # Create temporary PNG file
                png_path = image_path.rsplit('.', 1)[0] + '_temp.png'
                img.save(png_path, 'PNG')
                print(f"Saved temporary PNG: {png_path}")

            # Check file size
            file_size = os.path.getsize(png_path)
            print(f"PNG file size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")

            if file_size > 4 * 1024 * 1024:  # 4MB limit for OpenAI
                raise Exception(f"Image too large: {file_size / 1024 / 1024:.2f} MB (max 4MB)")

            print("Calling OpenAI image edit API...")
            print(f"Prompt length: {len(prompt)} characters")
            print(f"Prompt preview: {prompt[:200]}...")

            # Open file handle for API call
            file_handle = open(png_path, "rb")

            # Use OpenAI image editing with PNG file (gpt-image-1 format)
            response = self.client.images.edit(
                model="gpt-image-1",
                image=file_handle,
                prompt=prompt,
                size=size
                # Note: gpt-image-1 doesn't accept response_format parameter
            )

            print("OpenAI API call successful!")

            # Get the enhanced image
            enhanced_base64 = response.data[0].b64_json
            enhanced_bytes = base64.b64decode(enhanced_base64)

            # Save the enhanced image
            output_dir = "static/enhanced" if os.path.exists("static") else "../static/enhanced"
            os.makedirs(output_dir, exist_ok=True)

            output_filename = f"enhanced_{int(time.time())}.png"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "wb") as f:
                f.write(enhanced_bytes)

            # Return the URL for web access
            enhanced_url = f"/static/enhanced/{output_filename}"

            return {
                'success': True,
                'enhanced_image_url': enhanced_url,
                'output_path': output_path
            }

        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            print(f"Error type: {type(e).__name__}")

            # If it's an OpenAI API error, get more details
            if hasattr(e, 'response'):
                print(f"API response status: {e.response.status_code}")
                print(f"API response text: {e.response.text}")

            return {
                'success': False,
                'error': f"Image enhancement failed: {str(e)}"
            }

        finally:
            # Close file handle first
            if file_handle:
                file_handle.close()
                print("Closed file handle")

            # Clean up temporary PNG file
            if png_path and os.path.exists(png_path):
                try:
                    os.remove(png_path)
                    print(f"Cleaned up temporary PNG: {png_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temp file {png_path}: {cleanup_error}")

    async def process_photo_with_image_editing(self, image_path: str, style_options: Dict[str, str]) -> Dict[str, Any]:
        """Main function for processing photos with image editing approach"""
        start_time = time.time()

        try:
            print(f"Analyzing image: {image_path}")

            # Step 1: Analyze the food image
            analysis = await self.analyze_food_image(image_path)

            print(
                f"Detected: {analysis.get('food_item', 'Unknown')} ({analysis.get('cuisine_type', 'unknown')} cuisine)")

            # Step 2: Build enhancement prompt
            enhancement_prompt = self.build_enhancement_prompt(analysis, style_options)

            print(f"Enhancement prompt: {enhancement_prompt[:100]}...")

            # Step 3: Enhance the image
            enhancement_result = await self.enhance_image_with_editing(
                image_path,
                enhancement_prompt,
                style_options.get('quality', 'hd')
            )

            if not enhancement_result['success']:
                return enhancement_result

            # Calculate processing time and cost
            processing_time = round(time.time() - start_time, 2)

            # Estimate cost (OpenAI image editing pricing)
            base_cost = 0.20 if style_options.get('quality') == 'hd' else 0.40

            return {
                'success': True,
                'food_item': analysis.get('food_item', 'Delicious Food'),
                'cuisine_type': analysis.get('cuisine_type', 'unknown'),
                'original_analysis': f"Quality score: {analysis.get('quality_score', 'N/A')}/10. {analysis.get('improvements', 'General enhancements applied.')}",
                'enhanced_image_url': enhancement_result['enhanced_image_url'],
                'processing_steps': [
                    'AI analyzed your food photo',
                    f"Detected {analysis.get('food_item', 'food item')} ({analysis.get('cuisine_type', 'unknown')} cuisine)",
                    'Built custom enhancement prompt',
                    'Applied professional food photography editing',
                    'Generated enhanced image'
                ],
                'enhancement_instructions': enhancement_prompt,
                'style_options_used': style_options,
                'ai_cost': f"${base_cost:.2f}",
                'processing_time': f"{processing_time} seconds"
            }

        except Exception as e:
            print(f"Error in process_photo_with_image_editing: {str(e)}")
            return {
                'success': False,
                'error': f"Photo processing failed: {str(e)}"
            }

    async def process_photo_complete(self, image_path: str) -> Dict[str, Any]:
        """Legacy function for backward compatibility (original DALL-E generation method)"""
        # Use default style options for the legacy method
        default_style_options = {
            'lighting': 'natural',
            'angle': '45_degree_overhead',
            'surface': 'auto',
            'plating': 'authentic',
            'color_temperature': 'warm',
            'garnish': 'moderate',
            'quality': 'hd',
            'imperfections': 'on'
        }

        return await self.process_photo_with_image_editing(image_path, default_style_options)


# Create global instance
photo_ai = PhotoAI()