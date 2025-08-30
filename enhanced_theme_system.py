#!/usr/bin/env python3
"""
Enhanced Theme System with Descriptions and Automated Style Detection
"""

# --- Enhanced theme system with descriptions and automated style detection ---
THEME_DESCRIPTIONS = {
    # Realistic Themes
    "Realistic portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful": {
        "description": "Professional photography with natural skin texture and detailed facial features",
        "style": "realistic",
        "positive_prompt": "realistic portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Professional headshot, natural lighting, studio background, sharp focus, high resolution, detailed facial features": {
        "description": "Corporate headshot style with natural lighting and high resolution",
        "style": "realistic",
        "positive_prompt": "professional headshot, natural lighting, studio background, sharp focus, high resolution, detailed facial features",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Studio photography, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
        "description": "Studio photography with natural lighting and professional camera quality",
        "style": "realistic",
        "positive_prompt": "studio photography, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "High fashion portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality": {
        "description": "High fashion magazine style with 4K quality and detailed features",
        "style": "realistic",
        "positive_prompt": "high fashion portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Celebrity style portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful": {
        "description": "Celebrity magazine style with professional photography",
        "style": "realistic",
        "positive_prompt": "celebrity style portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Magazine cover portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
        "description": "Magazine cover style with professional camera and sharp focus",
        "style": "realistic",
        "positive_prompt": "magazine cover portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Artistic portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality": {
        "description": "Artistic photography with natural skin texture and detailed features",
        "style": "realistic",
        "positive_prompt": "artistic portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Cinematic portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
        "description": "Cinematic movie style with dramatic lighting and professional quality",
        "style": "realistic",
        "positive_prompt": "cinematic portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Vintage portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful": {
        "description": "Vintage photography style with classic studio lighting",
        "style": "realistic",
        "positive_prompt": "vintage portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    "Modern portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality": {
        "description": "Modern photography style with natural lighting and professional camera",
        "style": "realistic",
        "positive_prompt": "modern portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, deformed"
    },
    
    # Cartoon/Animation Themes
    "Van Gogh style": {
        "description": "Post-impressionist painting style with bold colors and expressive brushstrokes",
        "style": "cartoon",
        "positive_prompt": "Van Gogh style painting, post-impressionist, bold colors, expressive brushstrokes, artistic",
        "negative_prompt": "realistic, photograph, modern, clean, smooth"
    },
    "Cartoon version": {
        "description": "Classic cartoon style with bold outlines and vibrant colors",
        "style": "cartoon",
        "positive_prompt": "cartoon style, bold outlines, vibrant colors, animated, stylized",
        "negative_prompt": "realistic, photograph, detailed, natural"
    },
    "Turn into a cyborg": {
        "description": "Sci-fi cyborg style with mechanical parts and futuristic elements",
        "style": "cartoon",
        "positive_prompt": "cyborg, sci-fi, mechanical parts, futuristic, robotic, metallic",
        "negative_prompt": "realistic, natural, organic, human"
    },
    "Watercolor painting": {
        "description": "Soft watercolor painting style with flowing colors and artistic texture",
        "style": "cartoon",
        "positive_prompt": "watercolor painting, soft colors, flowing, artistic, painterly",
        "negative_prompt": "realistic, photograph, sharp, digital"
    },
    "Anime style": {
        "description": "Japanese anime style with large eyes and stylized features",
        "style": "cartoon",
        "positive_prompt": "anime style, Japanese animation, large eyes, stylized features, vibrant colors",
        "negative_prompt": "realistic, photograph, natural, detailed"
    },
    "Futuristic robot": {
        "description": "Futuristic robot design with metallic surfaces and advanced technology",
        "style": "cartoon",
        "positive_prompt": "futuristic robot, metallic, advanced technology, sci-fi, mechanical",
        "negative_prompt": "realistic, human, natural, organic"
    },
    "Old photo": {
        "description": "Vintage photograph style with sepia tones and aged appearance",
        "style": "realistic",
        "positive_prompt": "vintage photograph, sepia tones, aged, nostalgic, classic",
        "negative_prompt": "modern, digital, colorful, sharp"
    },
    "Pop art": {
        "description": "Pop art style with bold colors and graphic design elements",
        "style": "cartoon",
        "positive_prompt": "pop art style, bold colors, graphic design, Andy Warhol inspired",
        "negative_prompt": "realistic, natural, subtle, muted"
    },
    "Impressionist painting": {
        "description": "Impressionist painting style with visible brushstrokes and light effects",
        "style": "cartoon",
        "positive_prompt": "impressionist painting, visible brushstrokes, light effects, artistic",
        "negative_prompt": "realistic, photograph, sharp, detailed"
    },
    "Disney animation style": {
        "description": "Classic Disney animation style with charming and magical appearance",
        "style": "cartoon",
        "positive_prompt": "Disney animation style, charming, magical, colorful, family-friendly",
        "negative_prompt": "realistic, dark, mature, realistic"
    },
    "Studio Ghibli style": {
        "description": "Studio Ghibli animation style with detailed backgrounds and magical atmosphere",
        "style": "cartoon",
        "positive_prompt": "Studio Ghibli style, magical atmosphere, detailed backgrounds, anime",
        "negative_prompt": "realistic, photograph, modern, dark"
    },
    "Pixar 3D animation": {
        "description": "Pixar 3D animation style with detailed textures and expressive features",
        "style": "cartoon",
        "positive_prompt": "Pixar 3D animation, detailed textures, expressive features, colorful",
        "negative_prompt": "realistic, photograph, 2D, flat"
    },
    "Realistic humanoid": {
        "description": "Realistic humanoid robot with detailed mechanical features",
        "style": "realistic",
        "positive_prompt": "realistic humanoid robot, detailed mechanical features, advanced technology",
        "negative_prompt": "cartoon, anime, stylized, simple"
    },
    "Cartoon Network style": {
        "description": "Cartoon Network animation style with bold colors and exaggerated features",
        "style": "cartoon",
        "positive_prompt": "Cartoon Network style, bold colors, exaggerated features, animated",
        "negative_prompt": "realistic, natural, subtle, detailed"
    },
    "Marvel comic style": {
        "description": "Marvel comic book style with dynamic poses and bold colors",
        "style": "cartoon",
        "positive_prompt": "Marvel comic style, dynamic poses, bold colors, superhero",
        "negative_prompt": "realistic, photograph, natural, subtle"
    },
    "DC Comics style": {
        "description": "DC Comics style with dramatic lighting and comic book aesthetics",
        "style": "cartoon",
        "positive_prompt": "DC Comics style, dramatic lighting, comic book, superhero",
        "negative_prompt": "realistic, photograph, natural, subtle"
    },
    "Japanese manga": {
        "description": "Japanese manga style with distinctive black and white artwork",
        "style": "cartoon",
        "positive_prompt": "Japanese manga style, black and white, distinctive artwork, anime",
        "negative_prompt": "realistic, photograph, colorful, natural"
    },
    "European oil painting": {
        "description": "Classical European oil painting style with rich textures and colors",
        "style": "realistic",
        "positive_prompt": "European oil painting, classical, rich textures, artistic, masterwork",
        "negative_prompt": "cartoon, anime, modern, digital"
    },
    "Digital art": {
        "description": "Modern digital art style with clean lines and vibrant colors",
        "style": "cartoon",
        "positive_prompt": "digital art, clean lines, vibrant colors, modern, stylized",
        "negative_prompt": "realistic, photograph, traditional, natural"
    },
    "Steampunk style": {
        "description": "Steampunk aesthetic with Victorian-era mechanical elements",
        "style": "cartoon",
        "positive_prompt": "steampunk style, Victorian-era, mechanical elements, brass, gears",
        "negative_prompt": "realistic, modern, clean, simple"
    },
    "Cyberpunk aesthetic": {
        "description": "Cyberpunk aesthetic with neon lights and futuristic urban elements",
        "style": "cartoon",
        "positive_prompt": "cyberpunk aesthetic, neon lights, futuristic, urban, dystopian",
        "negative_prompt": "realistic, natural, rural, peaceful"
    },
    "Fantasy art": {
        "description": "Fantasy art style with magical elements and mystical atmosphere",
        "style": "cartoon",
        "positive_prompt": "fantasy art, magical elements, mystical atmosphere, enchanted",
        "negative_prompt": "realistic, modern, mundane, ordinary"
    },
    "Sci-fi concept art": {
        "description": "Science fiction concept art with futuristic technology and alien worlds",
        "style": "cartoon",
        "positive_prompt": "sci-fi concept art, futuristic technology, alien worlds, space",
        "negative_prompt": "realistic, historical, natural, Earth"
    },
    "Horror movie style": {
        "description": "Horror movie aesthetic with dark atmosphere and dramatic lighting",
        "style": "realistic",
        "positive_prompt": "horror movie style, dark atmosphere, dramatic lighting, suspenseful",
        "negative_prompt": "bright, cheerful, colorful, safe"
    },
    "Romantic painting": {
        "description": "Romantic era painting style with emotional and dramatic elements",
        "style": "realistic",
        "positive_prompt": "romantic painting, emotional, dramatic, artistic, classical",
        "negative_prompt": "cartoon, anime, modern, digital"
    }
}

# Enhanced theme categories with descriptions
THEME_CATEGORIES = {
    "Human": [
        # Realistic Themes
        "Realistic portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "Professional headshot, natural lighting, studio background, sharp focus, high resolution, detailed facial features",
        "Studio photography, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "High fashion portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
        "Celebrity style portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "Magazine cover portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "Artistic portrait, professional photography, natural skin texture, detailed eyes, natural hair, studio lighting, 4k quality",
        "Cinematic portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        "Vintage portrait, professional photography, natural skin, detailed features, studio background, sharp focus, beautiful",
        "Modern portrait, natural lighting, detailed face, professional camera, sharp focus, beautiful features, high quality",
        # Cartoon/Animation Themes
        "Van Gogh style",
        "Cartoon version",
        "Turn into a cyborg",
        "Watercolor painting",
        "Anime style",
        "Futuristic robot",
        "Old photo",
        "Pop art",
        "Impressionist painting",
        "Disney animation style",
        "Studio Ghibli style",
        "Pixar 3D animation",
        "Realistic humanoid",
        "Cartoon Network style",
        "Marvel comic style",
        "DC Comics style",
        "Japanese manga",
        "European oil painting",
        "Digital art",
        "Steampunk style",
        "Cyberpunk aesthetic",
        "Fantasy art",
        "Sci-fi concept art",
        "Horror movie style",
        "Romantic painting"
    ],
    "Architecture": [
        "Modern skyscraper",
        "Classic European landmark",
        "Futuristic cityscape",
        "Ancient temple",
        "Famous bridge",
        "Interior design",
        "Street scene",
        "Nature landscape",
    ],
    "Object": [
        "Furniture design",
        "Product rendering",
        "Vehicle concept art",
        "Gadget prototype",
        "Fashion design sketch",
        "Jewelry design",
        "Sports equipment",
        "Musical instrument",
        "Book cover art",
        "Poster design",
        "Still life painting",
        "Food photography",
        "Animal illustration",
    ],
    "Animal": [
        "Animal illustration",
        "Nature landscape",
        "Cartoon version",
        "Anime style",
        "Pixar 3D animation",
        "Studio Ghibli style",
        "Disney animation style",
        "Realistic humanoid",
    ],
    "Other": [
        "DALL-E surrealism",
        "Midjourney photorealism",
        "Stable Diffusion fantasy",
        "DreamBooth custom style",
        "Deep Dream psychedelic",
        "RunwayML cinematic",
        "NightCafe creative",
        "Artbreeder blend",
    ]
}

def get_theme_info(theme_name):
    """Get theme information including description, style, and prompts"""
    if theme_name in THEME_DESCRIPTIONS:
        return THEME_DESCRIPTIONS[theme_name]
    else:
        # Default for themes without specific descriptions
        return {
            "description": "Custom theme with unique artistic style",
            "style": "realistic",  # Default to realistic
            "positive_prompt": theme_name,
            "negative_prompt": "deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality"
        }

def detect_theme_style(theme_name):
    """Automatically detect if a theme is realistic or cartoon based on keywords"""
    theme_lower = theme_name.lower()
    
    # Cartoon/animation keywords
    cartoon_keywords = [
        "cartoon", "anime", "animation", "disney", "pixar", "ghibli", "manga", 
        "comic", "pop art", "watercolor", "impressionist", "van gogh", "cyborg",
        "robot", "steampunk", "cyberpunk", "fantasy", "sci-fi", "digital art"
    ]
    
    # Realistic keywords
    realistic_keywords = [
        "realistic", "photograph", "professional", "studio", "natural", "cinematic",
        "vintage", "modern", "headshot", "portrait", "fashion", "celebrity",
        "magazine", "artistic", "oil painting", "humanoid", "horror", "romantic"
    ]
    
    # Check for cartoon keywords first
    for keyword in cartoon_keywords:
        if keyword in theme_lower:
            return "cartoon"
    
    # Check for realistic keywords
    for keyword in realistic_keywords:
        if keyword in theme_lower:
            return "realistic"
    
    # Default to realistic if no clear match
    return "realistic"

def get_enhanced_prompts_for_theme(theme_name, sketch_type="face", user_prompt=""):
    """Get enhanced prompts based on theme, sketch type, and user input"""
    theme_info = get_theme_info(theme_name)
    
    # PRIORITY: User prompt comes FIRST to avoid truncation
    if user_prompt:
        # Start with user prompt, then add essential quality enhancements
        if sketch_type == "face":
            if theme_info['style'] == "realistic":
                enhanced_prompt = f"{user_prompt}, realistic portrait of one person, {theme_info['positive_prompt']}, detailed facial features, proper anatomy, single face"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error"
            else:  # cartoon
                enhanced_prompt = f"{user_prompt}, anime style portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single face"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph"
        else:  # full_body
            if theme_info['style'] == "realistic":
                enhanced_prompt = f"{user_prompt}, realistic full body portrait of one person, {theme_info['positive_prompt']}, proper anatomy, single figure, natural pose"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people"
            else:  # cartoon
                enhanced_prompt = f"{user_prompt}, anime style full body portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single figure, dynamic pose"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph"
    else:
        # Fallback: No user prompt, use theme-based construction
        if sketch_type == "face":
            if theme_info['style'] == "realistic":
                enhanced_prompt = f"realistic portrait of one person, {theme_info['positive_prompt']}, detailed facial features, proper anatomy, single face"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error"
            else:  # cartoon
                enhanced_prompt = f"anime style portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single face"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, realistic, photograph"
        else:  # full_body
            if theme_info['style'] == "realistic":
                enhanced_prompt = f"realistic full body portrait of one person, {theme_info['positive_prompt']}, proper anatomy, single figure, natural pose"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people"
            else:  # cartoon
                enhanced_prompt = f"anime style full body portrait, {theme_info['positive_prompt']}, stylized features, clean lines, detailed illustration, single figure, dynamic pose"
                enhanced_negative = f"{theme_info['negative_prompt']}, deformed, extra limbs, multiple faces, mutated hands, blurry, out of focus, low quality, glitch, split face, extra heads, extra bodies, anatomical error, multiple people, realistic, photograph"
    
    return enhanced_prompt, enhanced_negative, theme_info['style'] 