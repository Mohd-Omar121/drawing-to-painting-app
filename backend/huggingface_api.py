import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw, ImageFilter
import io

def get_pix2pix_pipeline():
    if not hasattr(get_pix2pix_pipeline, "pipe"):
        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        get_pix2pix_pipeline.pipe = pipe
    return get_pix2pix_pipeline.pipe

def generate_image_local(input_image: Image.Image, prompt: str) -> bytes:
    """
    Run Pix2Pix locally with improved quality settings.
    - Use real photos for best results.
    - Example prompts:
        'Turn this photo into a Van Gogh style painting.'
        'Make this person look like a cartoon character.'
        'Add a cyberpunk background.'
    """
    pipe = get_pix2pix_pipeline()
    input_image = input_image.convert("RGB").resize((512, 512))
    result = pipe(prompt, image=input_image, num_inference_steps=15, image_guidance_scale=1.5)
    output_image = result.images[0]
    
    # Add golden frame
    output_image = add_golden_frame(output_image)
    
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    return buf.getvalue()

def add_golden_frame(img: Image.Image, frame_width: int = 32) -> Image.Image:
    # Create a new image with a golden frame
    w, h = img.size
    new_w, new_h = w + 2*frame_width, h + 2*frame_width
    frame = Image.new("RGB", (new_w, new_h), (212, 175, 55))  # gold color
    # Add a gradient effect
    draw = ImageDraw.Draw(frame)
    for i in range(frame_width):
        color = (212 - i*2, 175 - i, 55 + i)
        draw.rectangle([i, i, new_w-i-1, new_h-i-1], outline=color)
    # Paste the original image in the center
    frame.paste(img, (frame_width, frame_width))
    # Optionally add a shadow
    frame = frame.filter(ImageFilter.GaussianBlur(radius=1))
    frame.paste(img, (frame_width, frame_width))
    return frame
