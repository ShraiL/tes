from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BlipForQuestionAnswering
from PIL import Image

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  🔧 CHANGE THESE FOR YOUR EXAM                                              ║
# ╠═════════════════════════════════════════════════════════════════════════════╣
# ║  IMAGE_PATH   → Path to your image file                                     ║
# ║                 Windows: use r"C:\path\to\image.jpg"                        ║
# ║                 Or relative: "images/test.jpg"                              ║
# ║  TEXT_PROMPT  → Starting text for conditional caption                       ║
# ║                 Examples: "a photo of", "this is", "I see"                  ║
# ║  QUESTION     → Question to ask about the image                             ║
# ║                 Examples: "What is in the image?", "What color is it?"      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

IMAGE_PATH = r"images/test.jpg"       # ← CHANGE to your image path!
TEXT_PROMPT = "a photo of"            # ← CHANGE if professor specifies
QUESTION = "What is in the image?"    # ← CHANGE to your question


# Load models (don't change these!)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load image
image = Image.open(IMAGE_PATH)


def basic_caption(img):
    """Generates caption without any prompt - just describes what it sees"""
    inputs = processor(img, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


def conditional_caption(img, text_prompt):
    """Generates caption starting with your text_prompt"""
    inputs = processor(img, text_prompt, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


def visual_qa(img, question):
    """Answers a question about the image"""
    inputs = processor(img, question, return_tensors="pt")
    outputs = qa_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    print(f"Basic Caption: {basic_caption(image)}")
    print(f"Conditional: {conditional_caption(image, TEXT_PROMPT)}")
    print(f"Q: {QUESTION}")
    print(f"A: {visual_qa(image, QUESTION)}")