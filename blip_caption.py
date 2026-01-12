from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BlipForQuestionAnswering
from PIL import Image

IMAGE_PATH = r"images/test.jpg"
TEXT_PROMPT = "a photo of"
QUESTION = "What is in the image?"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

image = Image.open(IMAGE_PATH)

def basic_caption(img):
    inputs = processor(img, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def conditional_caption(img, text_prompt):
    inputs = processor(img, text_prompt, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def visual_qa(img, question):
    inputs = processor(img, question, return_tensors="pt")
    outputs = qa_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    print(f"Basic Caption: {basic_caption(image)}")
    print(f"Conditional: {conditional_caption(image, TEXT_PROMPT)}")
    print(f"Q: {QUESTION}")
    print(f"A: {visual_qa(image, QUESTION)}")