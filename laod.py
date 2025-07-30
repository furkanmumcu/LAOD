import torch
from transformers import pipeline
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np
import cv2
import draw_utils



pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    #device="cuda:1",
    device_map='auto',
    torch_dtype=torch.bfloat16

)


def laod_yolo(image):
    model = YOLO("yolov8x-worldv2.pt")
    messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Just Give the list of objects in given picture seperated by comma. Do not write anything else. Use singular name of the objects."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List the objects that you see in given picture."},
                    {"type": "image", "url": image},

                ]
            },

        ]

    output = pipe(text=messages, max_new_tokens=500)
    print(output[0]["generated_text"][-1]["content"])

    llm_response = output[0]["generated_text"][-1]["content"]

    llm_response = llm_response.lower()
    llm_labels = llm_response.replace(', ', ',').split(',')

    print(llm_labels)

    model.set_classes(llm_labels)
    results = model.predict(image)

    return results[0].plot()




def laod_gdino(image):
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Just Give the list of objects in given picture seperated by comma. Do not write anything else."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List the objects that you see in given picture."},
                    {"type": "image", "url": image},

                ]
            },

        ]


    output = pipe(text=messages, max_new_tokens=500)
    print('LLM response: {}'.format(output[0]["generated_text"][-1]["content"]))

    llm_response = output[0]["generated_text"][-1]["content"]

    llm_response = llm_response.lower()
    llm_response = llm_response.replace('pedestrian', 'person')
    llm_response = llm_response.replace('people', 'person')
    llm_response = llm_response.replace('man', 'person')
    llm_response = llm_response.replace('woman', 'person')

    llm_labels = llm_response.replace(', ', ',').split(',')

    #print(llm_labels)

    llm_labels = [llm_labels]

    inputs = processor(images=image, text=llm_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]
    image = np.array(image)

    draw_results = [result["boxes"], result["scores"], result["labels"]]
    return draw_utils.visualize_detections(image, draw_results)
