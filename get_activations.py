import torch
from PIL import Image
import re
import json
import numpy as np
import pandas as pd
import os

from transformers import AutoProcessor, AutoModelForImageTextToText
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")



# ────────────────────────────────────────────────────────────
# 1) Build df_images from your local folder
# ────────────────────────────────────────────────────────────

image_folder = "extended_samples"
# grab all common image files
image_paths = [
    os.path.join(image_folder, fname)
    for fname in os.listdir(image_folder)
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
]

# create a DataFrame with a dummy interestingness_value column (so your
# df_images[['img_path','interestingness_value']] still works)
df_images = pd.DataFrame({
    "img_path": image_paths,
    "interestingness_value": np.nan,   # placeholder if you don’t have pre-labels
})

# ────────────────────────────────────────────────────────────
# 2) Create a simple df_personas_sample
# ────────────────────────────────────────────────────────────

# Define one “default” persona. You can of course add more rows here.
data = {
    "age":            [30],
    "gender":         ["female"],
    "country":        ["USA"],
    "continent":      ["North America"],
    "job_branch":     ["engineering"],
    "mental_workload":["medium"],
    "emotion":        ["neutral"],
}

# Use a meaningful index (this becomes your user_id)
df_personas_sample = pd.DataFrame(data, index=[0])

# =============================================================================
# 1. Setup Hook Registrations for Hugging Face Model
# =============================================================================

# Global dictionaries to store hook outputs
# attentions = {}  # Uncomment to store attention weights
layer_embeddings = {}

# Helper function to create hook functions that store outputs in a dictionary.
def get_hook(name, store_dict):
    def hook(module, input, output):
        store_dict[name] = output
    return hook

# Register hooks for language model layers.
for idx, layer in enumerate(model.language_model.model.layers):
    # if hasattr(layer, 'self_attn'):
    #     layer.self_attn.register_forward_hook(get_hook(f"lang_layer_{idx}_attn", attentions))
    # elif hasattr(layer, 'cross_attn'):
    #     layer.cross_attn.register_forward_hook(get_hook(f"lang_layer_{idx}_attn", attentions))
    # else:
    #     print(f"Layer {idx} has no attention module to hook.")
    layer.register_forward_hook(get_hook(f"language_layer_{idx}_embedding", layer_embeddings))

# Register hooks for the vision model’s main transformer layers.
for idx, layer in enumerate(model.vision_model.transformer.layers):
    # layer.self_attn.register_forward_hook(get_hook(f"vision_layer_{idx}_attn", attentions))
    layer.register_forward_hook(get_hook(f"vision_layer_{idx}_embedding", layer_embeddings))

# Register hooks for the vision model’s global transformer layers (if available).
for idx, layer in enumerate(model.vision_model.global_transformer.layers):
    # layer.self_attn.register_forward_hook(get_hook(f"global_vision_layer_{idx}_attn", attentions))
    layer.register_forward_hook(get_hook(f"global_vision_layer_{idx}_embedding", layer_embeddings))

# Helper function to remove the batch dimension (assumes batch size 1)
def remove_batch_dimension(tensor):
    # Tensor to numpy
    array = tensor[0].to(torch.float32).cpu().numpy()
    # Remove batch dimension
    return array.squeeze(0)

# =============================================================================
# 2. Define the Unified Model Response Function Using Hugging Face
# =============================================================================

def model_response(prompt, image_path):
    """
    Uses Hugging Face's generate() method to produce a textual response from the model.
    It loads the image, builds the input prompt via the processor, and returns the decoded response.
    This forward pass will also trigger the hooks to capture attentions and embeddings.
    """
    # Load the image
    image_obj = Image.open(image_path)
    
    # Prepare the prompt in a chat-style format.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Convert the image and text to model inputs.
    inputs = processor(image_obj, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    
    # Generate the response (this will trigger the hooks)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=1000, temperature=0.6, top_p=0.9)
    
    # Decode the generated tokens into text.
    response_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Return only the assistant's response. (cut everything before the first "assistant")
    response_text = response_text[response_text.find("assistant"):]

    return response_text

def get_response_with_embeddings(user_id, df_temp, image_path, df_images):
    """
    Constructs the prompt using user details from df_temp, calls model_response() to get the
    textual response and (via hooks) capture the embeddings and attentions. Then it parses the
    JSON output from the response and returns a dictionary containing all the data.
    """
    # Extract user details.
    age_temp = df_temp['age'].values[0]
    gender_temp = df_temp['gender'].values[0]
    country_temp = df_temp['country'].values[0]
    continent_temp = df_temp['continent'].values[0]
    job_branch_temp = df_temp['job_branch'].values[0]
    mental_workload_temp = df_temp['mental_workload'].values[0]
    emotion_temp = df_temp['emotion'].values[0]

    # Construct a persona description.
    persona_list = "\n".join([
        f"{country_temp}, {age_temp}, {gender_temp}, {job_branch_temp}, {mental_workload_temp}, {emotion_temp}"
    ])

    # Create the prompt.
    prompt = f"""
Imagine you are a person with the following specific details (origin, age, gender, professional background, mental workload, emotion):

{persona_list}

You are observing the provided image. Based on your details, rate how interesting you find this image by selecting only one of the following options: 
"Not Interesting," "Slightly Interesting," "Moderately Interesting," "Very Interesting," or "Extremely Interesting".

Provide a brief explanation in one short sentence without going into excessive detail. Present the output as a JSON object in the following format:
{{
    "interestingness": "...",
    "explanation": "..."
}}
    """
    # Get the textual response from the model. This single forward pass will also fill the hook dictionaries.
    response = model_response(prompt, image_path)
    print(response)
    
    # Use regex to extract the JSON portion.
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        json_part = match.group()
        data = json.loads(json_part)
        interestingness = data["interestingness"]
        explanation = data["explanation"]
        print(f"Interestingness: {interestingness}")
        print(f"Explanation: {explanation}")
    else:
        raise ValueError("No JSON found in the output.")

    # Retrieve the captured attentions and embeddings (removing the batch dimension).
    # collected_attentions = {k: remove_batch_dimension(v) for k, v in attentions.items()} # Uncomment to collect attentions
    collected_embeddings = {k: remove_batch_dimension(v) for k, v in layer_embeddings.items()}

    # Clear the hook dictionaries for the next forward pass.
    # attentions.clear() # Uncomment to clear attentions
    layer_embeddings.clear()

    # Get the index (img_id) of the image using the image path.
    img_id = df_images[df_images['img_path'] == image_path].index[0]

    # Build the result dictionary.
    result = {
        "user_id": user_id,
        "img_id": img_id,
        "interestingness": interestingness,
        "explanation": explanation,
        # "attentions": collected_attentions, # Uncomment to include attentions
        "embeddings": collected_embeddings
    }
    return result

# =============================================================================
# 3. Example of Processing Multiple Users/Images and Saving as a NumPy Dictionary File
# =============================================================================

# Load your user and image DataFrames
df_personas_sample = pd.read_pickle('/nasdata/abdu/demographics/df_generated-personas-sample.pkl')
df_images = pd.read_pickle('/nasdata/abdu/demographics/df_common_machine_int.pkl')

df_images = df_images[['img_path', 'interestingness_value']]

results_list = []
counter = 0

# Loop over each user.
for i in range(len(df_personas_sample)):
    user_temp = df_personas_sample.index[i]
    
    # Determine images not yet rated (this example uses an empty list for images_seen).
    images_seen = []  # Replace with your actual logic if available.
    images_not_seen = df_images[~df_images['img_path'].isin(images_seen)]
    print(f"User {user_temp} has seen {len(images_seen)} images and has {len(images_not_seen)} images left to rate.")
    
    if len(images_not_seen) == 0:
        continue
    
    # Loop over each image not rated.
    for image_path in images_not_seen['img_path'].values:
        df_personas_temp = df_personas_sample.loc[[user_temp]]
        attempt = 0
        valid_response_received = False

        while not valid_response_received:
            attempt += 1
            try:
                response_data = get_response_with_embeddings(user_temp, df_personas_temp, image_path, df_images)
                interesting_val = response_data['interestingness']
                explanation_val = response_data['explanation']
                
                if (interesting_val in ['Not Interesting', 'Slightly Interesting', 'Moderately Interesting', 'Very Interesting', 'Extremely Interesting']
                    and len(explanation_val) > 0):
                    valid_response_received = True
                    results_list.append(response_data)
                    counter += 1

                    if counter % 5 == 0:
                        print(f"Processed {counter} valid entries. Saving intermediate results...")
                        results_dict = {'results': results_list}
                        np.save('data/results-FAU_50.npy', results_dict)
                else:
                    print(f"Attempt {attempt}: Invalid interestingness/explanation. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt}: Error! {e}")
                # Optionally break or implement a retry limit.
                
                
# Save the final results as a NumPy dictionary file.
results_dict = {'results': results_list}
np.save('data/results-FAU_50.npy', results_dict)
print(f"All done. Processed a total of {counter} valid entries.")