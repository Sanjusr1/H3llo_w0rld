from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

# Load models lazily
script_generator = None
scene_tokenizer = None
scene_model = None

def get_script_generator():
    global script_generator
    if script_generator is None:
        script_generator = pipeline("text-generation", model="gpt2")
    return script_generator

def get_scene_tokenizer():
    global scene_tokenizer
    if scene_tokenizer is None:
        scene_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return scene_tokenizer

def get_scene_model():
    global scene_model
    if scene_model is None:
        scene_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return scene_model

def compute_cognitive_load(behavior, face):
    score = 0
    score += behavior["pause_time"] * 0.5
    score += behavior["rereads"] * 10

    if face["blink_rate"] > 20:
        score += 15
    if face["eyebrow_movements"] > 10:
        score += 10
    if face["head_movements"] > 12:
        score += 10

    if score > 60:
        return "high"
    elif score > 30:
        return "medium"
    return "low"

def map_parameters(load):
    if load == "high":
        return 0.2, "slow"
    elif load == "medium":
        return 0.4, "normal"
    return 0.6, "fast"

def generate_script(text, temperature):
    generator = get_script_generator()
    prompt = f"Explain this concept simply:\n{text}\nExplanation:"
    result = generator(prompt, max_length=150, temperature=temperature)
    return result[0]["generated_text"]

def generate_scenes(script):
    tokenizer = get_scene_tokenizer()
    model = get_scene_model()
    prompt = f"""
Convert this explanation into short visual scene descriptions.
Only describe what is visible.

{script}

Scene 1:
Scene 2:
Scene 3:
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=300, num_beams=4, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
