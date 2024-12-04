import torch
from PIL import Image
import base64
import io
import requests
from main import VLM

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        raise ValueError("Invalid base64 string") from e

def load_image_from_url(image_url):
    """Load an image from a URL."""
    if not image_url.startswith("http"):
        raise ValueError("Invalid URL format.")
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise RuntimeError("Failed to load image from URL") from e

def preprocess_image(image_input):
    """Handle image input from various formats."""
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            return load_image_from_url(image_input)
        return base64_to_image(image_input)
    elif isinstance(image_input, Image.Image):
        return image_input
    raise ValueError("Invalid image input format.")

def generate_response(
    model, image, question, max_length=100, temperature=0.9, top_p=0.9, top_k=50, debug=False
):
    """Generate a response for a given image and question."""
    model.eval()
    device = model.device

    image = preprocess_image(image)
    prompt = f"<user>{question}<user/>"

    encoding = model.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True
    ).to(device)

    clip_inputs = model.clip_processor(
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    generated = encoding.input_ids
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(text=generated, image=clip_inputs['pixel_values'])
            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                mask = torch.zeros_like(next_token_logits).scatter_(1, top_k_indices, 1.0)
                next_token_logits = torch.where(mask > 0, next_token_logits, -float('inf'))

            # Sample token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if debug:
                token_str = model.tokenizer.decode([next_token.item()])
                prob = probs[0, next_token.item()].item()
                print(f"Selected token: {token_str}, Probability: {prob:.4f}")

            # Stop if EOS token is generated
            if next_token.item() == model.tokenizer.eos_token_id and generated.shape[1] > 5:
                break

            generated = torch.cat([generated, next_token], dim=-1)

    return model.tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    """Main function to load model and run examples."""
    model = VLM()

    checkpoint = torch.load('vlm_checkpoint_epoch_20.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    image_url = "http://images.cocodataset.org/val2017/000000174482.jpg"
    image = load_image_from_url(image_url)

    questions = ["Can you tell me what this is?"]
    for question in questions:
        response = generate_response(model, image, question, max_length=100, temperature=0.7, top_p=0.9)
        print(f"\nQ: {question}")
        print(f"A: {response}")

def interactive_mode(model):
    """Interactive mode to ask questions about an image."""
    image_path = input("Enter image path or URL: ")
    image = preprocess_image(image_path)

    print("\nYou can now ask questions. Type 'exit' to quit.\n")
    while True:
        question = input("Your question: ")
        if question.lower() == 'exit':
            break
        response = generate_response(model, image, question)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main()
