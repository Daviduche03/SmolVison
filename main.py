from PIL import Image
import requests
import torch
import torch.nn as nn
from einops import rearrange
import time

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import base64
from PIL import Image
import io
from tqdm import tqdm
import wandb
import numpy as np
import requests
from questions import get_random_question


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, clip_processor, max_length=77):
        """
        Custom Dataset to handle paired image-text data.

        Args:
            hf_dataset (Dataset): The dataset object containing 'image' and 'text' fields.
            tokenizer (Tokenizer): Tokenizer for processing the text data.
            clip_processor (Processor): Processor for preparing the image for CLIP model.
            max_length (int): Maximum sequence length for tokenized text. Default is 77.
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def base64_to_image(image_data):
        """
        Converts base64 string, URL, or PIL Image to a PIL Image object.

        Args:
            image_data (str or PIL.Image.Image): Base64-encoded string, URL, or PIL Image.

        Returns:
            PIL.Image.Image: A PIL Image object.
        """
        if isinstance(image_data, str):
            if image_data.startswith("http"):  # Handle URL
                response = requests.get(image_data)
                response.raise_for_status()  # Ensure the request was successful
                image = Image.open(io.BytesIO(response.content))
            elif "base64" in image_data:  # Handle base64 string
                # Remove the base64 prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_data = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_data))
            else:
                raise ValueError("String input must be a valid URL or a base64-encoded image.")
        elif isinstance(image_data, Image.Image):  # Already a PIL Image
            image = image_data
        else:
            raise TypeError(f"Unsupported type for image_data: {type(image_data)}")

        return image.convert("RGB")  # Ensure consistent RGB format

    def __getitem__(self, idx):
        """
        Fetches an item from the dataset, processes the image and text, and returns the data.

        Args:
            idx (int): Index of the item in the dataset.

        Returns:
            dict: A dictionary with processed pixel values and tokenized text data.
        """
        item = self.dataset[idx]

        # Convert to PIL Image
        image = self.base64_to_image(item['url'])

        # Process image using CLIP processor
        clip_inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        )

        # Format and tokenize the text
        formatted_text = f"<user>{get_random_question()}<user/>\n\n<assistant>{item['caption']}<assistant/>"
        # print(formatted_text)
        text_inputs = self.tokenizer(
            formatted_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Normalize pixel values and ensure consistent shapes
        pixel_values = clip_inputs['pixel_values'].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'labels': text_inputs['input_ids'].squeeze(0)
        }


class CrossAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x_query, x_kv, mask=None):
        # x_query: text features to generate queries
        # x_kv: image features to generate keys and values
        queries = rearrange(self.q_proj(x_query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.k_proj(x_kv), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.v_proj(x_kv), "b n (h d) -> b h n d", h=self.num_heads)
        
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        energy = energy / (self.emb_size ** 0.5)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        att = torch.softmax(energy, dim=-1)
        att = self.att_drop(att)
        
        out = torch.einsum('bhqk, bhvd -> bhqd', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class VLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.llm = AutoModelForCausalLM.from_pretrained("Daviduche03/SmolLM2-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model dimensions
        self.clip_dim = self.clip_model.config.projection_dim
        self.llm_dim = self.llm.config.hidden_size

        # Projection layers
        self.clip_projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
            nn.GELU()
        )

        # Cross-attention layer
        self.cross_attention = CrossAttention(
            emb_size=self.llm_dim,
            num_heads=8
        )

        self.output_projection = nn.Linear(
            self.llm_dim,
            self.llm.config.vocab_size
        )

        self.layer_norm = nn.LayerNorm(self.llm_dim)
        self.to(self.device)

    def forward(self, text, image, text_only = False):
        # Process with CLIP - handle both tensor and PIL image inputs
        if isinstance(image, torch.Tensor):
            # If image is already processed (during training)
            clip_vision_outputs = self.clip_model.vision_model(pixel_values=image)
            image_features = self.clip_model.visual_projection(clip_vision_outputs[1])
        else:
            # If image is PIL (during inference)
            clip_inputs = self.clip_processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            clip_vision_outputs = self.clip_model.vision_model(pixel_values=clip_inputs['pixel_values'])
            image_features = self.clip_model.visual_projection(clip_vision_outputs[1])

        # Process text
        if isinstance(text, torch.Tensor):
            input_ids = text
        else:
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True
            ).input_ids.to(self.device)

        # Get LLM outputs
        if text_only:
            llm_outputs = self.llm(input_ids)
            logits = llm_outputs.logits
        else: 
            llm_outputs = self.llm(input_ids, output_hidden_states=True)
            llm_hidden = llm_outputs.hidden_states[-1]

            # Project CLIP image features
            clip_proj = self.clip_projection(image_features)
            clip_proj = clip_proj.unsqueeze(1)

            # Apply cross-attention
            attended = self.cross_attention(llm_hidden, clip_proj)
            attended = self.layer_norm(attended)

            # Project to vocabulary size
            logits = self.output_projection(attended)

        return logits

def train_model(model, dataset, num_epochs=10, batch_size=8, learning_rate=5e-5):
    # Initialize wandb
    wandb.init(project="vlm-training", name="cross-attention-run")
    
    # Create dataset and dataloader
    train_dataset = ImageTextDataset(
        dataset, 
        model.tokenizer, 
        model.clip_processor
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    device = model.device
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(
                text=batch['input_ids'],
                image=batch['pixel_values']
            )
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                outputs.view(-1, model.llm.config.vocab_size),
                batch['labels'].view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/(progress_bar.n+1):.4f}"
            })
            
            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'vlm_checkpoint_epoch_{epoch+1}.pt')
            
        wandb.log({
            "epoch": epoch,
            "epoch_loss": epoch_loss / len(train_dataloader)
        })

    wandb.finish()
    return model

from datasets import load_dataset

# Load your dataset
dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS")
num_of_samples = 5000
dataset = dataset["train"].select(range(num_of_samples))  # Assuming 'train' split

# Initialize model
model = VLM()

# Train
model = train_model(
    model,
    dataset,
    num_epochs=35,
    batch_size=8,
    learning_rate=5e-5
)