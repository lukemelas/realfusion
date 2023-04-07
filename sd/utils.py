from typing import Optional, Union, Mapping

import torch
from transformers.models.clip import CLIPTextModel, CLIPTokenizer
from torch import Tensor


def add_tokens_to_model(learned_embeds: Mapping[str, Tensor], text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, override_token: Optional[Union[str, dict]] = None) -> None:
    r"""Adds tokens to the tokenizer and text encoder of a model."""
    
    # Loop over learned embeddings
    new_tokens = []
    for token, embedding in learned_embeds.items():
        embedding = embedding.to(text_encoder.get_input_embeddings().weight.dtype)
        if override_token is not None:
            token = override_token if isinstance(override_token, str) else override_token[token]
        
        # Add the token to the tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError((f"The tokenizer already contains the token {token}. Please pass a "
                               "different `token` that is not already in the tokenizer."))
  
        # Resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
  
        # Get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embedding    
        new_tokens.append(token)

    print(f'Added {len(new_tokens)} tokens to tokenizer and text embedding: {new_tokens}')


def add_tokens_to_model_from_path(learned_embeds_path: str, text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, override_token: Optional[Union[str, dict]] = None) -> None:
    r"""Loads tokens from a file and adds them to the tokenizer and text encoder of a model."""
    learned_embeds: Mapping[str, Tensor] = torch.load(learned_embeds_path, map_location='cpu')
    add_tokens_to_model(learned_embeds, text_encoder, tokenizer, override_token)
