"""
Cosine Similarity LLM Judge - Open-ended with similarity matching.

Asks "What do you see?" and compares the answer to class names using
cosine similarity of embeddings. More flexible than exact matching.
"""

import ollama
import torch
import numpy as np
from typing import List, Tuple
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation.judging.base_llm_judge import BaseLLMJudge, MAX_PARALLEL_WORKERS

# Silence httpx logging from ollama
logging.getLogger("httpx").setLevel(logging.WARNING)


class CosineSimilarityLLMJudge(BaseLLMJudge):
    """
    Cosine Similarity LLM Judge using embeddings.
    
    Asks "What do you see?" and computes cosine similarity between
    the response and class names. Returns the class with highest similarity
    if above threshold.
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str = "imagenet",
        temperature: float = 0.1,
        similarity_threshold: float = 0.8,
        embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize Cosine Similarity LLM Judge.
        
        Args:
            model_name: Ollama vision model name (e.g., "llama3.2-vision")
            dataset_name: Dataset name to get class names from
            temperature: Temperature for LLM (0.0 = deterministic)
            similarity_threshold: Minimum cosine similarity to accept (0.0-1.0)
            embedding_model: Ollama embedding model name
        """
        super().__init__(model_name, dataset_name)
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        
        # Load or compute class name embeddings
        self.class_embeddings = self._load_or_compute_embeddings()
        
        logging.info(
            f"CosineSimilarityLLMJudge initialized: {len(self.class_names)} classes, "
            f"threshold={similarity_threshold}, temperature={temperature}"
        )
    
    
    def _load_or_compute_embeddings(self) -> np.ndarray:
        """
        Load embeddings from cache or compute if not cached.
        
        Returns:
            Array of embeddings (shape: [num_classes, embedding_dim])
        """
        # Create cache directory
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Cache file path
        cache_file = cache_dir / f"embeddings_{self.dataset_name}_{self.embedding_model.replace('/', '_')}.npy"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                embeddings = np.load(cache_file)
                logging.info(f"Loaded cached embeddings from {cache_file}")
                return embeddings
            except Exception as e:
                logging.warning(f"Failed to load cached embeddings: {e}. Recomputing...")
        
        # Compute fresh embeddings
        logging.info(f"Computing embeddings for {len(self.class_names)} classes (this may take 2-3 minutes)...")
        embeddings = self._compute_class_embeddings()
        
        # Save to cache
        try:
            np.save(cache_file, embeddings)
            logging.info(f"Saved embeddings to cache: {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to save embeddings to cache: {e}")
        
        return embeddings
    
    def _compute_class_embeddings(self) -> np.ndarray:
        """
        Pre-compute embeddings for all class names.
        
        Returns:
            Array of embeddings (shape: [num_classes, embedding_dim])
        """
        formatted_names = [self._format_class_name(name) for name in self.class_names]
        
        try:
            # Use Ollama's embedding API
            embeddings = []
            
            # Process in batches for efficiency
            batch_size = 100
            for i in range(0, len(formatted_names), batch_size):
                batch = formatted_names[i:i + batch_size]
                
                for text in batch:
                    response = ollama.embeddings(
                        model=self.embedding_model,
                        prompt=text
                    )
                    embeddings.append(response['embedding'])
            
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
            return embeddings
        
        except Exception as e:
            logging.error(f"Failed to compute embeddings: {e}")
            logging.error(f"Make sure Ollama embedding model '{self.embedding_model}' is available")
            raise
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Normalized embedding vector
        """
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            embedding = np.array(response['embedding'], dtype=np.float32)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            embedding = embedding / (norm + 1e-8)
            
            return embedding
        except Exception as e:
            logging.error(f"Failed to get embedding: {e}")
            return None
    
    def _compute_similarity(self, response_text: str) -> Tuple[int, float]:
        """
        Compute similarity between response and all class names.
        
        Args:
            response_text: LLM response text
            
        Returns:
            Tuple of (best_class_idx, max_similarity)
        """
        # Get embedding for response
        response_embedding = self._get_embedding(response_text)
        
        if response_embedding is None:
            return -1, 0.0
        
        # Compute cosine similarities with all classes
        similarities = np.dot(self.class_embeddings, response_embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        max_similarity = similarities[best_idx]
        
        return int(best_idx), float(max_similarity)
    
    def _predict_single_image(
        self,
        img_tensor: torch.Tensor,
        img_index: int
    ) -> Tuple[int, int, float]:
        """
        Predict class for a single image using open-ended question.
        
        Args:
            img_tensor: Image tensor (C, H, W)
            img_index: Original index in batch
            
        Returns:
            Tuple of (img_index, predicted_class_index, similarity_score)
        """
        try:
            temp_image_path = self._tensor_to_temp_file(img_tensor)
            
            try:
                # Ask open-ended question
                prompt = (
                    "Look at this image carefully. What do you see? "
                    "Describe the main object or subject in one or two words."
                )
                
                # Call Ollama API
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [temp_image_path]
                        }
                    ],
                    options={
                        'temperature': self.temperature,
                    }
                )
                
                response_text = response.message.content.strip()
                
                # Compute similarity with class names
                best_class, similarity = self._compute_similarity(response_text)
                
                logging.debug(
                    f"Image {img_index}: '{response_text}' -> "
                    f"Class {best_class} ({self.class_names[best_class]}), "
                    f"similarity={similarity:.3f}"
                )
                
                # Check if similarity meets threshold
                if similarity >= self.similarity_threshold:
                    return (img_index, best_class, similarity)
                else:
                    # Below threshold - no confident match
                    logging.debug(
                        f"Image {img_index}: Similarity {similarity:.3f} below threshold "
                        f"{self.similarity_threshold}, returning -1"
                    )
                    return (img_index, -1, similarity)
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass
        
        except Exception as e:
            logging.error(f"Error predicting image {img_index} with CosineSimilarityLLMJudge: {e}")
            return (img_index, -1, 0.0)
    
    def predict(self, images: List[torch.Tensor], **kwargs) -> np.ndarray:
        """
        Predict classes for given images using open-ended questions.
        
        Args:
            images: List of image tensors (C, H, W) - normalized ImageNet format
            **kwargs: Additional parameters
            
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        if len(images) == 0:
            return np.array([], dtype=np.int64)
        
        # Process images in parallel
        max_workers = min(MAX_PARALLEL_WORKERS, len(images))
        predictions = [None] * len(images)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._predict_single_image, img, idx): idx
                for idx, img in enumerate(images)
            }
            
            for future in as_completed(future_to_idx):
                try:
                    img_idx, class_idx, similarity = future.result()
                    predictions[img_idx] = class_idx
                except Exception as e:
                    idx = future_to_idx[future]
                    logging.error(f"Unexpected error processing image {idx}: {e}")
                    predictions[idx] = -1
        
        return np.array(predictions, dtype=np.int64)

