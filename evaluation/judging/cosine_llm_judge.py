"""
Cosine Similarity LLM Judge - Open-ended with similarity matching.

Asks "What do you see?" and compares the answer to class names using
cosine similarity of embeddings. More flexible than exact matching.
"""
import logging
import numpy as np
from typing import Tuple
from pathlib import Path

import ollama

from evaluation.judging.base_llm_judge import BaseLLMJudge


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

    def _compute_similarity_to_class(self, response_embedding: np.ndarray, true_label: int) -> float:
        """
        Compute cosine similarity between LLM response embedding and the true class name.
        
        Args:
            response_embedding: Pre-computed embedding for the response text
            true_label: True class label index
            
        Returns:
            Cosine similarity score (0.0-1.0), or 0.0 if invalid label
        """
        # Validate true_label
        if true_label < 0 or true_label >= len(self.class_embeddings):
            logging.warning(f"Invalid true_label {true_label} (valid range: 0-{len(self.class_embeddings)-1})")
            return 0.0
        
        # Compute cosine similarity only with the true class
        true_class_embedding = self.class_embeddings[true_label]
        similarity = np.dot(true_class_embedding, response_embedding)
        
        return float(similarity)

    def _compute_similarity(self, response_embedding: np.ndarray) -> Tuple[int, float]:
        """
        Compute similarity between response embedding and all class names.
        
        Args:
            response_embedding: Pre-computed embedding for the response text
            
        Returns:
            Tuple of (best_class_idx, max_similarity)
        """
        if response_embedding is None:
            return -1, 0.0

        # Compute cosine similarities with all classes
        similarities = np.dot(self.class_embeddings, response_embedding)

        # Find best match
        best_idx = np.argmax(similarities)
        max_similarity = similarities[best_idx]

        return int(best_idx), float(max_similarity)

    def _predict_single_image_from_path(
            self,
            image_path: str,
            true_label: int,
            img_index: int
    ) -> Tuple[int, int, float]:
        """
        Predict class for a single image using open-ended question (optimized - uses file path directly).
        
        Args:
            image_path: Path to image file (PNG/JPG)
            true_label: True class label (for logging purposes)
            img_index: Original index in batch
            
        Returns:
            Tuple of (img_index, predicted_class_index, similarity_score)
        """
        try:
            # Ask open-ended question
            prompt = (
                "Look at this image carefully. What do you see? "
                "Describe the main object or subject in one or two words."
            )

            # Use shared retry helper method
            response_text = self._call_ollama_with_retry(
                prompt=prompt,
                image_path=image_path,
                max_retries=3,
                temperature=self.temperature
            )

            # Compute embedding ONCE for this response
            response_embedding = self._get_embedding(response_text)
            
            if response_embedding is None:
                logging.warning(f"Image {img_index}: Failed to compute embedding")
                return img_index, -1, 0.0
            
            # Step 1: Check similarity to TRUE class first
            true_class_similarity = self._compute_similarity_to_class(response_embedding, true_label)
            
            # Validate true_label for logging
            true_class_name = "invalid" if (true_label < 0 or true_label >= len(self.class_names)) else self.class_names[true_label]
            
            # Step 2: If true class similarity meets threshold, return it (skip computing all similarities)
            if true_class_similarity >= self.similarity_threshold:
                logging.info(
                    f"Image {img_index}: '{response_text}' -> "
                    f"True class match: {true_class_name}, similarity={true_class_similarity:.3f} (>= threshold)"
                )
                return img_index, true_label, true_class_similarity
            
            # Step 3: Otherwise, find best match among all classes (only if true class didn't pass)
            best_class, best_similarity = self._compute_similarity(response_embedding)
            
            if best_class < 0:
                logging.warning(
                    f"Image {img_index}: Failed to compute similarity. "
                    f"True class similarity={true_class_similarity:.3f}"
                )
                return img_index, -1, true_class_similarity
            
            best_class_name = self.class_names[best_class]
            
            logging.info(
                f"Image {img_index}: '{response_text}' -> "
                f"True class similarity={true_class_similarity:.3f} (< threshold), "
                f"Best match: Class {best_class} ({best_class_name}), similarity={best_similarity:.3f}, "
                f"real class is: {true_class_name}"
            )
            
            # Return best match (could be true class if it's the best, just below threshold)
            return img_index, best_class, best_similarity

        except Exception as e:
            logging.error(f"Error predicting image {img_index} with CosineSimilarityLLMJudge: {e}")
            return (img_index, -1, 0.0)
