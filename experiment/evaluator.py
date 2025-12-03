"""
Main evaluation pipeline for SSMS vs P-Metric comparison.

Evaluates images with both SSMS and P-Metric, saving results to CSVs.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import logging
import csv
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Import existing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import get_dataloader
from models.loader import load_model
from evaluation.judging.pytorch_judge import PyTorchJudgingModel
from attribution.registry import get_attribution_method
from core.file_manager import FileManager
from experiment.metrics_ssms import compute_ssms
from experiment.metrics_pmetric import compute_pmetric


class ExperimentEvaluator:
    """Main evaluator for SSMS vs P-Metric comparison."""
    
    def __init__(self, config):
        """
        Initialize evaluator.
        
        Args:
            config: ExperimentConfig instance
        """
        self.config = config
        self.file_manager = FileManager(config.base_dir)
        
        # Model caches
        self.generating_models = {}
        self.judge_models = {}
        self.attribution_methods = {}
        
        # Results storage
        self.pmetric_results = []
        self.ssms_results = []
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all required models and attribution methods."""
        logging.info("Loading models...")
        
        # Load generating models (for heatmap generation)
        for model_name in self.config.generating_models:
            try:
                model = load_model(model_name)
                self.generating_models[model_name] = model
                logging.info(f"Loaded generating model: {model_name}")
            except Exception as e:
                logging.error(f"Failed to load generating model {model_name}: {e}")
        
        # Load judge models (for evaluation)
        for model_name in self.config.judge_models:
            try:
                model = load_model(model_name)
                judge_model = PyTorchJudgingModel(model, model_name, device=self.config.device)
                self.judge_models[model_name] = judge_model
                logging.info(f"Loaded judge model: {model_name}")
            except Exception as e:
                logging.error(f"Failed to load judge model {model_name}: {e}")
        
        # Load attribution methods
        for method_name in self.config.explainers:
            try:
                method = get_attribution_method(method_name)
                self.attribution_methods[method_name] = method
                logging.info(f"Loaded attribution method: {method_name}")
            except Exception as e:
                logging.error(f"Failed to load attribution method {method_name}: {e}")
    
    def _load_or_generate_heatmap(
        self,
        image: torch.Tensor,
        generating_model: torch.nn.Module,
        explainer_name: str,
        true_label: int,
        image_id: str
    ) -> np.ndarray:
        """
        Load heatmap from disk if available, otherwise generate it.
        
        Args:
            image: Image tensor (C, H, W)
            generating_model: Model for generating heatmap
            explainer_name: Name of attribution method
            true_label: True class label
            image_id: Image identifier
            
        Returns:
            2D numpy array heatmap (H, W)
        """
        # Try to load from disk first (check if sorted heatmap exists)
        # Note: sorted heatmaps contain indices, not values, so we generate on-the-fly
        # But we can check if it exists to skip generation if already computed
        
        # For now, always generate heatmap on-the-fly for simplicity
        # (We need the actual heatmap values, not just sorted indices)
        
        attribution_method = self.attribution_methods[explainer_name]
        
        # Prepare inputs
        image_batch = image.unsqueeze(0).to(self.config.device)  # Add batch dimension
        target_batch = torch.tensor([true_label], device=self.config.device)
        
        # Generate heatmap
        with torch.set_grad_enabled(True):
            heatmap_tensor = attribution_method.compute(
                generating_model,
                image_batch,
                target_batch
            )
        
        # Convert to numpy and ensure 2D
        if isinstance(heatmap_tensor, torch.Tensor):
            heatmap = heatmap_tensor.squeeze().cpu().detach().numpy()
        else:
            heatmap = np.array(heatmap_tensor).squeeze()
        
        # Handle multi-channel heatmaps
        if heatmap.ndim == 3:
            heatmap = np.mean(heatmap, axis=0)
        elif heatmap.ndim > 2:
            heatmap = heatmap.squeeze()
        
        # Ensure 2D
        if heatmap.ndim != 2:
            raise ValueError(f"Heatmap should be 2D, got shape {heatmap.shape}")
        
        return heatmap
    
    def _evaluate_image(
        self,
        image_id: str,
        image: torch.Tensor,
        true_label: int
    ):
        """
        Evaluate single image with SSMS and P-Metric.
        
        Args:
            image_id: Image identifier
            image: Image tensor (C, H, W) - already normalized
            true_label: True class label
        """
        # Process each explainer × judge combination
        for explainer_name in self.config.explainers:
            # Get generating model (use first one)
            gen_model_name = self.config.generating_models[0]
            if gen_model_name not in self.generating_models:
                logging.warning(f"Generating model {gen_model_name} not loaded, skipping")
                continue
            generating_model = self.generating_models[gen_model_name]
            
            # Generate/load heatmap
            try:
                heatmap = self._load_or_generate_heatmap(
                    image, generating_model, explainer_name, true_label, image_id
                )
            except Exception as e:
                logging.warning(f"Failed to generate heatmap for {image_id} with {explainer_name}: {e}")
                continue
            
            # Evaluate with each judge model
            for judge_name, judge_model in self.judge_models.items():
                try:
                    # Compute SSMS with continuous differential masking
                    ssms_score, ssms_metadata, _ = compute_ssms(
                        heatmap,
                        image,
                        judge_model,
                        true_label,
                        alpha_max=self.config.alpha_max,
                        eps=self.config.eps,
                        power_factor=2.5,  # Higher power = stronger masking
                        sparsity_penalty_factor=3.0,  # Penalty for non-informative heatmaps
                        base_alpha=1.0
                    )
                    
                    # Compute P-Metric
                    pmetric_metrics, accuracy_curve = compute_pmetric(
                        heatmap,
                        image,
                        judge_model,
                        true_label,
                        self.config.occlusion_percents,
                        fill_strategy=self.config.fill_strategy
                    )
                    
                    # Store SSMS results
                    self.ssms_results.append({
                        'image_id': image_id,
                        'explainer': explainer_name,
                        'judge_model': judge_name,
                        'SSMS_score': ssms_score,
                        'S': ssms_metadata['S'],
                        'alpha': ssms_metadata['alpha'],
                        'entropy': ssms_metadata['entropy'],
                        'sparsity': ssms_metadata['sparsity']
                    })
                    
                    # Store P-Metric results
                    self.pmetric_results.append({
                        'image_id': image_id,
                        'explainer': explainer_name,
                        'judge_model': judge_name,
                        'AUC': pmetric_metrics['AUC'],
                        'DROP': pmetric_metrics['DROP'],
                        'InflectionPoint_max_slope': pmetric_metrics['InflectionPoint_max_slope'],
                        'InflectionPoint_threshold': pmetric_metrics['InflectionPoint_threshold']
                    })
                    
                except Exception as e:
                    logging.warning(f"Failed to evaluate {image_id} with {explainer_name}/{judge_name}: {e}")
                    continue
    
    def run(self):
        """Run evaluation on all images."""
        logging.info(f"Starting evaluation on {self.config.num_images} images")
        logging.info(f"Explainers: {self.config.explainers}")
        logging.info(f"Judge models: {list(self.judge_models.keys())}")
        
        # Load dataset
        dataloader = get_dataloader(
            self.config.dataset_name,
            batch_size=1,
            shuffle=False
        )
        
        # Process images
        processed = 0
        with tqdm(total=self.config.num_images, desc="Evaluating images") as pbar:
            try:
                for batch_images, batch_labels in dataloader:
                    if processed >= self.config.num_images:
                        break
                    
                    # Process each image in batch (batch_size=1, so single image)
                    image = batch_images[0]  # Remove batch dimension
                    true_label = batch_labels[0].item()
                    image_id = f"image_{processed:05d}"
                    
                    self._evaluate_image(image_id, image, true_label)
                    
                    processed += 1
                    pbar.update(1)
            except KeyboardInterrupt:
                logging.info("\nEvaluation interrupted by user. Saving partial results...")
            except Exception as e:
                logging.error(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        self._save_results()
        
        logging.info(f"Evaluation complete. Processed {processed} images.")
        logging.info(f"SSMS results: {len(self.ssms_results)} entries")
        logging.info(f"P-Metric results: {len(self.pmetric_results)} entries")
    
    def _save_results(self):
        """Save results to CSV files."""
        # Save SSMS results
        ssms_path = self.config.results_dir / "results_ssms.csv"
        ssms_header = ['image_id', 'explainer', 'judge_model', 'SSMS_score', 'S', 'alpha', 'entropy', 'sparsity']
        
        with open(ssms_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ssms_header)
            writer.writeheader()
            writer.writerows(self.ssms_results)
        
        logging.info(f"Saved SSMS results to {ssms_path}")
        
        # Save P-Metric results
        pmetric_path = self.config.results_dir / "results_pmetric.csv"
        pmetric_header = [
            'image_id', 'explainer', 'judge_model', 'AUC', 'DROP',
            'InflectionPoint_max_slope', 'InflectionPoint_threshold'
        ]
        
        with open(pmetric_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pmetric_header)
            writer.writeheader()
            writer.writerows(self.pmetric_results)
        
        logging.info(f"Saved P-Metric results to {pmetric_path}")

