"""
AIML-Compliant Multi-Task Classification Heads
Implements the exact 6 tasks from the AIML paper with joint probability optimization
Optimized for Qdrant Free Tier constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
try:
    from .memory_manager import get_memory_manager
except ImportError:
    from memory_manager import get_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskSpecification:
    """Specification for each AIML-compliant task"""
    name: str
    num_classes: int
    class_names: List[str]
    target_f1: float  # Target F1-score from AIML paper
    weight: float  # Task weight in joint optimization

class AIMLMultiTaskHeads(nn.Module):
    """
    Multi-task classification heads implementing AIML paper specifications
    6 specific tasks with joint probability optimization for task correlation
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        """
        Initialize AIML-compliant multi-task heads

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.memory_manager = get_memory_manager()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # AIML Paper: 6 specific classification tasks
        self.task_specs = {
            'driving_status': TaskSpecification(
                name='driving_status',
                num_classes=5,
                class_names=['driving', 'parked', 'stopped', 'passenger', 'unknown'],
                target_f1=0.93,  # From AIML paper
                weight=1.2  # Higher weight due to high F1
            ),
            'accident_type': TaskSpecification(
                name='accident_type',
                num_classes=12,
                class_names=[
                    'collision', 'rollover', 'side_impact', 'rear_end', 'head_on',
                    'single_vehicle', 'multi_vehicle', 'pedestrian', 'animal', 'object',
                    'parking_lot', 'other'
                ],
                target_f1=0.84,  # From AIML paper
                weight=1.0
            ),
            'road_type': TaskSpecification(
                name='road_type',
                num_classes=11,
                class_names=[
                    'highway', 'urban', 'rural', 'parking', 'intersection',
                    'residential', 'commercial', 'industrial', 'bridge', 'tunnel', 'other'
                ],
                target_f1=0.79,  # From AIML paper
                weight=0.9
            ),
            'cause_accident': TaskSpecification(
                name='cause_accident',
                num_classes=11,
                class_names=[
                    'negligence', 'weather', 'mechanical', 'medical', 'intentional',
                    'distraction', 'fatigue', 'impaired', 'speed', 'road_condition', 'other'
                ],
                target_f1=0.85,  # From AIML paper
                weight=1.1
            ),
            'vehicle_count': TaskSpecification(
                name='vehicle_count',
                num_classes=4,
                class_names=['single', 'two', 'multiple', 'unknown'],
                target_f1=0.94,  # From AIML paper
                weight=1.3  # Highest weight due to best F1
            ),
            'parties_involved': TaskSpecification(
                name='parties_involved',
                num_classes=5,
                class_names=['single', 'two', 'multiple', 'pedestrian', 'property_only'],
                target_f1=0.89,  # From AIML paper (interpolated)
                weight=1.0
            )
        }

        # Task correlation matrix (learned)
        self.task_correlation = nn.Parameter(
            torch.eye(len(self.task_specs)) + 0.1 * torch.randn(len(self.task_specs), len(self.task_specs))
        )

        # Create shared feature transformation
        self.shared_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_spec in self.task_specs.items():
            self.task_heads[task_name] = self._create_task_head(task_spec)

        # Task uncertainty weights (learned for joint optimization)
        self.task_uncertainties = nn.Parameter(torch.zeros(len(self.task_specs)))

        # Initialize weights
        self._initialize_weights()

    def _create_task_head(self, task_spec: TaskSpecification) -> nn.Module:
        """
        Create a task-specific classification head

        Args:
            task_spec: Task specification

        Returns:
            Task-specific neural network head
        """
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, task_spec.num_classes)
        )

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Forward pass through all task heads

        Args:
            x: Input features [batch_size, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with task predictions and metadata
        """
        batch_size = x.size(0)

        # Shared feature transformation
        shared_features = self.shared_transform(x)

        # Task-specific predictions
        task_outputs = {}
        task_logits_dict = {}
        task_probabilities_dict = {}

        for task_name, task_head in self.task_heads.items():
            # Task-specific logits
            logits = task_head(shared_features)
            task_logits_dict[task_name] = logits

            # Apply temperature scaling for better calibration
            temperature = torch.max(torch.exp(self.task_uncertainties[self._get_task_index(task_name)]),
                                   torch.tensor(0.1))
            scaled_logits = logits / temperature

            # Compute probabilities
            probabilities = F.softmax(scaled_logits, dim=-1)
            task_probabilities_dict[task_name] = probabilities

            task_outputs[task_name] = {
                'logits': logits,
                'probabilities': probabilities,
                'predictions': torch.argmax(probabilities, dim=-1),
                'confidence': torch.max(probabilities, dim=-1)[0]
            }

        # Joint probability optimization using task correlations
        joint_probabilities = self._compute_joint_probabilities(task_probabilities_dict)

        # Add joint optimization to outputs
        for task_name in task_outputs:
            task_idx = self._get_task_index(task_name)
            if task_idx < joint_probabilities.size(0):
                task_outputs[task_name]['joint_probabilities'] = joint_probabilities[task_idx]
            else:
                # Fallback if index out of bounds
                task_outputs[task_name]['joint_probabilities'] = torch.zeros(batch_size, device=x.device)
            task_outputs[task_name]['task_weight'] = self.task_specs[task_name].weight

        return task_outputs

    def _compute_joint_probabilities(self, task_probabilities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute joint probabilities using task correlation matrix

        Args:
            task_probabilities: Dictionary of task probabilities

        Returns:
            Joint probabilities tensor
        """
        num_tasks = len(self.task_specs)
        batch_size = next(iter(task_probabilities.values())).size(0)

        # Create joint probability matrix
        joint_matrix = torch.zeros(batch_size, num_tasks, device=task_probabilities['driving_status'].device)

        # Apply task correlations
        task_list = list(self.task_specs.keys())
        for i, task_i in enumerate(task_list):
            if task_i in task_probabilities:
                prob_i = task_probabilities[task_i]  # [batch_size, num_classes_i]

                # Average probability across classes as task representation
                task_repr_i = torch.mean(prob_i, dim=-1)  # [batch_size]

                # Apply correlations with other tasks
                correlated_score = task_repr_i.clone()

                for j, task_j in enumerate(task_list):
                    if i != j and task_j in task_probabilities:
                        prob_j = task_probabilities[task_j]
                        task_repr_j = torch.mean(prob_j, dim=-1)

                        # Apply learned correlation
                        correlation_weight = torch.sigmoid(self.task_correlation[i, j])
                        correlated_score += correlation_weight * task_repr_j

                joint_matrix[:, i] = correlated_score

        return joint_matrix

    def _get_task_index(self, task_name: str) -> int:
        """Get index of task in correlation matrix"""
        task_list = list(self.task_specs.keys())
        return task_list.index(task_name)

    def compute_loss(self, predictions: Dict[str, Dict[str, Any]],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with joint optimization

        Args:
            predictions: Model predictions for each task
            targets: Target labels for each task

        Returns:
            Dictionary with loss components
        """
        total_loss = 0.0
        task_losses = {}
        weighted_losses = {}

        for task_name, task_spec in self.task_specs.items():
            if task_name in predictions and task_name in targets:
                logits = predictions[task_name]['logits']
                target = targets[task_name]

                # Standard cross-entropy loss
                task_loss = F.cross_entropy(logits, target)
                task_losses[task_name] = task_loss

                # Apply task weight and uncertainty
                task_idx = self._get_task_index(task_name)
                task_weight = task_spec.weight
                uncertainty = torch.exp(self.task_uncertainties[task_idx])

                # Weighted loss with uncertainty regularization
                weighted_loss = task_weight * task_loss + 0.1 * uncertainty
                weighted_losses[task_name] = weighted_loss

                total_loss += weighted_loss

        # Add task correlation regularization loss
        correlation_loss = self._compute_correlation_loss()
        total_loss += 0.01 * correlation_loss

        loss_dict = {
            'total_loss': total_loss,
            'task_losses': task_losses,
            'weighted_losses': weighted_losses,
            'correlation_loss': correlation_loss,
            'mean_task_loss': torch.mean(torch.stack(list(task_losses.values()))) if task_losses else torch.tensor(0.0)
        }

        return loss_dict

    def _compute_correlation_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for task correlation matrix
        """
        # Encourage reasonable correlation values
        correlation_matrix = torch.sigmoid(self.task_correlation)

        # Penalize extreme correlations
        identity_penalty = torch.mean(torch.abs(correlation_matrix - torch.eye(correlation_matrix.size(0), device=correlation_matrix.device)))

        return identity_penalty

    def predict(self, x: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """
        Make predictions with confidence scores and top-K alternatives

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary with detailed predictions
        """
        self.eval()
        with torch.no_grad():
            raw_predictions = self.forward(x)

            detailed_predictions = {}

            for task_name, task_spec in self.task_specs.items():
                if task_name in raw_predictions:
                    raw_pred = raw_predictions[task_name]

                    # Get top-K predictions
                    probabilities = raw_pred['probabilities']
                    top_k_values, top_k_indices = torch.topk(probabilities, k=min(3, task_spec.num_classes), dim=-1)

                    # Convert to class names
                    batch_size = x.size(0)
                    batch_predictions = []

                    for batch_idx in range(batch_size):
                        pred_info = {
                            'predicted_class': task_spec.class_names[raw_pred['predictions'][batch_idx].item()],
                            'confidence': raw_pred['confidence'][batch_idx].item(),
                            'top_k_predictions': [
                                {
                                    'class_name': task_spec.class_names[idx.item()],
                                    'confidence': conf.item()
                                }
                                for conf, idx in zip(top_k_values[batch_idx], top_k_indices[batch_idx])
                            ],
                            'all_probabilities': {
                                task_spec.class_names[i]: prob.item()
                                for i, prob in enumerate(probabilities[batch_idx])
                            },
                            'joint_score': raw_pred['joint_probabilities'][batch_idx].item(),
                            'task_weight': task_spec.weight,
                            'target_f1': task_spec.target_f1
                        }
                        batch_predictions.append(pred_info)

                    detailed_predictions[task_name] = {
                        'predictions': batch_predictions,
                        'class_names': task_spec.class_names,
                        'num_classes': task_spec.num_classes
                    }

            return detailed_predictions

    def extract_structured_features(self, predictions: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Extract structured features from predictions for downstream use

        Args:
            predictions: Model predictions

        Returns:
            Structured feature tensor
        """
        features = []

        # Process each task
        for task_name, task_spec in self.task_specs.items():
            if task_name in predictions:
                task_pred = predictions[task_name]

                # One-hot encode primary predictions
                for batch_pred in task_pred['predictions']:
                    # One-hot encoding
                    one_hot = [0.0] * task_spec.num_classes
                    pred_class = batch_pred['predicted_class']
                    if pred_class in task_spec.class_names:
                        class_idx = task_spec.class_names.index(pred_class)
                        one_hot[class_idx] = 1.0

                    # Add confidence scores
                    confidence_features = [batch_pred['confidence']]

                    # Add top-K probabilities
                    top_k_probs = [pred['confidence'] for pred in batch_pred['top_k_predictions'][:2]]
                    while len(top_k_probs) < 2:
                        top_k_probs.append(0.0)

                    # Add joint score
                    joint_score = [batch_pred['joint_score']]

                    # Combine features for this task
                    task_features = one_hot + confidence_features + top_k_probs + joint_score
                    features.extend(task_features)

        return torch.tensor(features, dtype=torch.float32)

    def get_task_statistics(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about task predictions

        Args:
            predictions: Model predictions

        Returns:
            Dictionary with task statistics
        """
        stats = {
            'task_count': len(self.task_specs),
            'task_stats': {},
            'overall_confidence': [],
            'joint_scores': []
        }

        for task_name, task_spec in self.task_specs.items():
            if task_name in predictions:
                task_pred = predictions[task_name]['predictions']

                confidences = [pred['confidence'] for pred in task_pred]
                joint_scores = [pred['joint_score'] for pred in task_pred]

                stats['task_stats'][task_name] = {
                    'mean_confidence': np.mean(confidences),
                    'mean_joint_score': np.mean(joint_scores),
                    'target_f1': task_spec.target_f1,
                    'task_weight': task_spec.weight,
                    'num_samples': len(task_pred)
                }

                stats['overall_confidence'].extend(confidences)
                stats['joint_scores'].extend(joint_scores)

        # Overall statistics
        if stats['overall_confidence']:
            stats['overall_mean_confidence'] = np.mean(stats['overall_confidence'])
            stats['overall_mean_joint_score'] = np.mean(stats['joint_scores'])

        return stats

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information
        """
        memory_info = self.memory_manager.check_memory_usage()

        # Calculate model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'component': 'aiml_multi_task_heads',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'task_count': len(self.task_specs),
            'memory_usage': memory_info,
            'input_dimension': self.input_dim,
            'hidden_dimension': self.hidden_dim
        }

# Global instance
_aiml_heads = None

def get_aiml_multi_task_heads(input_dim: int = 128, hidden_dim: int = 64) -> AIMLMultiTaskHeads:
    """Get or create AIML multi-task heads instance"""
    global _aiml_heads
    if _aiml_heads is None or _aiml_heads.input_dim != input_dim:
        _aiml_heads = AIMLMultiTaskHeads(input_dim=input_dim, hidden_dim=hidden_dim)
    return _aiml_heads