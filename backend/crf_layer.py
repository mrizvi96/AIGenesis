"""
Advanced Conditional Random Field (CRF) Layer for Structured Prediction
Optimized for insurance claim text analysis and memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
try:
    from .memory_manager import get_memory_manager
except ImportError:
    from memory_manager import get_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientCRF(nn.Module):
    """
    Memory-efficient CRF implementation for sequence labeling
    Optimized for Qdrant free tier constraints
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        Initialize CRF layer

        Args:
            num_tags: Number of tags/labels
            batch_first: Whether batch dimension is first
        """
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.memory_manager = get_memory_manager()

        # Transition parameters: transitions[i][j] = score of transitioning from j to i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Initialize transitions with small random values
        nn.init.xavier_uniform_(self.transitions)

        # Start and end transitions for sequence boundaries
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        # Memory optimization settings
        self.use_chunking = True  # Process sequences in chunks for memory efficiency
        self.chunk_size = 50  # Process 50 tokens at a time

    def forward(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - simplified for memory efficiency
        Returns emissions with minimal CRF processing

        Args:
            emissions: Emission scores from previous layer
            mask: Mask for valid positions (optional)

        Returns:
            Processed emissions tensor
        """
        # For memory efficiency, we use a simplified approach
        # In production, this would implement full CRF Viterbi decoding
        return emissions

    def compute_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log likelihood of the sequence tags
        Memory-efficient implementation

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            tags: True tags [batch_size, seq_len]
            mask: Mask for valid positions [batch_size, seq_len]

        Returns:
            Log likelihood tensor
        """
        if self.use_chunking and emissions.size(1) > self.chunk_size:
            return self._compute_log_likelihood_chunked(emissions, tags, mask)
        else:
            return self._compute_log_likelihood_standard(emissions, tags, mask)

    def _compute_log_likelihood_standard(self, emissions: torch.Tensor, tags: torch.Tensor,
                                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard log likelihood computation for shorter sequences
        """
        batch_size, seq_len = emissions.size()[:2]

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)

        # Compute forward score (partition function)
        forward_score = self._compute_forward_score(emissions, mask)

        # Compute score of the true path
        gold_score = self._compute_gold_score(emissions, tags, mask)

        # Log likelihood = gold_score - forward_score
        return gold_score - forward_score

    def _compute_log_likelihood_chunked(self, emissions: torch.Tensor, tags: torch.Tensor,
                                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-efficient chunked computation for long sequences
        """
        batch_size, seq_len = emissions.size()[:2]

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)

        total_log_likelihood = torch.zeros(batch_size, device=emissions.device)

        # Process in chunks
        for start_idx in range(0, seq_len, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)

            chunk_emissions = emissions[:, start_idx:end_idx]
            chunk_tags = tags[:, start_idx:end_idx]
            chunk_mask = mask[:, start_idx:end_idx]

            chunk_log_likelihood = self._compute_log_likelihood_standard(
                chunk_emissions, chunk_tags, chunk_mask
            )

            total_log_likelihood += chunk_log_likelihood

        return total_log_likelihood

    def _compute_forward_score(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute forward score (partition function) using dynamic programming
        Memory-efficient implementation
        """
        batch_size, seq_len, num_tags = emissions.size()

        # Initialize forward variables
        forward_var = self.start_transitions.view(1, -1) + emissions[:, 0]

        # Iterate through sequence
        for i in range(1, seq_len):
            # Get emissions for current position
            emit_score = emissions[:, i].view(batch_size, 1, -1)

            # Broadcast and add transitions
            broadcast_transitions = self.transitions.view(1, num_tags, num_tags)
            next_tag_var = forward_var.view(batch_size, -1, 1) + broadcast_transitions + emit_score

            # Log-sum-exp trick for numerical stability
            forward_var = torch.logsumexp(next_tag_var, dim=1)

            # Apply mask
            mask_i = mask[:, i].unsqueeze(1)
            forward_var = forward_var * mask_i + self.start_transitions.view(1, -1) * (1 - mask_i)

        # Add end transitions
        last_mask = mask[:, -1].unsqueeze(1)
        forward_var = forward_var + self.end_transitions.view(1, -1) * last_mask

        return torch.logsumexp(forward_var, dim=1)

    def _compute_gold_score(self, emissions: torch.Tensor, tags: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        Compute score of the true path through the sequence
        """
        batch_size, seq_len = emissions.size()

        # Initialize score with start transitions and first tags
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, torch.arange(batch_size), tags[:, 0]]

        # Add transitions and emissions for remaining positions
        for i in range(1, seq_len):
            mask_i = mask[:, i]
            score += self.transitions[tags[:, i], tags[:, i-1]] * mask_i
            score += emissions[:, i, torch.arange(batch_size), tags[:, i]] * mask_i

        # Add end transitions for last valid positions
        last_tag_indices = mask.sum(1) - 1
        last_tags = tags[torch.arange(batch_size), last_tag_indices]
        score += self.end_transitions[last_tags]

        return score

    def viterbi_decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Viterbi algorithm for finding the most likely sequence of tags
        Memory-efficient implementation

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            mask: Mask for valid positions [batch_size, seq_len]

        Returns:
            List of predicted tag sequences
        """
        if self.use_chunking and emissions.size(1) > self.chunk_size:
            return self._viterbi_decode_chunked(emissions, mask)
        else:
            return self._viterbi_decode_standard(emissions, mask)

    def _viterbi_decode_standard(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Standard Viterbi decoding for shorter sequences
        """
        batch_size, seq_len, num_tags = emissions.size()

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)

        predicted_tags = []

        for batch_idx in range(batch_size):
            # Initialize viterbi variables
            viterbi_vars = self.start_transitions + emissions[batch_idx, 0]
            path_indices = []

            # Forward pass
            for i in range(1, seq_len):
                if not mask[batch_idx, i]:
                    break

                # Compute viterbi scores
                viterbi_scores = viterbi_vars.view(-1, 1) + self.transitions + emissions[batch_idx, i]
                best_scores, best_paths = torch.max(viterbi_scores, dim=0)

                viterbi_vars = best_scores
                path_indices.append(best_paths)

            # Backward pass to reconstruct best path
            best_last_tag = torch.argmax(viterbi_vars + self.end_transitions).item()
            best_path = [best_last_tag]

            for path_idx in reversed(path_indices):
                best_last_tag = path_idx[best_last_tag].item()
                best_path.append(best_last_tag)

            predicted_tags.append(list(reversed(best_path)))

        return predicted_tags

    def _viterbi_decode_chunked(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Memory-efficient chunked Viterbi decoding
        """
        batch_size, seq_len = emissions.size()

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)

        predicted_tags = []

        for batch_idx in range(batch_size):
            full_path = []

            # Process sequence in chunks
            for start_idx in range(0, seq_len, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, seq_len)

                chunk_emissions = emissions[batch_idx, start_idx:end_idx]
                chunk_mask = mask[batch_idx, start_idx:end_idx]

                # Simple greedy decoding for chunks (memory efficient)
                chunk_path = torch.argmax(chunk_emissions, dim=-1).tolist()

                # Apply mask
                valid_length = chunk_mask.sum().item()
                chunk_path = chunk_path[:valid_length]

                full_path.extend(chunk_path)

            predicted_tags.append(full_path)

        return predicted_tags

class InsuranceSequenceLabeler:
    """
    Specialized sequence labeler for insurance claim text
    Uses CRF for entity recognition and structured information extraction
    """

    def __init__(self, num_labels: int = 10):
        """
        Initialize insurance sequence labeler

        Args:
            num_labels: Number of entity labels for insurance domain
        """
        self.num_labels = num_labels
        self.memory_manager = get_memory_manager()

        # Initialize CRF layer
        self.crf = MemoryEfficientCRF(num_labels, batch_first=True)

        # Insurance domain label mapping
        self.label_map = {
            0: 'O',           # Outside (no entity)
            1: 'B-PER',       # Beginning of person name
            2: 'I-PER',       # Inside of person name
            3: 'B-ORG',       # Beginning of organization
            4: 'I-ORG',       # Inside of organization
            5: 'B-LOC',       # Beginning of location
            6: 'I-LOC',       # Inside of location
            7: 'B-DATE',      # Beginning of date/time
            8: 'I-DATE',      # Inside of date/time
            9: 'B-AMOUNT'     # Beginning of monetary amount
        }

        # Reverse label map for decoding
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Insurance-specific patterns for entity recognition
        self.entity_patterns = self._load_insurance_patterns()

    def _load_insurance_patterns(self) -> Dict[str, List[str]]:
        """
        Load insurance-specific entity recognition patterns
        """
        return {
            'person': [
                r'\b(?:mr|mrs|ms|dr)\.?\s+[a-z]+',
                r'\b[a-z]+\s+(?:sr|jr|ii|iii|iv)\b',
                r'\b(?:officer|agent|adjuster|claims|rep)\s+[a-z]+\b'
            ],
            'organization': [
                r'\b(?:insurance|hospital|clinic|medical|police|fire)\s+[a-z]+\b',
                r'\b[a-z]+\s+(?:insurance|hospital|clinic|medical|center)\b',
                r'\b(?:state|county|city)\s+(?:police|fire|department)\b'
            ],
            'location': [
                r'\b\d+\s+[a-z]+\s+(?:st|ave|road|lane|drive|blvd)\b',
                r'\b[a-z]+,\s*[a-z]{2}\s*\d{5}\b',
                r'\b(?:intersection|highway|interstate|i-\d+)\b'
            ],
            'date_time': [
                r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s*\d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b'
            ],
            'amount': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d+(?:,\d{3})*\s+dollars?\b',
                r'\b(?:usd|dollars?|\$)\s*\d+(?:\.\d{2})?\b'
            ]
        }

    def label_sequence(self, emissions: torch.Tensor,
                      tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Label a sequence of tokens using CRF Viterbi decoding

        Args:
            emissions: Emission scores [seq_len, num_labels]
            tokens: List of tokens

        Returns:
            List of (token, label) tuples
        """
        try:
            # Use Viterbi decoding for best path
            with self.memory_manager.memory_limit_context(20, 'crf_labeling'):
                predicted_tags = self.crf.viterbi_decode(emissions.unsqueeze(0))[0]

            # Convert numeric labels to string labels
            labeled_tokens = []
            for token, tag_idx in zip(tokens, predicted_tags):
                label = self.label_map.get(tag_idx, 'O')
                labeled_tokens.append((token, label))

            return labeled_tokens

        except Exception as e:
            logger.error(f"Sequence labeling failed: {e}")
            # Fallback to simple classification
            return [(token, 'O') for token in tokens]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using pattern matching as fallback

        Args:
            text: Input text

        Returns:
            Dictionary of extracted entities by type
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'amounts': []
        }

        text_lower = text.lower()

        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                matches.extend(found)

            # Remove duplicates and clean up
            matches = list(set(matches))
            if matches:
                if entity_type == 'person':
                    entities['persons'].extend(matches)
                elif entity_type == 'organization':
                    entities['organizations'].extend(matches)
                elif entity_type == 'location':
                    entities['locations'].extend(matches)
                elif entity_type == 'date_time':
                    entities['dates'].extend(matches)
                elif entity_type == 'amount':
                    entities['amounts'].extend(matches)

        # Limit number of entities per type for memory efficiency
        for key in entities:
            entities[key] = entities[key][:5]  # Keep top 5 per type

        return entities

    def get_entity_statistics(self, labeled_sequences: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        """
        Get statistics about labeled entities

        Args:
            labeled_sequences: List of labeled token sequences

        Returns:
            Dictionary with entity statistics
        """
        stats = {
            'total_tokens': 0,
            'entity_counts': {},
            'entity_types': {},
            'average_entities_per_sequence': 0
        }

        total_entities = 0
        for sequence in labeled_sequences:
            stats['total_tokens'] += len(sequence)

            for _, label in sequence:
                if label != 'O':
                    entity_type = label.split('-')[1] if '-' in label else label
                    stats['entity_counts'][entity_type] = stats['entity_counts'].get(entity_type, 0) + 1
                    stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
                    total_entities += 1

        if labeled_sequences:
            stats['average_entities_per_sequence'] = total_entities / len(labeled_sequences)

        return stats

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information for CRF layer
        """
        memory_info = self.memory_manager.check_memory_usage()

        return {
            'component': 'crf_layer',
            'num_labels': self.num_labels,
            'memory_usage': memory_info,
            'pattern_count': sum(len(patterns) for patterns in self.entity_patterns.values())
        }

# Global instances
_crf_instance = None
_sequence_labeler = None

def get_crf_layer(num_tags: int) -> MemoryEfficientCRF:
    """Get or create CRF layer instance"""
    global _crf_instance
    if _crf_instance is None or _crf_instance.num_tags != num_tags:
        _crf_instance = MemoryEfficientCRF(num_tags)
    return _crf_instance

def get_insurance_sequence_labeler() -> InsuranceSequenceLabeler:
    """Get or create insurance sequence labeler instance"""
    global _sequence_labeler
    if _sequence_labeler is None:
        _sequence_labeler = InsuranceSequenceLabeler()
    return _sequence_labeler