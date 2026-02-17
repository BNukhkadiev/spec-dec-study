"""Configuration management for vLLM experiments."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class VLLMConfig:
    """Configuration for vLLM engine and generation."""
    
    # Model configuration
    model: str
    
    # Speculative decoding settings
    speculative_decoding: Optional[Dict[str, Any]] = None
    
    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    
    # Engine parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    trust_remote_code: bool = False
    
    # Additional vLLM parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_vllm_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs for vLLM LLMEngine initialization.
        
        Note: The exact vLLM API for speculative decoding may vary by version.
        Check vLLM documentation for the latest parameter names.
        Common parameters include:
        - speculative_model: Draft model path
        - num_speculative_tokens: Number of tokens to speculate
        - enable_speculative_decoding: Boolean flag (some versions)
        """
        kwargs = {
            'model': self.model,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'trust_remote_code': self.trust_remote_code,
        }
        
        if self.max_model_len is not None:
            kwargs['max_model_len'] = self.max_model_len
        
        if self.speculative_decoding:
            # vLLM uses speculative_config as a dictionary
            spec_config = {}
            
            # Model (draft model path or "[ngram]" for n-gram matching)
            spec_model = self.speculative_decoding.get('model')
            if spec_model:
                spec_config['model'] = spec_model
            
            # Number of speculative tokens
            num_tokens = self.speculative_decoding.get('num_tokens', 5)
            spec_config['num_speculative_tokens'] = num_tokens
            
            # N-gram matching parameters (for self-drafting)
            if 'ngram_prompt_lookup_max' in self.speculative_decoding:
                spec_config['prompt_lookup_max'] = self.speculative_decoding['ngram_prompt_lookup_max']
            
            # Draft tensor parallel size (for separate draft models)
            if 'draft_tensor_parallel_size' in self.speculative_decoding:
                spec_config['draft_tensor_parallel_size'] = self.speculative_decoding['draft_tensor_parallel_size']
            
            # Method (e.g., "eagle", "eagle3", "ngram")
            if 'method' in self.speculative_decoding:
                spec_config['method'] = self.speculative_decoding['method']
            
            # Add other speculative decoding params
            for key, value in self.speculative_decoding.items():
                if key not in ['model', 'num_tokens', 'enabled', 'ngram_prompt_lookup_max', 
                              'draft_tensor_parallel_size', 'method']:
                    spec_config[key] = value
            
            kwargs['speculative_config'] = spec_config
        
        # Add extra params (useful for version-specific parameters)
        kwargs.update(self.extra_params)
        
        return kwargs
    
    def to_generation_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for generation."""
        kwargs = {
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
        }
        # top_k: -1 means disabled, None is not accepted by vLLM
        if self.top_k > 0:
            kwargs['top_k'] = self.top_k
        # If top_k is -1 or 0, don't include it (vLLM will use default)
        return kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VLLMConfig':
        """Create from dictionary, handling nested structures."""
        # Flatten nested structures if present
        config_data = {}
        
        # Model (required)
        config_data['model'] = data.get('model')
        if config_data['model'] is None:
            raise ValueError("'model' field is required in config")
        
        # Speculative decoding
        if 'speculative_decoding' in data:
            spec_config = data['speculative_decoding']
            if isinstance(spec_config, dict) and spec_config.get('enabled', True):
                # Preserve all fields from speculative_decoding config
                config_data['speculative_decoding'] = {k: v for k, v in spec_config.items() if k != 'enabled'}
            else:
                config_data['speculative_decoding'] = spec_config
        
        # Generation parameters (can be nested or flat)
        if 'generation' in data:
            gen_config = data['generation']
            config_data['max_tokens'] = gen_config.get('max_tokens', 512)
            config_data['temperature'] = gen_config.get('temperature', 0.7)
            config_data['top_p'] = gen_config.get('top_p', 1.0)
            config_data['top_k'] = gen_config.get('top_k', -1)
        else:
            config_data['max_tokens'] = data.get('max_tokens', 512)
            config_data['temperature'] = data.get('temperature', 0.7)
            config_data['top_p'] = data.get('top_p', 1.0)
            config_data['top_k'] = data.get('top_k', -1)
        
        # Engine parameters (can be nested or flat)
        if 'engine' in data:
            engine_config = data['engine']
            config_data['tensor_parallel_size'] = engine_config.get('tensor_parallel_size', 1)
            config_data['gpu_memory_utilization'] = engine_config.get('gpu_memory_utilization', 0.9)
            config_data['max_model_len'] = engine_config.get('max_model_len')
            config_data['trust_remote_code'] = engine_config.get('trust_remote_code', False)
            # Extract extra_params from engine section if present
            if 'extra_params' in engine_config:
                config_data['extra_params'] = engine_config.get('extra_params', {})
            else:
                config_data['extra_params'] = data.get('extra_params', {})
        else:
            config_data['tensor_parallel_size'] = data.get('tensor_parallel_size', 1)
            config_data['gpu_memory_utilization'] = data.get('gpu_memory_utilization', 0.9)
            config_data['max_model_len'] = data.get('max_model_len')
            config_data['trust_remote_code'] = data.get('trust_remote_code', False)
            config_data['extra_params'] = data.get('extra_params', {})
        
        return cls(**config_data)
    
    @classmethod
    def from_file(cls, file_path: str | Path) -> 'VLLMConfig':
        """
        Load configuration from YAML or JSON file.
        
        Args:
            file_path: Path to config file
            
        Returns:
            VLLMConfig instance
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
        
        return cls.from_dict(data)
    
    def save(self, file_path: str | Path):
        """Save configuration to file."""
        file_path = Path(file_path)
        data = self.to_dict()
        
        with open(file_path, 'w') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            elif file_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
