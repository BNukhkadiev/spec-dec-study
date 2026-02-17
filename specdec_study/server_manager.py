"""vLLM server manager for starting/stopping servers with configurations."""

import subprocess
import time
import signal
import os
from pathlib import Path
from typing import Optional
import requests

from .config import VLLMConfig


class VLLMServerManager:
    """Manage vLLM OpenAI API server lifecycle."""
    
    def __init__(
        self,
        config: VLLMConfig,
        host: str = "127.0.0.1",
        port: int = 8000,
        wait_timeout: int = 120,
    ):
        """
        Initialize server manager.
        
        Args:
            config: vLLM configuration
            host: Server host address
            port: Server port
            wait_timeout: Maximum seconds to wait for server to start
        """
        self.config = config
        self.host = host
        self.port = port
        self.wait_timeout = wait_timeout
        self.server_url = f"http://{host}:{port}"
        self.health_url = f"{self.server_url}/health"
        self.process: Optional[subprocess.Popen] = None
    
    def _build_server_command(self) -> list[str]:
        """Build vLLM server command from config."""
        # Use vllm serve command (simpler)
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"]
        
        # Model
        cmd.extend(["--model", self.config.model])
        
        # Engine parameters
        vllm_kwargs = self.config.to_vllm_kwargs()
        
        if "tensor_parallel_size" in vllm_kwargs:
            cmd.extend(["--tensor-parallel-size", str(vllm_kwargs["tensor_parallel_size"])])
        
        if "gpu_memory_utilization" in vllm_kwargs:
            cmd.extend(["--gpu-memory-utilization", str(vllm_kwargs["gpu_memory_utilization"])])
        
        if "max_model_len" in vllm_kwargs:
            cmd.extend(["--max-model-len", str(vllm_kwargs["max_model_len"])])
        
        if "trust_remote_code" in vllm_kwargs and vllm_kwargs["trust_remote_code"]:
            cmd.append("--trust-remote-code")
        
        # Speculative decoding - vLLM uses --speculative-config with JSON string
        if "speculative_config" in vllm_kwargs:
            spec_config = vllm_kwargs["speculative_config"]
            import json
            spec_json = json.dumps(spec_config)
            cmd.extend(["--speculative-config", spec_json])
        
        # Server address
        cmd.extend(["--host", self.host])
        cmd.extend(["--port", str(self.port)])
        
        return cmd
    
    def start(self):
        """Start vLLM server and wait for it to be ready."""
        if self.process is not None:
            raise RuntimeError("Server is already running")
        
        cmd = self._build_server_command()
        print(f"Starting vLLM server: {' '.join(cmd)}")
        
        # Start server process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,  # Create new process group
        )
        
        # Wait for server to be ready
        print(f"Waiting for server at {self.server_url}...")
        start_time = time.time()
        health_ready = False
        models_ready = False
        
        while time.time() - start_time < self.wait_timeout:
            # Check if process died
            if self.process.poll() is not None:
                stdout, _ = self.process.communicate()
                raise RuntimeError(
                    f"Server process exited with code {self.process.returncode}\n"
                    f"Output: {stdout}"
                )
            
            # Check health endpoint
            if not health_ready:
                try:
                    response = requests.get(self.health_url, timeout=2)
                    if response.status_code == 200:
                        health_ready = True
                        print(f"✓ Health endpoint ready")
                except (requests.ConnectionError, requests.Timeout):
                    pass
            
            # Check /v1/models endpoint (OpenAI-compatible API validation)
            if health_ready and not models_ready:
                try:
                    models_response = requests.get(
                        f"{self.server_url}/v1/models",
                        timeout=10
                    )
                    if models_response.status_code == 200:
                        models_ready = True
                        print(f"✓ Models endpoint ready")
                except (requests.ConnectionError, requests.Timeout) as e:
                    pass  # Not ready yet, continue waiting
            
            # Both endpoints ready - give server a moment to fully initialize
            if health_ready and models_ready:
                print(f"✓ Server fully ready at {self.server_url}")
                time.sleep(3)  # Extra delay to ensure server is stable
                return
            
            time.sleep(1)
        
        # Timeout
        self.stop()
        raise RuntimeError(f"Server failed to start within {self.wait_timeout} seconds")
    
    def stop(self):
        """Stop vLLM server."""
        if self.process is None:
            return
        
        print(f"Stopping server at {self.server_url}...")
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
                print("✓ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
                print("✓ Server force stopped")
        except ProcessLookupError:
            # Process already dead
            pass
        finally:
            self.process = None
    
    def is_running(self) -> bool:
        """Check if server is running."""
        if self.process is None:
            return False
        
        if self.process.poll() is not None:
            return False
        
        try:
            response = requests.get(self.health_url, timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
