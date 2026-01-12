"""Progress bar manager for telemetry events using tqdm."""

import sys
import threading
from typing import Dict, Optional, Any

# tqdm is optional, like websockets
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover – runtime fallback when tqdm not installed
    tqdm = None  # type: ignore


class TelemetryProgressManager:
    """Manages progress bars aggregated by algorithm (model_slug).
    
    Tracks multiple concurrent requests per algorithm and updates progress bars
    dynamically. Also displays resource status from Activity WebSocket.
    """

    def __init__(self, enable: bool = True):
        """Initialize progress manager.
        
        Args:
            enable: Whether to show progress bars. If False, all methods are no-ops.
        """
        self.enable = enable and (tqdm is not None)
        self._algorithm_bars: Dict[str, Any] = {}  # model_slug -> tqdm progress bar
        self._request_tracking: Dict[str, dict] = {}  # request_id -> request state
        self._resource_status: Optional[dict] = None
        self._cache_hits: Dict[str, int] = {}  # per algorithm
        self._lock = threading.Lock()
        self._resource_line: Optional[Any] = None  # tqdm bar for resource status
        # Force tqdm to display even if not a TTY (for scripts)
        if self.enable and tqdm is not None:
            import os
            # If stderr is not a TTY, force tqdm to display anyway
            if not os.isatty(sys.stderr.fileno()):
                # Set environment variable to force display
                os.environ['TQDM_DISABLE'] = '0'

    def start_request(self, request_id: str, model: str, action: str, n_items: int):
        """Create/update algorithm bar, track request.
        
        Args:
            request_id: Unique request identifier
            model: Model slug (e.g., "protein-mpnn")
            action: Action name (e.g., "predict")
            n_items: Number of items in the request
        """
        if not self.enable:
            return
        
        # Debug: log that we're starting a request
        sys.stderr.write(f"[Progress] start_request: {model}.{action} ({n_items} items, req={request_id[:8]})\n")
        sys.stderr.flush()

        with self._lock:
            # Track request state
            self._request_tracking[request_id] = {
                "model": model,
                "action": action,
                "n_items": n_items,
                "backend_items": 0,
                "status": "pending",
                "elapsed": 0.0,
            }

            # Get or create algorithm bar
            algorithm_key = model
            if algorithm_key not in self._algorithm_bars:
                desc = f"{model}.{action}"
                bar = tqdm(
                    total=0,  # Will be updated when we know backend_items
                    desc=desc,
                    unit="item",
                    leave=True,  # Keep bars visible after completion
                    dynamic_ncols=True,
                    mininterval=0.1,  # Update more frequently
                    file=sys.stderr,  # Ensure visibility
                    disable=False,  # Explicitly enable
                )
                self._algorithm_bars[algorithm_key] = {
                    "bar": bar,
                    "total_items": 0,
                    "completed_items": 0,
                    "active_requests": set(),
                    "actions": set(),  # Track which actions are active
                }

            # Add request to active set
            algo_state = self._algorithm_bars[algorithm_key]
            algo_state["active_requests"].add(request_id)
            algo_state["actions"].add(action)
            algo_state["total_items"] += n_items

            # Update bar description to include all active actions
            actions_str = ",".join(sorted(algo_state["actions"]))
            algo_state["bar"].set_description(f"{model}.{actions_str}")

            # Update bar total (will be adjusted when call_submitted is received)
            # Set a minimum total to ensure bar is visible
            algo_state["bar"].total = max(algo_state["total_items"], 1)
            algo_state["bar"].n = 0
            algo_state["bar"].refresh()
            # Force flush to ensure visibility
            sys.stderr.flush()

    def submit_request(self, request_id: str, backend_items: int):
        """Update bar total, mark as submitted.
        
        Args:
            request_id: Unique request identifier
            backend_items: Number of items sent to backend (0 if cached)
        """
        if not self.enable:
            return

        with self._lock:
            if request_id not in self._request_tracking:
                return

            req_state = self._request_tracking[request_id]
            req_state["backend_items"] = backend_items
            req_state["status"] = "submitted"

            model = req_state["model"]
            algorithm_key = model

            if algorithm_key in self._algorithm_bars:
                algo_state = self._algorithm_bars[algorithm_key]

                # Adjust total: subtract original n_items, add backend_items
                # This handles the case where some items were cached
                original_n = req_state["n_items"]
                algo_state["total_items"] = (
                    algo_state["total_items"] - original_n + backend_items
                )

                # Update bar total
                algo_state["bar"].total = max(algo_state["total_items"], 0)
                algo_state["bar"].refresh()
                sys.stderr.flush()

    def cache_hit(self, request_id: str):
        """Complete immediately, increment cache hit count.
        
        Args:
            request_id: Unique request identifier
        """
        if not self.enable:
            return

        with self._lock:
            if request_id not in self._request_tracking:
                return

            req_state = self._request_tracking[request_id]
            req_state["status"] = "cached"
            model = req_state["model"]
            algorithm_key = model

            # Increment cache hit count
            self._cache_hits[algorithm_key] = self._cache_hits.get(algorithm_key, 0) + 1

            if algorithm_key in self._algorithm_bars:
                algo_state = self._algorithm_bars[algorithm_key]
                n_items = req_state["n_items"]

                # Mark items as completed immediately
                algo_state["completed_items"] += n_items
                algo_state["bar"].n = algo_state["completed_items"]
                algo_state["bar"].set_postfix({"status": "cache hit"})
                algo_state["bar"].refresh()

                # Remove from active requests
                algo_state["active_requests"].discard(request_id)

                # Update resource display
                self._update_resource_display()

    def finish_request(self, request_id: str, elapsed: float):
        """Mark finished, update elapsed time.
        
        Args:
            request_id: Unique request identifier
            elapsed: Elapsed time in seconds
        """
        if not self.enable:
            return

        with self._lock:
            if request_id not in self._request_tracking:
                return

            req_state = self._request_tracking[request_id]
            req_state["status"] = "finished"
            req_state["elapsed"] = elapsed

            model = req_state["model"]
            algorithm_key = model

            if algorithm_key in self._algorithm_bars:
                algo_state = self._algorithm_bars[algorithm_key]
                backend_items = req_state.get("backend_items", req_state["n_items"])

                # Mark backend items as completed
                algo_state["completed_items"] += backend_items
                algo_state["bar"].n = min(
                    algo_state["completed_items"], algo_state["bar"].total
                )

                # Update postfix with elapsed time
                num_requests = len(algo_state["active_requests"])
                postfix = {
                    "status": "finished",
                    "elapsed": f"{elapsed:.2f}s",
                    "requests": num_requests,
                }
                algo_state["bar"].set_postfix(postfix)
                algo_state["bar"].refresh()

    def error_request(self, request_id: str, status_code: int):
        """Mark error, update bar status.
        
        Args:
            request_id: Unique request identifier
            status_code: HTTP status code
        """
        if not self.enable:
            return

        with self._lock:
            if request_id not in self._request_tracking:
                return

            req_state = self._request_tracking[request_id]
            req_state["status"] = "error"
            model = req_state["model"]
            algorithm_key = model

            if algorithm_key in self._algorithm_bars:
                algo_state = self._algorithm_bars[algorithm_key]
                algo_state["bar"].set_postfix(
                    {"status": "error", "code": status_code}
                )
                algo_state["bar"].refresh()

                # Remove from active requests
                algo_state["active_requests"].discard(request_id)

    def complete_request(self, request_id: str):
        """Remove request, update bar.
        
        Args:
            request_id: Unique request identifier
        """
        if not self.enable:
            return

        with self._lock:
            if request_id not in self._request_tracking:
                return

            req_state = self._request_tracking[request_id]
            model = req_state["model"]
            algorithm_key = model

            # Remove request tracking
            del self._request_tracking[request_id]

            if algorithm_key in self._algorithm_bars:
                algo_state = self._algorithm_bars[algorithm_key]

                # Remove from active requests
                algo_state["active_requests"].discard(request_id)

                # Update actions set
                remaining_actions = set()
                for rid in algo_state["active_requests"]:
                    if rid in self._request_tracking:
                        remaining_actions.add(
                            self._request_tracking[rid].get("action", "")
                        )
                algo_state["actions"] = remaining_actions

                # Update description
                if remaining_actions:
                    actions_str = ",".join(sorted(remaining_actions))
                    algo_state["bar"].set_description(f"{model}.{actions_str}")
                else:
                    # No active requests, close bar
                    algo_state["bar"].close()
                    del self._algorithm_bars[algorithm_key]
                    return

                # Update postfix with active request count
                num_requests = len(algo_state["active_requests"])
                if num_requests > 0:
                    current_postfix = algo_state["bar"].postfix or {}
                    if isinstance(current_postfix, dict):
                        current_postfix["requests"] = num_requests
                        algo_state["bar"].set_postfix(current_postfix)
                    else:
                        algo_state["bar"].set_postfix({"requests": num_requests})
                algo_state["bar"].refresh()

    def update_resources(self, activity_data: dict):
        """Update resource status display from Activity WebSocket.
        
        Args:
            activity_data: Activity update data from WebSocket
        """
        if not self.enable:
            return

        with self._lock:
            self._resource_status = activity_data
            self._update_resource_display()

    def update_billing(self, billing_data: dict):
        """Update billing information (currently not displayed).
        
        Args:
            billing_data: Billing update data from WebSocket
        """
        # Future: could display budget information
        pass

    def update_hint(self, hint_data: dict):
        """Update activity hint (currently not displayed).
        
        Args:
            hint_data: Activity hint data from WebSocket
        """
        # Future: could show hints like "starting", "cache", "error"
        pass

    def _update_resource_display(self):
        """Update the resource status line above progress bars."""
        if not self.enable:
            return

        # Build resource status string
        parts = []

        if self._resource_status:
            algorithms = self._resource_status.get("algorithms", {})
            totals = self._resource_status.get("totals", {})

            num_algorithms = len(algorithms)
            if num_algorithms > 0:
                parts.append(f"{num_algorithms} algorithms")

            gpus = totals.get("gpus", 0)
            if gpus > 0:
                parts.append(f"{gpus} GPUs")

            cpu_cores = totals.get("cpu_cores", 0)
            if cpu_cores > 0:
                parts.append(f"{cpu_cores} CPUs")

            memory_gb = totals.get("memory_gb", 0)
            if memory_gb > 0:
                parts.append(f"{memory_gb:.1f}GB memory")

        # Add cache hits
        total_cache_hits = sum(self._cache_hits.values())
        if total_cache_hits > 0:
            parts.append(f"{total_cache_hits} cache hits")

        if parts:
            status_text = "Active: " + " | ".join(parts)

            # Create or update resource status line
            if self._resource_line is None:
                self._resource_line = tqdm(
                    total=0,
                    desc=status_text,
                    bar_format="{desc}",
                    leave=True,
                    position=0,
                    file=sys.stderr,  # Ensure visibility
                )
            else:
                self._resource_line.set_description(status_text)
                self._resource_line.refresh()

    def close_all(self):
        """Clean up all progress bars."""
        if not self.enable:
            return

        with self._lock:
            for algo_state in self._algorithm_bars.values():
                algo_state["bar"].close()

            if self._resource_line is not None:
                self._resource_line.close()

            self._algorithm_bars.clear()
            self._request_tracking.clear()
            self._resource_line = None

