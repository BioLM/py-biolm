"""Watch live activity updates from Activity WebSocket."""

import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import websockets  # type: ignore
except ImportError:
    websockets = None  # type: ignore

from biolmai.client import ActivityListener, CredentialsProvider, DEFAULT_BASE_URL


class ActivityWatchDisplay:
    """Manages tqdm display for activity updates."""

    def __init__(self):
        self._resource_bar: Optional[Any] = None
        self._billing_bar: Optional[Any] = None
        self._cache_hits: Dict[str, int] = {}  # algorithm -> count
        self._resource_status: Optional[dict] = None
        self._billing_status: Optional[dict] = None

    def update_resources(self, data: dict):
        """Update resource display from activity_update event.

        Args:
            data: Activity update data from WebSocket
        """
        self._resource_status = data
        self._update_resource_display()

    def update_billing(self, data: dict):
        """Update billing display from billing_update event.

        Args:
            data: Billing update data from WebSocket
        """
        self._billing_status = data
        self._update_billing_display()

    def update_hint(self, data: dict):
        """Update cache hit counts from activity_hint event.

        Args:
            data: Activity hint data from WebSocket
        """
        algorithm = data.get("algorithm")
        phase = data.get("phase")
        if algorithm and phase == "cache":
            self._cache_hits[algorithm] = self._cache_hits.get(algorithm, 0) + 1
            self._update_resource_display()

    def _update_resource_display(self):
        """Update the resource status line."""
        if tqdm is None:
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
            if self._resource_bar is None:
                self._resource_bar = tqdm(
                    total=0,
                    desc=status_text,
                    bar_format="{desc}",
                    leave=True,
                    position=0,
                    file=sys.stderr,
                )
            else:
                self._resource_bar.set_description(status_text)
                self._resource_bar.refresh()
        elif self._resource_bar is not None:
            # No active resources, but bar exists - update to show empty state
            self._resource_bar.set_description("Active: (no active resources)")
            self._resource_bar.refresh()

    def _update_billing_display(self):
        """Update the billing status line."""
        if tqdm is None:
            return

        if not self._billing_status:
            return

        remaining = self._billing_status.get("remaining_budget", 0)
        total = self._billing_status.get("total_budget", 0)
        usage = self._billing_status.get("current_usage", 0)

        if total > 0:
            budget_text = f"Budget: ${remaining:.2f} / ${total:.2f} (${usage:.2f} used)"

            # Create or update billing status line
            if self._billing_bar is None:
                self._billing_bar = tqdm(
                    total=0,
                    desc=budget_text,
                    bar_format="{desc}",
                    leave=True,
                    position=1,
                    file=sys.stderr,
                )
            else:
                self._billing_bar.set_description(budget_text)
                self._billing_bar.refresh()

    def close(self):
        """Close all tqdm bars."""
        if self._resource_bar is not None:
            self._resource_bar.close()
            self._resource_bar = None
        if self._billing_bar is not None:
            self._billing_bar.close()
            self._billing_bar = None


async def watch_activity_async(json_output: bool = False):
    """Async function to watch activity updates.

    Args:
        json_output: If True, output raw JSON events. If False, use formatted display.
    """
    if websockets is None:
        print("Error: websockets library not installed. Install with: pip install websockets", file=sys.stderr)
        return

    if tqdm is None and not json_output:
        print("Error: tqdm library not installed. Install with: pip install tqdm", file=sys.stderr)
        return

    # Get auth headers
    try:
        headers = CredentialsProvider.get_auth_headers()
    except AssertionError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please set BIOLMAI_TOKEN or run 'biolmai login'", file=sys.stderr)
        return

    # Get base URL from environment or use default
    base_url = os.getenv("BIOLM_BASE_URL") or os.getenv("BIOLMAI_BASE_URL") or DEFAULT_BASE_URL

    # Construct WebSocket URL
    parsed = urlparse(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.netloc
    activity_ws_url = f"{ws_scheme}://{host}/ws/activity/"

    # Create display manager (only if not JSON output)
    display = ActivityWatchDisplay() if not json_output else None

    # Event handler
    async def activity_handler(data):
        """Handle activity WebSocket events."""
        if json_output:
            # Output raw JSON
            print(json.dumps(data))
            sys.stdout.flush()
        else:
            # Update display
            if display is None:
                return

            event_type = data.get("type")
            event_data = data.get("data", {})

            if event_type == "activity_update":
                display.update_resources(event_data)
            elif event_type == "billing_update":
                display.update_billing(event_data)
            elif event_type == "activity_hint":
                display.update_hint(event_data)

    # Show initial message
    if not json_output:
        print("Connecting to Activity WebSocket...", file=sys.stderr)
        if display is not None:
            # Initialize display with empty state
            display._update_resource_display()
            display._update_billing_display()

    # Keep listening until interrupted
    # If connection closes, reconnect automatically
    while True:
        try:
            # Create ActivityListener
            listener = ActivityListener(
                activity_ws_url,
                headers=headers,
                handler=activity_handler,
            )

            # Start listening (this will block until connection closes or error)
            # The listen() method has its own retry logic, but we wrap it
            # in a loop to reconnect if it exits
            await listener.listen()
            
            # If we get here, connection closed - wait a bit and reconnect
            if not json_output:
                print("Connection closed, reconnecting...", file=sys.stderr)
            await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            if display is not None:
                display.close()
            print("\nStopped watching activity updates.", file=sys.stderr)
            break
        except Exception as e:
            # For connection errors, retry after a delay
            error_str = str(e).lower()
            if "connection" in error_str or "timeout" in error_str or "closed" in error_str:
                if not json_output:
                    print(f"Connection error: {e}. Retrying in 2 seconds...", file=sys.stderr)
                await asyncio.sleep(2)
                continue
            else:
                # Fatal error, clean up and exit
                if display is not None:
                    display.close()
                print(f"Error: {e}", file=sys.stderr)
                raise


def watch_activity(json_output: bool = False):
    """Main entry point for watch command.

    Args:
        json_output: If True, output raw JSON events. If False, use formatted display.
    """
    try:
        asyncio.run(watch_activity_async(json_output=json_output))
    except KeyboardInterrupt:
        # Already handled in async function, but catch here too just in case
        pass

