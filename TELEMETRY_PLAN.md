# Telemetry & Live Progress Display - Implementation Plan

## Executive Summary

This document outlines the plan for enhancing the BioLM Python SDK with real-time telemetry visualization using WebSocket connections and dynamic progress displays. The goal is to provide users with live feedback about API request status, algorithm resource acquisition, and account-level activity updates.

---

## 1. Current State Audit

### 1.1 Existing Telemetry Infrastructure

**Location**: `biolmai/client.py`

**Current Implementation**:
- ✅ **TelemetryListener class** (lines 263-337): Basic WebSocket listener that:
  - Connects to `/ws/telemetry/<channel>/` endpoint
  - Captures events in `self.events` list
  - Supports custom handler callbacks
  - Currently prints to stderr via `debug()` function
  - Handles events: `request_start`, `cache_hit`, `call_submitted`, `call_finished`, `error`, `response_sent`

- ✅ **BioLMApiClient telemetry support** (lines 339-576):
  - `telemetry: bool` parameter to enable/disable
  - `telemetry_handler: Optional[Callable]` for custom callbacks
  - Single persistent WebSocket channel per client instance (`telemetry_{uuid}`)
  - Per-request `X-Request-Id` header for event correlation
  - Event capture in `_last_telemetry_events` for testing/debugging

**Current Output Method**:
- Uses `debug()` function (line 55-57) which writes to `sys.stderr`
- Format: `[Telemetry-{tag_prefix}] {evt}{suffix}`
- Example: `[Telemetry-mpnn_protein.predict-abc123] call_submitted 3 sent`

**Limitations**:
1. ❌ No Activity WebSocket connection (`/ws/activity/`) for account-level updates
2. ❌ No progress bars or dynamic visual updates
3. ❌ Basic text output only (no rich formatting)
4. ❌ No Jupyter notebook-specific handling
5. ❌ No aggregation of multiple requests
6. ❌ No display of algorithm/resource status

### 1.2 WebSocket API Documentation Summary

**Two WebSocket Endpoints Available**:

1. **Telemetry WebSocket** (`/ws/telemetry/<channel>/`)
   - Per-request updates for individual API calls
   - Channel format: `telemetry_<request_id>`
   - Events: `request_start`, `cache_hit`, `call_submitted`, `call_finished`, `error`, `response_sent`
   - Currently implemented ✅

2. **Activity WebSocket** (`/ws/activity/`)
   - Account-level activity updates
   - Auto-subscribes to: `user_activity_<user_id>`, `activity.inst.<institute_id>`, `activity.inst.<institute_id>.env.<environment_id>`
   - Events: `activity_update`, `billing_update`, `activity_hint`, `ctx_mismatch`
   - **NOT currently implemented** ❌

**Key Data from Activity WebSocket**:
- Active algorithms with resource counts (containers, CPU cores, memory, GPUs)
- Total resource utilization
- Budget information (remaining, total, current usage)
- Activity hints (throttled updates for UI feedback)

---

## 2. Progress Bar Library Selection

### 2.1 Options Evaluated

#### Option A: `tqdm` (Recommended)
**Pros**:
- ✅ Most popular Python progress bar library (3.8k+ stars on pandarallel's usage)
- ✅ Excellent Jupyter notebook support (`tqdm.notebook` or `tqdm.auto`)
- ✅ Thread-safe and async-compatible
- ✅ Lightweight dependency
- ✅ Supports nested progress bars
- ✅ Can be updated manually (not just for iterables)
- ✅ Rich customization options

**Cons**:
- ⚠️ Basic styling (though functional)
- ⚠️ Limited multi-line display capabilities

**Usage Pattern**:
```python
from tqdm.auto import tqdm  # Auto-detects terminal vs notebook

# Manual updates
pbar = tqdm(total=100, desc="Processing")
pbar.update(50)
pbar.set_description("Almost done")
pbar.close()
```

#### Option B: `rich` (Alternative)
**Pros**:
- ✅ Beautiful terminal output with colors, tables, panels
- ✅ Excellent for complex multi-line displays
- ✅ `rich.progress` module for progress bars
- ✅ Great for showing algorithm/resource status tables

**Cons**:
- ⚠️ Larger dependency
- ⚠️ More complex API
- ⚠️ Jupyter support is newer/less mature than tqdm

**Usage Pattern**:
```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

with Progress() as progress:
    task = progress.add_task("[green]Processing...", total=100)
    progress.update(task, advance=50)
```

### 2.2 Recommendation: **`tqdm` with `tqdm.auto`**

**Rationale**:
1. **Proven track record**: Used by pandarallel and many other Python libraries
2. **Jupyter compatibility**: `tqdm.auto` automatically detects environment
3. **Simplicity**: Easier to integrate with existing async code
4. **Lightweight**: Minimal dependency overhead
5. **Flexibility**: Can create multiple progress bars for different algorithms/resources

**For future enhancement**: Consider `rich` for more sophisticated displays (e.g., resource status tables), but start with `tqdm` for core functionality.

---

## 3. Architecture Design

### 3.1 WebSocket Connection Strategy

**Two-Connection Approach** (Recommended):

1. **Telemetry WebSocket** (per-client, persistent)
   - Already implemented
   - One connection per `BioLMApiClient` instance
   - Reused across multiple API calls
   - Channel: `telemetry_{uuid}` (generated once per client)

2. **Activity WebSocket** (per-client, persistent)
   - **NEW**: Connect to `/ws/activity/` endpoint
   - One connection per `BioLMApiClient` instance (or optionally shared)
   - Receives account-level updates
   - Auto-subscribes to user/institute/environment groups

**Why Two Connections?**
- Telemetry WebSocket: Request-specific, high-frequency events
- Activity WebSocket: Account-level, lower-frequency but important for resource status
- Separation of concerns: per-request vs account-level

**Alternative: Single Activity Connection**
- Could use only Activity WebSocket and rely on `activity_hint` events
- **Not recommended**: Less granular per-request feedback

### 3.2 Progress Display Architecture

#### 3.2.1 Display Modes

**Mode 1: Per-Request Progress** (from Telemetry WebSocket)
- Single progress bar per API call
- Shows: request status, items processed, elapsed time
- Updates: `request_start` → `call_submitted` → `call_finished` → `response_sent`
- **Use case**: Individual API calls

**Mode 2: Batch Progress** (aggregated from Telemetry WebSocket)
- Single progress bar for multiple requests
- Shows: total items, completed items, active requests
- Updates: Aggregate across multiple `request_id`s
- **Use case**: Batch processing with `_batch_call_autoschema_or_manual()`

**Mode 3: Resource Status Display** (from Activity WebSocket)
- Multi-line display showing:
  - Active algorithms (name, containers, CPU, memory, GPUs)
  - Total resource utilization
  - Budget information
- Updates: On `activity_update` events
- **Use case**: Long-running sessions, monitoring resource acquisition

#### 3.2.2 Progress Bar Manager

**New Class: `TelemetryProgressManager`**

**Responsibilities**:
1. Create and manage progress bars
2. Update progress bars based on WebSocket events
3. Handle Jupyter vs terminal detection
4. Aggregate multiple requests for batch operations
5. Display resource status from Activity WebSocket

**Interface**:
```python
class TelemetryProgressManager:
    def __init__(self, enable: bool = True, style: str = "auto"):
        # style: "auto", "tqdm", "rich", "none"
        pass
    
    def start_request(self, request_id: str, model: str, action: str, n_items: int):
        """Create progress bar for new request"""
        pass
    
    def update_request(self, request_id: str, event: dict):
        """Update progress bar based on telemetry event"""
        pass
    
    def finish_request(self, request_id: str):
        """Close progress bar for completed request"""
        pass
    
    def update_resources(self, activity_data: dict):
        """Update resource status display from Activity WebSocket"""
        pass
    
    def close_all(self):
        """Clean up all progress bars"""
        pass
```

### 3.3 Integration Points

#### 3.3.1 BioLMApiClient Modifications

**Changes needed**:

1. **Add Activity WebSocket connection**:
   - New `ActivityListener` class (similar to `TelemetryListener`)
   - Connect in `__init__` or lazily on first API call
   - Store in `self._activity_listener` and `self._activity_task`

2. **Add Progress Manager**:
   - New parameter: `progress: bool = True` (default True for better UX)
   - New parameter: `progress_style: str = "auto"` (tqdm/rich/none)
   - Instantiate `TelemetryProgressManager` in `__init__`

3. **Wire TelemetryListener to Progress Manager**:
   - Modify `TelemetryListener._notify()` to call progress manager
   - Or: Progress manager subscribes to telemetry events

4. **Wire ActivityListener to Progress Manager**:
   - Activity listener calls `progress_manager.update_resources()`

#### 3.3.2 Event Flow

```
API Request
    ↓
BioLMApiClient._api_call()
    ↓
HTTP POST with X-Telemetry-Channel & X-Request-Id
    ↓
┌─────────────────────────────────────┐
│ Telemetry WebSocket                │
│ - request_start                    │ → ProgressManager.start_request()
│ - call_submitted                   │ → ProgressManager.update_request()
│ - call_finished                    │ → ProgressManager.update_request()
│ - response_sent                    │ → ProgressManager.finish_request()
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Activity WebSocket                  │
│ - activity_update                   │ → ProgressManager.update_resources()
│ - billing_update                    │ → ProgressManager.update_resources()
│ - activity_hint                     │ → ProgressManager.update_hint()
└─────────────────────────────────────┘
```

---

## 4. Implementation Plan

### 4.1 Phase 1: Core Progress Bar Infrastructure

**Tasks**:
1. ✅ Add `tqdm` to dependencies (optional dependency, like `websockets`)
2. ✅ Create `TelemetryProgressManager` class
3. ✅ Implement Jupyter/terminal auto-detection
4. ✅ Basic per-request progress bar (single request)
5. ✅ Wire to existing `TelemetryListener`

**Files to create/modify**:
- `biolmai/telemetry_progress.py` (new file)
- `biolmai/client.py` (modify `TelemetryListener` and `BioLMApiClient`)

**Dependencies**:
- Add `tqdm>=4.65.0` to `pyproject.toml` (optional, like websockets)

### 4.2 Phase 2: Activity WebSocket Integration

**Tasks**:
1. ✅ Create `ActivityListener` class (similar to `TelemetryListener`)
2. ✅ Connect to `/ws/activity/` endpoint
3. ✅ Handle authentication (reuse existing headers)
4. ✅ Parse `activity_update`, `billing_update`, `activity_hint` events
5. ✅ Integrate with `TelemetryProgressManager` for resource display

**Files to create/modify**:
- `biolmai/client.py` (add `ActivityListener` class)
- `biolmai/telemetry_progress.py` (add resource status display)

### 4.3 Phase 3: Enhanced Display Features

**Tasks**:
1. ✅ Batch progress aggregation (multiple requests)
2. ✅ Resource status table/display
3. ✅ Budget information display
4. ✅ Activity hints integration
5. ✅ Error state visualization

**Files to modify**:
- `biolmai/telemetry_progress.py` (enhance display logic)

### 4.4 Phase 4: Polish & Testing

**Tasks**:
1. ✅ Jupyter notebook testing
2. ✅ Terminal testing
3. ✅ Error handling (WebSocket disconnections)
4. ✅ Performance testing (multiple concurrent requests)
5. ✅ Documentation updates

---

## 5. Detailed Component Specifications

### 5.1 TelemetryProgressManager

**Location**: `biolmai/telemetry_progress.py`

**Key Methods**:

```python
class TelemetryProgressManager:
    def __init__(self, enable: bool = True, style: str = "auto"):
        """
        Args:
            enable: Whether to show progress bars
            style: "auto" (detect), "tqdm", "rich", "none"
        """
        self.enable = enable
        self.style = self._detect_style(style)
        self._request_bars: Dict[str, tqdm] = {}  # request_id -> progress bar
        self._resource_display: Optional[ResourceDisplay] = None
        self._lock = threading.Lock()  # For thread safety
    
    def start_request(self, request_id: str, model: str, action: str, n_items: int):
        """Create progress bar for new request"""
        if not self.enable:
            return
        
        desc = f"{model}.{action}"
        bar = tqdm(
            total=n_items,
            desc=desc,
            unit="item",
            leave=False,  # Don't leave bar after completion
        )
        self._request_bars[request_id] = bar
    
    def update_request(self, request_id: str, event: dict):
        """Update progress bar based on telemetry event"""
        if not self.enable:
            return
        
        bar = self._request_bars.get(request_id)
        if not bar:
            return
        
        evt = event.get("event")
        
        if evt == "request_start":
            bar.set_description(f"{event.get('model', '?')}.{event.get('action', '?')}")
        elif evt == "call_submitted":
            backend_items = event.get("backend_items", 0)
            bar.total = backend_items
            bar.set_postfix({"status": "submitted"})
        elif evt == "call_finished":
            elapsed = event.get("elapsed", 0)
            bar.set_postfix({"status": "finished", "elapsed": f"{elapsed:.2f}s"})
            bar.n = bar.total  # Complete the bar
        elif evt == "cache_hit":
            bar.set_postfix({"status": "cache hit"})
            bar.n = bar.total
        elif evt == "error":
            bar.set_postfix({"status": "error", "code": event.get("status_code", "?")})
            bar.close()
            del self._request_bars[request_id]
    
    def finish_request(self, request_id: str):
        """Close progress bar for completed request"""
        bar = self._request_bars.get(request_id)
        if bar:
            bar.close()
            del self._request_bars[request_id]
    
    def update_resources(self, activity_data: dict):
        """Update resource status display from Activity WebSocket"""
        # Implementation for resource display
        pass
    
    def _detect_style(self, style: str) -> str:
        """Detect if running in Jupyter notebook"""
        if style != "auto":
            return style
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return "tqdm"  # Use tqdm.auto which handles notebook
        except ImportError:
            pass
        return "tqdm"
```

### 5.2 ActivityListener

**Location**: `biolmai/client.py` (alongside `TelemetryListener`)

**Key Methods**:

```python
class ActivityListener:
    """Listens to account-level activity updates via Activity WebSocket"""
    
    def __init__(self, ws_url: str, handler: Optional[Callable] = None):
        self.ws_url = ws_url
        self.events: List[Any] = []
        self._connected = asyncio.Event()
        self._handler = handler
    
    async def listen(self):
        """Connect and listen for activity updates"""
        if websockets is None:
            return
        try:
            # Include auth headers
            headers = self._get_auth_headers()
            async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
                self._connected.set()
                async for raw in ws:
                    try:
                        data = _json.loads(raw)
                    except Exception:
                        data = raw
                    self.events.append(data)
                    await self._notify(data)
        except Exception as e:
            debug(f"[Activity] Listener error: {e}")
    
    async def _notify(self, data):
        """Invoke handler callback"""
        if self._handler:
            try:
                res = self._handler(data)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                debug(f"[Activity] handler error: {e}")
```

### 5.3 BioLMApiClient Integration

**Modifications to `BioLMApiClient.__init__`**:

```python
def __init__(
    self,
    model_name: str,
    # ... existing params ...
    *,
    telemetry: bool = False,
    telemetry_handler: Optional[Callable] = None,
    progress: bool = True,  # NEW: Enable progress bars
    progress_style: str = "auto",  # NEW: "auto", "tqdm", "rich", "none"
):
    # ... existing initialization ...
    
    # NEW: Progress manager
    self.progress_enabled = progress and (telemetry or True)  # Enable if telemetry or standalone
    if self.progress_enabled:
        from biolmai.telemetry_progress import TelemetryProgressManager
        self._progress_manager = TelemetryProgressManager(
            enable=True,
            style=progress_style
        )
    else:
        self._progress_manager = None
    
    # NEW: Activity WebSocket (if progress enabled)
    self._activity_listener: Optional[ActivityListener] = None
    self._activity_task: Optional[asyncio.Task] = None
    
    if self.progress_enabled:
        # Set up activity listener
        parsed = urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        host = parsed.netloc
        activity_ws_url = f"{ws_scheme}://{host}/ws/activity/"
        self._activity_listener = ActivityListener(
            activity_ws_url,
            handler=self._handle_activity_event
        )
```

**Modifications to `TelemetryListener._notify`**:

```python
async def _notify(self, data):
    """Invoke handler and progress manager"""
    # Existing handler
    if self._handler:
        # ... existing handler code ...
    
    # NEW: Progress manager update
    if hasattr(self, '_progress_manager') and self._progress_manager:
        request_id = data.get("request_id") or data.get("rid")
        evt = data.get("event")
        
        if evt == "request_start":
            self._progress_manager.start_request(
                request_id,
                data.get("model", "?"),
                data.get("action", "?"),
                data.get("n_items", 0)
            )
        elif evt in ("call_submitted", "call_finished", "cache_hit", "error"):
            self._progress_manager.update_request(request_id, data)
        elif evt == "response_sent":
            self._progress_manager.finish_request(request_id)
```

---

## 6. Dependencies

### 6.1 New Dependencies

**Required**:
- None (websockets already optional)

**Optional** (for progress bars):
- `tqdm>=4.65.0` - Progress bar library

**Future consideration**:
- `rich>=13.0.0` - For enhanced displays (optional)

### 6.2 Dependency Management

Follow existing pattern for `websockets`:
- Make `tqdm` optional at runtime
- Check for availability: `try: import tqdm; except ImportError: tqdm = None`
- Gracefully degrade if not available (show text output instead)

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Location**: `tests/test_telemetry_progress.py` (new file)

**Test Cases**:
1. Progress bar creation and updates
2. Jupyter vs terminal detection
3. Multiple concurrent requests
4. Error handling (missing tqdm)
5. Activity WebSocket event parsing

### 7.2 Integration Tests

**Test Cases**:
1. End-to-end: API call with progress bar
2. Batch processing with aggregated progress
3. Activity WebSocket connection and updates
4. Resource status display updates

### 7.3 Manual Testing

**Scenarios**:
1. Terminal: Single API call
2. Terminal: Batch processing
3. Jupyter: Single API call
4. Jupyter: Batch processing
5. Long-running session with resource monitoring

---

## 8. Backward Compatibility

### 8.1 API Changes

**No breaking changes**:
- All new parameters are optional with sensible defaults
- `progress=True` by default (better UX)
- `telemetry=False` remains default (existing behavior preserved)
- Existing `telemetry_handler` callback still works

**Migration path**:
- Existing code continues to work unchanged
- Users can opt-in to progress bars: `BioLMApiClient(..., progress=True)`
- Users can disable: `BioLMApiClient(..., progress=False)`

### 8.2 Output Changes

**Current**: Text output to stderr via `debug()`
**New**: Progress bars (if `progress=True`) + optional text output (if `DEBUG=1`)

**Compatibility**:
- Keep `debug()` calls for backward compatibility
- Add flag to disable progress bars: `progress=False`

---

## 9. Future Enhancements

### 9.1 Rich Display (Phase 5 - Optional)

**Features**:
- Resource status table with colors
- Budget visualization
- Algorithm status dashboard
- Multi-line layout for complex displays

**Implementation**:
- Add `rich` as optional dependency
- Use `progress_style="rich"` parameter
- Create `RichProgressManager` class

### 9.2 Advanced Features

**Ideas**:
- Progress bar persistence across notebook restarts
- Export progress data to file
- Custom progress bar themes
- WebSocket reconnection with state recovery
- Progress bar in web UI (if SDK used in web app)

---

## 10. Implementation Checklist

### Phase 1: Core Progress Bar
- [ ] Add `tqdm` to `pyproject.toml` (optional dependency)
- [ ] Create `biolmai/telemetry_progress.py`
- [ ] Implement `TelemetryProgressManager` class
- [ ] Add Jupyter/terminal auto-detection
- [ ] Wire `TelemetryListener` to progress manager
- [ ] Test single request progress bar

### Phase 2: Activity WebSocket
- [ ] Create `ActivityListener` class in `client.py`
- [ ] Add Activity WebSocket connection in `BioLMApiClient`
- [ ] Implement event parsing (`activity_update`, `billing_update`, `activity_hint`)
- [ ] Wire to progress manager for resource display
- [ ] Test Activity WebSocket connection

### Phase 3: Enhanced Display
- [ ] Implement batch progress aggregation
- [ ] Add resource status display
- [ ] Add budget information display
- [ ] Integrate activity hints
- [ ] Error state visualization

### Phase 4: Polish & Testing
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Manual testing in terminal
- [ ] Manual testing in Jupyter
- [ ] Update documentation
- [ ] Performance testing

---

## 11. Open Questions

1. **Should progress bars be enabled by default?**
   - **Recommendation**: Yes (`progress=True` by default)
   - Rationale: Better UX, can be disabled if needed

2. **Should we show progress for cached responses?**
   - **Recommendation**: Yes, but very brief (instant completion)
   - Shows user that request was processed (even if cached)

3. **How to handle multiple concurrent requests?**
   - **Recommendation**: Multiple progress bars (one per request)
   - Alternative: Single aggregated bar (less informative)

4. **Should Activity WebSocket be optional?**
   - **Recommendation**: Yes, only connect if `progress=True`
   - Reduces overhead for users who don't want progress

5. **What to show in resource status display?**
   - **Recommendation**: Active algorithms, total resources, budget
   - Keep it simple initially, enhance later

---

## 12. References

- [WebSocket API Documentation](./WEBSOCKET_API.md) (provided by user)
- [tqdm Documentation](https://github.com/tqdm/tqdm)
- [Modal Client](https://github.com/modal-labs/modal-client) - Reference implementation
- [pandarallel](https://github.com/nalepae/pandarallel) - Reference implementation
- [Rich Documentation](https://rich.readthedocs.io/) - Future enhancement option

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-04  
**Status**: Planning Phase (No code changes yet)

