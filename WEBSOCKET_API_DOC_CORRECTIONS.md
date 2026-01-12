# WebSocket API Documentation Corrections

This document contains corrections and clarifications for the WebSocket API documentation based on the actual Python SDK implementation.

## Key Corrections

### 1. Channel Usage Patterns

**Two Patterns Supported:**

#### Pattern A: Per-Request Channel (Documented Pattern)
- Create a unique channel per API request: `telemetry_<request_id>`
- Connect to WebSocket before making the request
- Close WebSocket after receiving `response_sent` or `error` event
- Use the same `request_id` in both channel name and `X-Request-Id` header

#### Pattern B: Persistent Channel (Python SDK Pattern)
- Create a single persistent channel per client instance: `telemetry_<channel_id>`
- Reuse the same WebSocket connection for all requests from that client
- Each request gets a unique `request_id` sent in `X-Request-Id` header
- Server correlates events by matching `request_id` in event payload to `X-Request-Id` header
- Channel remains open for the lifetime of the client

**Python SDK Implementation:**
```python
# Single channel per client instance
self._telemetry_channel = f"telemetry_{uuid.uuid4().hex}"
# Sent in every request header
headers["X-Telemetry-Channel"] = self._telemetry_channel

# Unique request_id per API call
request_id = uuid.uuid4().hex
headers["X-Request-Id"] = request_id
```

### 2. Request ID Format Specification

**Exact Format:**
- **Length**: Exactly 32 characters
- **Format**: Lowercase hexadecimal (0-9, a-f)
- **Pattern**: `^[0-9a-f]{32}$`
- **Generation**: UUID4 hex format (no hyphens)
- **Example**: `03da65c1a2b3c4d5e6f7a8b9c0d1e2f3`

**Server Validation Regex:**
```python
import re
TELEMETRY_REQUEST_ID_PATTERN = re.compile(r'^[0-9a-f]{32}$')
```

### 3. Request ID Field Names in Events

**Both Fields Supported:**
Events may include either `request_id` or `rid` field (or both). The Python SDK checks both:
```python
request_id = data.get("request_id") or data.get("rid")
```

**Documentation should specify:**
- Primary field: `request_id` (recommended)
- Alternative field: `rid` (for backward compatibility)
- Both fields may be present, `request_id` takes precedence

### 4. X-Request-Id Header

**Clarification:**
- **Required** when using persistent channel pattern (Pattern B)
- **Optional but recommended** when using per-request channel pattern (Pattern A)
- Used for correlating telemetry events to specific API requests
- Must match the `request_id` in telemetry event payloads

### 5. Authentication Header Formats

**Correct Formats:**

1. **API Token Authentication:**
   ```
   Authorization: Token <api_token>
   ```
   - Used when `BIOLMAI_TOKEN` environment variable is set
   - Used when `api_key` parameter is provided to client

2. **JWT Session Authentication:**
   ```
   Cookie: access=<access_token>;refresh=<refresh_token>
   Content-Type: application/json
   ```
   - Used when credentials file exists at `~/.biolmai/credentials`
   - Contains both `access` and `refresh` tokens

3. **Bearer Token (if supported):**
   ```
   Authorization: Bearer <jwt_access_token>
   ```
   - May be used for direct JWT access tokens (verify server support)

**WebSocket Headers:**
WebSocket connections use the same authentication headers as REST API calls, converted to tuple format for the `websockets` library:
```python
extra_headers = [(k, v) for k, v in headers.items()]
```

### 6. Activity WebSocket Subscription

**Automatic Subscription:**
The server automatically subscribes the connected client to groups based on the authenticated user's context:
- Personal group: `user_activity_<user_id>`
- Institute group: `activity.inst.<institute_id>`
- Environment group: `activity.inst.<institute_id>.env.<environment_id>`

**No Initial Message:**
- The server does **NOT** send an initial `activity_update` message upon connection
- Clients should fetch initial state via REST API (`/api/activity-rollup/`) before connecting
- WebSocket will only send updates when activity changes occur

### 7. Model Slug Format

**Correct Format:**
- Use hyphens, not underscores: `protein-mpnn` (not `mpnn_protein`)
- Examples: `esmfold`, `esm2-8m`, `protein-mpnn`, `peptides`

**Update all examples in documentation:**
- Change `"mpnn_protein"` → `"protein-mpnn"`
- Verify all model slugs match actual API endpoints

### 8. Connection Timing

**Two Approaches:**

#### Approach A: Connect Before Request (Recommended for Per-Request Pattern)
- Connect to WebSocket before making API request
- Ensures no events are missed
- Close after request completes

#### Approach B: Lazy Connection (Python SDK Pattern)
- Connect on first API call
- Use retry logic (up to 3 attempts, 20s timeout each)
- Keep connection open for reuse
- May miss early events on first request, but subsequent requests benefit from persistent connection

### 9. Event Field Names

**Standard Fields:**
- `event`: Event type (required)
- `request_id` or `rid`: Request identifier (required for correlation)
- `model`: Model slug (e.g., "protein-mpnn")
- `action`: Action name (e.g., "predict", "encode", "generate")
- Additional fields vary by event type

### 10. Activity WebSocket Initial State

**Correction:**
- The server does **NOT** send an initial `activity_update` message upon connection
- Clients must fetch initial state via REST API endpoint: `GET /api/activity-rollup/`
- WebSocket will only send updates when activity changes occur after connection

## Updated Usage Pattern Section

### Recommended Usage Pattern for Python SDK

For the Python SDK to immediately show activity status:

1. **Before making API request**: Fetch initial state via REST API
   ```python
   # Get current activity state
   activity_resp = requests.get(
       "https://api.biolm.ai/api/activity-rollup/",
       headers={"Authorization": "Token your_api_token"}
   )
   current_state = activity_resp.json()
   # Display current algorithms, containers, GPUs, etc.
   ```

2. **Connect to Activity WebSocket**: Subscribe to real-time updates
   ```python
   ws = websocket.WebSocketApp(
       "wss://api.biolm.ai/ws/activity/",
       header=["Authorization: Token your_api_token"],
       on_message=lambda ws, msg: handle_activity_update(json.loads(msg))
   )
   ```
   **Note**: No initial message is sent; only updates after connection.

3. **Make API request with Telemetry WebSocket**: For per-request progress
   
   **Pattern A: Per-Request Channel**
   ```python
   request_id = uuid.uuid4().hex  # 32-char hex string
   telemetry_channel = f"telemetry_{request_id}"
   
   # Connect before request
   telemetry_ws = connect_telemetry_ws(telemetry_channel)
   
   response = requests.post(
       "https://api.biolm.ai/api/v3/protein-mpnn/predict/",
       headers={
           "Authorization": "Token your_api_token",
           "X-Telemetry-Channel": telemetry_channel,
           "X-Request-Id": request_id,  # Required for correlation
       },
       json={"items": [...]}
   )
   
   # Close after response
   telemetry_ws.close()
   ```
   
   **Pattern B: Persistent Channel (Python SDK)**
   ```python
   # Single channel per client instance
   client = BioLMApiClient("protein-mpnn", telemetry=True)
   # Channel created automatically: telemetry_<channel_id>
   # Each request gets unique request_id in X-Request-Id header
   
   result = await client.predict(items=[...])
   # WebSocket stays open for subsequent requests
   ```

## Updated Channel Format Section

### Channel Format

The channel parameter must follow the format: `telemetry_<identifier>`

Where `<identifier>` is:
- **Pattern A**: A unique 32-character lowercase hexadecimal string (UUID4 hex) per request
- **Pattern B**: A unique 32-character lowercase hexadecimal string (UUID4 hex) per client instance

**Format Validation**: `^telemetry_[0-9a-f]{32}$`

**Examples**:
- `telemetry_03da65c1a2b3c4d5e6f7a8b9c0d1e2f3` (valid)
- `telemetry_abc123` (invalid - too short)
- `telemetry_ABC123DEF456` (invalid - uppercase)

## Updated Event Examples

All event examples should use correct model slug format:

```json
{
  "event": "request_start",
  "model": "protein-mpnn",  // Not "mpnn_protein"
  "action": "predict",
  "n_items": 5,
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "03da65c1a2b3c4d5e6f7a8b9c0d1e2f3"  // 32-char hex
}
```

**Note**: Events may also include `rid` field as alternative to `request_id`:
```json
{
  "event": "call_submitted",
  "model": "protein-mpnn",
  "action": "predict",
  "backend_items": 3,
  "request_id": "03da65c1a2b3c4d5e6f7a8b9c0d1e2f3",
  "rid": "03da65c1a2b3c4d5e6f7a8b9c0d1e2f3"  // Alternative field
}
```

## Summary of Required Documentation Updates

1. ✅ Add section explaining two channel usage patterns (per-request vs persistent)
2. ✅ Specify exact request ID format: `^[0-9a-f]{32}$` (32 lowercase hex chars)
3. ✅ Document that events may use `request_id` or `rid` field
4. ✅ Clarify authentication header formats (Token vs Cookie vs Bearer)
5. ✅ Fix model slug examples (`protein-mpnn` not `mpnn_protein`)
6. ✅ Add note about Python SDK's persistent channel pattern
7. ✅ Clarify that Activity WebSocket does NOT send initial message
8. ✅ Update X-Request-Id header to be required for persistent channel pattern
9. ✅ Document connection timing approaches (before vs lazy)
10. ✅ Remove mention of initial activity_update message


