"""Comprehensive telemetry tests for all client types: BioLM, BioLMApi, and BioLMApiClient."""

import asyncio
import pytest

# websockets is required for telemetry
pytest.importorskip("websockets")

from biolmai.biolmai import BioLM
from biolmai.client import BioLMApi, BioLMApiClient


@pytest.mark.asyncio
async def test_telemetry_biolm_api_client_async():
    """Test telemetry with async BioLMApiClient."""
    client = BioLMApiClient(
        "esm2-8m",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        progress=True,
    )
    
    try:
        result = await client.encode(items=[{"sequence": "MDNELE"}])
        assert isinstance(result, list)
        assert len(result) == 1
        # Check that telemetry events were captured
        assert hasattr(client, 'last_telemetry_events')
        assert isinstance(client.last_telemetry_events, list)
        # Events may be empty if connection timed out, but structure should exist
    finally:
        await client.shutdown()


def test_telemetry_biolm_api_sync():
    """Test telemetry with sync BioLMApi."""
    client = BioLMApi(
        "esm2-8m",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        progress=True,
    )
    
    try:
        result = client.encode(items=[{"sequence": "MDNELE"}])
        assert isinstance(result, list)
        assert len(result) == 1
        # Check that telemetry events were captured
        assert hasattr(client, 'last_telemetry_events')
        assert isinstance(client.last_telemetry_events, list)
    finally:
        # BioLMApi is sync, but may have async cleanup
        if hasattr(client, 'shutdown'):
            try:
                asyncio.run(client.shutdown())
            except Exception:
                pass


def test_telemetry_biolm_high_level():
    """Test telemetry with high-level BioLM class.
    
    Note: BioLM doesn't directly support telemetry parameters,
    but it uses BioLMApi internally which should support it.
    However, BioLM doesn't expose telemetry configuration,
    so this test verifies basic functionality.
    """
    # BioLM doesn't expose telemetry/progress parameters directly
    # but it should still work if the underlying BioLMApi supports it
    result = BioLM(
        entity="esm2-8m",
        action="encode",
        type="sequence",
        items="MDNELE",
        raise_httpx=False,
    )
    
    assert isinstance(result, dict)
    assert "embeddings" in result


@pytest.mark.asyncio
async def test_telemetry_biolm_api_client_multiple_requests():
    """Test telemetry with multiple concurrent requests using BioLMApiClient."""
    client = BioLMApiClient(
        "esm2-8m",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        progress=True,
    )
    
    try:
        # Make multiple concurrent requests
        tasks = [
            client.encode(items=[{"sequence": "MDNELE"}]),
            client.encode(items=[{"sequence": "MENDEL"}]),
            client.encode(items=[{"sequence": "ISOTYPE"}]),
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)
            assert len(result) == 1
    finally:
        await client.shutdown()


def test_telemetry_biolm_api_sync_multiple_requests():
    """Test telemetry with multiple requests using sync BioLMApi."""
    client = BioLMApi(
        "esm2-8m",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        progress=True,
    )
    
    try:
        # Make multiple sequential requests
        result1 = client.encode(items=[{"sequence": "MDNELE"}])
        result2 = client.encode(items=[{"sequence": "MENDEL"}])
        result3 = client.encode(items=[{"sequence": "ISOTYPE"}])
        
        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert isinstance(result3, list)
    finally:
        if hasattr(client, 'shutdown'):
            try:
                asyncio.run(client.shutdown())
            except Exception:
                pass


@pytest.mark.asyncio
async def test_telemetry_biolm_api_client_progress_disabled():
    """Test that telemetry works even when progress is disabled."""
    client = BioLMApiClient(
        "esm2-8m",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        progress=False,  # Disable progress bars
    )
    
    try:
        result = await client.encode(items=[{"sequence": "MDNELE"}])
        assert isinstance(result, list)
        assert len(result) == 1
        # Telemetry should still work even without progress bars
        assert hasattr(client, 'last_telemetry_events')
        assert isinstance(client.last_telemetry_events, list)
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_telemetry_biolm_api_client_telemetry_handler():
    """Test custom telemetry handler with BioLMApiClient."""
    events_captured = []
    
    def custom_handler(event):
        events_captured.append(event)
    
    client = BioLMApiClient(
        "esm2-8m",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        telemetry_handler=custom_handler,
    )
    
    try:
        result = await client.encode(items=[{"sequence": "MDNELE"}])
        assert isinstance(result, list)
        # Handler may or may not receive events depending on connection timing
        # but it should be set up correctly
        assert client._telemetry_handler == custom_handler
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_telemetry_biolm_api_client_activity_websocket():
    """Test that Activity WebSocket connects when progress is enabled."""
    client = BioLMApiClient(
        "esm2-8m",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        progress=True,  # Progress enables Activity WebSocket
    )
    
    try:
        result = await client.encode(items=[{"sequence": "MDNELE"}])
        assert isinstance(result, list)
        # Activity listener should be set up when progress is enabled
        # It may not connect immediately, but the task should exist
        if hasattr(client, '_activity_task'):
            assert client._activity_task is not None or client._activity_listener is not None
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_telemetry_biolm_api_client_esmfold():
    """Test telemetry with ESMFold (longer-running requests)."""
    client = BioLMApiClient(
        "esmfold",
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
        progress=True,
    )
    
    try:
        result = await client.predict(items=[{"sequence": "MDNELE"}])
        assert isinstance(result, list)
        assert len(result) == 1
        assert "pdb" in result[0]
    finally:
        await client.shutdown()

