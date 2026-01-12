"""Unit tests for TelemetryProgressManager."""

import pytest

# tqdm is optional, skip tests if not available
pytest.importorskip("tqdm")

from biolmai.telemetry_progress import TelemetryProgressManager


class TestTelemetryProgressManager:
    """Test suite for TelemetryProgressManager."""

    def test_init_enabled(self):
        """Test initialization with progress enabled."""
        manager = TelemetryProgressManager(enable=True)
        assert manager.enable is True
        assert manager._algorithm_bars == {}
        assert manager._request_tracking == {}
        assert manager._resource_status is None
        assert manager._cache_hits == {}

    def test_init_disabled(self):
        """Test initialization with progress disabled."""
        manager = TelemetryProgressManager(enable=False)
        assert manager.enable is False

    def test_start_request_creates_bar(self):
        """Test that start_request creates a progress bar for new algorithm."""
        manager = TelemetryProgressManager(enable=True)
        manager.start_request("req1", "esmfold", "predict", 5)

        assert "esmfold" in manager._algorithm_bars
        assert "req1" in manager._request_tracking
        assert manager._request_tracking["req1"]["model"] == "esmfold"
        assert manager._request_tracking["req1"]["action"] == "predict"
        assert manager._request_tracking["req1"]["n_items"] == 5
        assert manager._request_tracking["req1"]["status"] == "pending"

        algo_state = manager._algorithm_bars["esmfold"]
        assert "req1" in algo_state["active_requests"]
        assert algo_state["total_items"] == 5

    def test_multiple_requests_same_algorithm(self):
        """Test that multiple requests for same algorithm aggregate."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.start_request("req2", "esmfold", "predict", 3)
        manager.start_request("req3", "esmfold", "encode", 2)

        assert "esmfold" in manager._algorithm_bars
        algo_state = manager._algorithm_bars["esmfold"]
        assert len(algo_state["active_requests"]) == 3
        assert algo_state["total_items"] == 10  # 5 + 3 + 2
        assert "predict" in algo_state["actions"]
        assert "encode" in algo_state["actions"]

    def test_different_algorithms_separate_bars(self):
        """Test that different algorithms get separate progress bars."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.start_request("req2", "esm2-8m", "encode", 3)

        assert "esmfold" in manager._algorithm_bars
        assert "esm2-8m" in manager._algorithm_bars
        assert len(manager._algorithm_bars) == 2

    def test_submit_request_updates_total(self):
        """Test that submit_request updates bar total with backend_items."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 10)
        manager.submit_request("req1", 7)  # 3 items cached

        req_state = manager._request_tracking["req1"]
        assert req_state["backend_items"] == 7
        assert req_state["status"] == "submitted"

        algo_state = manager._algorithm_bars["esmfold"]
        # Total should be adjusted: 10 - 10 + 7 = 7
        assert algo_state["total_items"] == 7

    def test_cache_hit_completes_immediately(self):
        """Test that cache_hit completes progress bar immediately."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.cache_hit("req1")

        req_state = manager._request_tracking["req1"]
        assert req_state["status"] == "cached"

        algo_state = manager._algorithm_bars["esmfold"]
        assert "req1" not in algo_state["active_requests"]
        assert manager._cache_hits["esmfold"] == 1

    def test_finish_request_updates_elapsed(self):
        """Test that finish_request updates elapsed time."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.submit_request("req1", 5)
        manager.finish_request("req1", 12.5)

        req_state = manager._request_tracking["req1"]
        assert req_state["status"] == "finished"
        assert req_state["elapsed"] == 12.5

    def test_error_request_updates_status(self):
        """Test that error_request marks request as error."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.error_request("req1", 400)

        req_state = manager._request_tracking["req1"]
        assert req_state["status"] == "error"

        algo_state = manager._algorithm_bars["esmfold"]
        assert "req1" not in algo_state["active_requests"]

    def test_complete_request_removes_tracking(self):
        """Test that complete_request removes request from tracking."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.submit_request("req1", 5)
        manager.finish_request("req1", 10.0)
        manager.complete_request("req1")

        assert "req1" not in manager._request_tracking
        # When no active requests remain, the bar is closed and removed
        assert "esmfold" not in manager._algorithm_bars

    def test_complete_request_closes_bar_when_no_active(self):
        """Test that complete_request closes bar when no active requests."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.submit_request("req1", 5)
        manager.finish_request("req1", 10.0)
        manager.complete_request("req1")

        # Bar should be closed and removed
        assert "esmfold" not in manager._algorithm_bars

    def test_update_resources(self):
        """Test that update_resources updates resource status."""
        manager = TelemetryProgressManager(enable=True)

        activity_data = {
            "algorithms": {
                "esmfold": {
                    "containers": 2,
                    "cpu_cores": 8,
                    "memory_gb": 16.0,
                    "gpus": 1,
                }
            },
            "totals": {
                "containers": 2,
                "cpu_cores": 8,
                "memory_gb": 16.0,
                "gpus": 1,
            },
        }

        manager.update_resources(activity_data)
        assert manager._resource_status == activity_data

    def test_cache_hits_counted_per_algorithm(self):
        """Test that cache hits are counted per algorithm."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.cache_hit("req1")

        manager.start_request("req2", "esmfold", "predict", 3)
        manager.cache_hit("req2")

        manager.start_request("req3", "esm2-8m", "encode", 2)
        manager.cache_hit("req3")

        assert manager._cache_hits["esmfold"] == 2
        assert manager._cache_hits["esm2-8m"] == 1

    def test_concurrent_requests_aggregation(self):
        """Test that concurrent requests aggregate correctly."""
        manager = TelemetryProgressManager(enable=True)

        # Start multiple concurrent requests
        manager.start_request("req1", "esmfold", "predict", 5)
        manager.start_request("req2", "esmfold", "predict", 3)
        manager.start_request("req3", "esmfold", "encode", 2)

        algo_state = manager._algorithm_bars["esmfold"]
        assert algo_state["total_items"] == 10
        assert len(algo_state["active_requests"]) == 3

        # Submit some
        manager.submit_request("req1", 5)
        manager.submit_request("req2", 2)  # 1 cached

        # Total should be: 10 - 5 - 3 + 5 + 2 = 9
        assert algo_state["total_items"] == 9

        # Finish and complete
        manager.finish_request("req1", 10.0)
        manager.complete_request("req1")

        # Should still have 2 active
        assert len(algo_state["active_requests"]) == 2

    def test_close_all_cleans_up(self):
        """Test that close_all cleans up all bars."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.start_request("req2", "esm2-8m", "encode", 3)

        manager.close_all()

        assert len(manager._algorithm_bars) == 0
        assert len(manager._request_tracking) == 0
        assert manager._resource_line is None

    def test_disabled_manager_no_ops(self):
        """Test that disabled manager methods are no-ops."""
        manager = TelemetryProgressManager(enable=False)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.submit_request("req1", 5)
        manager.cache_hit("req1")
        manager.finish_request("req1", 10.0)
        manager.error_request("req1", 400)
        manager.complete_request("req1")
        manager.update_resources({})
        manager.close_all()

        # Should not have created any bars
        assert len(manager._algorithm_bars) == 0

    def test_missing_request_id_handled_gracefully(self):
        """Test that methods handle missing request_id gracefully."""
        manager = TelemetryProgressManager(enable=True)

        # These should not raise exceptions
        manager.submit_request("nonexistent", 5)
        manager.cache_hit("nonexistent")
        manager.finish_request("nonexistent", 10.0)
        manager.error_request("nonexistent", 400)
        manager.complete_request("nonexistent")

        assert len(manager._algorithm_bars) == 0

    def test_resource_display_with_cache_hits(self):
        """Test resource display includes cache hits."""
        manager = TelemetryProgressManager(enable=True)

        # Add some cache hits
        manager.start_request("req1", "esmfold", "predict", 5)
        manager.cache_hit("req1")
        manager.start_request("req2", "esm2-8m", "encode", 3)
        manager.cache_hit("req2")

        # Update resources
        activity_data = {
            "algorithms": {
                "esmfold": {"gpus": 1, "cpu_cores": 4, "memory_gb": 8.0}
            },
            "totals": {"gpus": 1, "cpu_cores": 4, "memory_gb": 8.0},
        }
        manager.update_resources(activity_data)

        # Resource line should be created
        assert manager._resource_line is not None

    def test_multiple_actions_in_description(self):
        """Test that bar description includes all active actions."""
        manager = TelemetryProgressManager(enable=True)

        manager.start_request("req1", "esmfold", "predict", 5)
        manager.start_request("req2", "esmfold", "encode", 3)

        algo_state = manager._algorithm_bars["esmfold"]
        # Description should include both actions
        desc = algo_state["bar"].desc
        assert "predict" in desc or "encode" in desc

