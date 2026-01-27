# -*- coding: utf-8 -*-
"""Memory monitoring and automatic resource management."""

import logging
import gc
import threading
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    current_mb: float
    peak_mb: float
    available_mb: float
    percent_used: float
    process_mb: float
    timestamp: float


@dataclass
class MemoryLimits:
    """Memory usage limits and thresholds."""

    max_memory_mb: float = 2048.0  # Maximum memory to use
    warning_threshold: float = 0.8  # Warning at 80% of max
    critical_threshold: float = 0.9  # Critical at 90% of max
    gc_threshold: float = 0.7  # Force GC at 70% of max


class MemoryMonitor:
    """Real-time memory monitoring with automatic limits."""

    def __init__(
        self,
        limits: Optional[MemoryLimits] = None,
        check_interval: float = 1.0,
        auto_gc: bool = True,
    ):
        """Initialize memory monitor.

        Args:
            limits: Memory limits configuration.
            check_interval: How often to check memory (seconds).
            auto_gc: Whether to automatically trigger garbage collection.
        """
        self.limits = limits or MemoryLimits()
        self.check_interval = check_interval
        self.auto_gc = auto_gc

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, Callable] = {}
        self._stats_history: list[MemoryStats] = []
        self._max_history = 1000

        # Peak tracking
        self._peak_memory = 0.0

        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available. Memory monitoring will be limited.")

    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")

    def add_callback(self, name: str, callback: Callable[[MemoryStats], None]):
        """Add callback for memory events.

        Args:
            name: Callback name.
            callback: Function to call with memory stats.
        """
        self._callbacks[name] = callback

    def remove_callback(self, name: str):
        """Remove memory callback.

        Args:
            name: Callback name to remove.
        """
        self._callbacks.pop(name, None)

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if PSUTIL_AVAILABLE:
            # System memory
            memory = psutil.virtual_memory()
            current_mb = memory.used / (1024 * 1024)
            available_mb = memory.available / (1024 * 1024)
            percent_used = memory.percent

            # Process memory
            process = psutil.Process()
            process_mb = process.memory_info().rss / (1024 * 1024)
        else:
            # Fallback using os
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    # Parse memory info (Linux only)
                    mem_total = 0
                    mem_available = 0
                    for line in meminfo.split("\n"):
                        if "MemTotal:" in line:
                            mem_total = int(line.split()[1]) / 1024
                        elif "MemAvailable:" in line:
                            mem_available = int(line.split()[1]) / 1024

                    current_mb = (mem_total - mem_available) / 1024
                    available_mb = mem_available / 1024
                    percent_used = ((mem_total - mem_available) / mem_total) * 100
                    process_mb = 0  # Not available without psutil
            except (OSError, IOError):
                # Default fallback values
                current_mb = 0
                available_mb = self.limits.max_memory_mb
                percent_used = 0
                process_mb = 0

        # Update peak
        self._peak_memory = max(self._peak_memory, current_mb)

        stats = MemoryStats(
            current_mb=current_mb,
            peak_mb=self._peak_memory,
            available_mb=available_mb,
            percent_used=percent_used,
            process_mb=process_mb,
            timestamp=time.time(),
        )

        # Add to history
        self._stats_history.append(stats)
        if len(self._stats_history) > self._max_history:
            self._stats_history.pop(0)

        return stats

    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)."""
        stats = self.get_current_stats()
        return min(1.0, stats.current_mb / self.limits.max_memory_mb)

    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced due to memory pressure."""
        pressure = self.get_memory_pressure()
        return pressure >= self.limits.warning_threshold

    def should_force_gc(self) -> bool:
        """Check if garbage collection should be forced."""
        pressure = self.get_memory_pressure()
        return pressure >= self.limits.gc_threshold

    def is_critical_memory(self) -> bool:
        """Check if memory usage is critical."""
        pressure = self.get_memory_pressure()
        return pressure >= self.limits.critical_threshold

    def force_garbage_collection(self):
        """Force garbage collection and return memory freed."""
        if not self.auto_gc:
            return 0

        before_stats = self.get_current_stats()

        # Force garbage collection
        gc.collect()

        after_stats = self.get_current_stats()
        freed_mb = before_stats.current_mb - after_stats.current_mb

        if freed_mb > 0:
            logger.info(f"Garbage collection freed {freed_mb:.1f} MB")

        return freed_mb

    def get_optimal_batch_size(
        self,
        current_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 100,
    ) -> int:
        """Calculate optimal batch size based on current memory usage."""
        pressure = self.get_memory_pressure()

        if pressure < 0.5:
            # Low pressure, can increase batch size
            return min(max_batch_size, current_batch_size * 2)
        elif pressure < 0.7:
            # Medium pressure, keep current size
            return current_batch_size
        elif pressure < 0.9:
            # High pressure, reduce batch size
            new_size = max(min_batch_size, current_batch_size // 2)
            logger.warning(
                f"High memory pressure ({pressure:.1%}), reducing batch size to {new_size}"
            )
            return new_size
        else:
            # Critical pressure, use minimum batch size
            logger.warning(
                f"Critical memory pressure ({pressure:.1%}), using minimum batch size"
            )
            return min_batch_size

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_current_stats()

                # Check thresholds and trigger callbacks
                pressure = stats.current_mb / self.limits.max_memory_mb

                if pressure >= self.limits.critical_threshold:
                    logger.critical(
                        f"Critical memory usage: {stats.current_mb:.1f} MB "
                        f"({pressure:.1%} of limit)"
                    )
                    self._trigger_callbacks("critical", stats)

                    # Force GC
                    if self.auto_gc:
                        self.force_garbage_collection()

                elif pressure >= self.limits.warning_threshold:
                    logger.warning(
                        f"High memory usage: {stats.current_mb:.1f} MB "
                        f"({pressure:.1%} of limit)"
                    )
                    self._trigger_callbacks("warning", stats)

                elif pressure >= self.limits.gc_threshold:
                    if self.auto_gc:
                        self.force_garbage_collection()

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.check_interval)

    def _trigger_callbacks(self, event_type: str, stats: MemoryStats):
        """Trigger registered callbacks."""
        for name, callback in self._callbacks.items():
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"Error in memory callback '{name}': {e}")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        current_stats = self.get_current_stats()

        report = {
            "current": {
                "memory_mb": current_stats.current_mb,
                "available_mb": current_stats.available_mb,
                "percent_used": current_stats.percent_used,
                "process_mb": current_stats.process_mb,
                "peak_mb": current_stats.peak_mb,
            },
            "limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "warning_threshold": self.limits.warning_threshold,
                "critical_threshold": self.limits.critical_threshold,
                "gc_threshold": self.limits.gc_threshold,
            },
            "pressure": {
                "current_pressure": self.get_memory_pressure(),
                "should_reduce_batch": self.should_reduce_batch_size(),
                "should_force_gc": self.should_force_gc(),
                "is_critical": self.is_critical_memory(),
            },
            "monitoring": {
                "active": self._monitoring,
                "check_interval": self.check_interval,
                "auto_gc": self.auto_gc,
                "history_count": len(self._stats_history),
            },
        }

        # Add trend analysis if we have history
        if len(self._stats_history) >= 2:
            recent = self._stats_history[-10:]  # Last 10 samples
            memory_trend = recent[-1].current_mb - recent[0].current_mb
            report["trend"] = {
                "recent_change_mb": memory_trend,
                "samples_analyzed": len(recent),
            }

        return report


class AdaptiveBatchManager:
    """Batch manager that adapts to memory conditions."""

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        memory_monitor: Optional[MemoryMonitor] = None,
    ):
        """Initialize adaptive batch manager.

        Args:
            initial_batch_size: Starting batch size.
            min_batch_size: Minimum allowed batch size.
            max_batch_size: Maximum allowed batch size.
            memory_monitor: Memory monitor instance.
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_monitor = memory_monitor or MemoryMonitor()

        self.current_batch_size = initial_batch_size
        self.performance_history: list[Dict[str, Any]] = []

        # Start monitoring if not already active
        if not self.memory_monitor._monitoring:
            self.memory_monitor.start_monitoring()

    def get_adaptive_batch_size(self) -> int:
        """Get batch size adapted to current memory conditions."""
        if self.memory_monitor:
            return self.memory_monitor.get_optimal_batch_size(
                self.current_batch_size, self.min_batch_size, self.max_batch_size
            )
        return self.current_batch_size

    def update_performance(
        self, batch_size: int, processing_time: float, success: bool
    ):
        """Update performance history for batch size optimization.

        Args:
            batch_size: Batch size used.
            processing_time: Time taken to process batch.
            success: Whether processing was successful.
        """
        performance = {
            "batch_size": batch_size,
            "processing_time": processing_time,
            "success": success,
            "throughput": batch_size / processing_time if processing_time > 0 else 0,
            "timestamp": time.time(),
        }

        self.performance_history.append(performance)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        # Update current batch size based on performance
        self._optimize_batch_size()

    def _optimize_batch_size(self):
        """Optimize batch size based on performance history."""
        if len(self.performance_history) < 5:
            return

        # Analyze recent performance
        recent = self.performance_history[-10:]
        successful = [p for p in recent if p["success"]]

        if not successful:
            return

        # Calculate average throughput by batch size
        throughput_by_size = {}
        for perf in successful:
            size = perf["batch_size"]
            if size not in throughput_by_size:
                throughput_by_size[size] = []
            throughput_by_size[size].append(perf["throughput"])

        # Find best performing batch size
        best_size = self.current_batch_size
        best_throughput = 0

        for size, throughputs in throughput_by_size.items():
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_size = size

        # Update if we found a better size
        if best_size != self.current_batch_size:
            logger.info(
                f"Optimizing batch size: {self.current_batch_size} -> {best_size}"
            )
            self.current_batch_size = best_size

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report."""
        if not self.performance_history:
            return {"status": "no_data"}

        recent = self.performance_history[-20:]
        successful = [p for p in recent if p["success"]]

        if not successful:
            return {"status": "no_successful_batches"}

        avg_throughput = sum(p["throughput"] for p in successful) / len(successful)
        avg_batch_size = sum(p["batch_size"] for p in successful) / len(successful)

        return {
            "status": "active",
            "current_batch_size": self.current_batch_size,
            "avg_batch_size": avg_batch_size,
            "avg_throughput": avg_throughput,
            "total_batches": len(self.performance_history),
            "success_rate": len(successful) / len(recent),
            "memory_pressure": self.memory_monitor.get_memory_pressure()
            if self.memory_monitor
            else 0,
        }
