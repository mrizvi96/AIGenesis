"""
Cloud Performance Monitoring System for Qdrant Cloud Deployment
Real-time monitoring of resource usage, performance metrics, and system health
Optimized for cloud free tier constraints with intelligent alerting
"""

import time
import json
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class PerformanceStatus(Enum):
    """System performance status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    active_threads: int
    open_files: int

@dataclass
class CloudMetrics:
    """Cloud-specific performance metrics"""
    timestamp: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    qdrant_connection_status: bool
    active_components: int
    memory_efficiency_score: float
    storage_efficiency_score: float

@dataclass
class Alert:
    """Performance alert"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metrics_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class CloudPerformanceMonitor:
    """
    Real-time performance monitoring for Qdrant Cloud deployment
    Provides intelligent alerting and performance optimization recommendations
    """

    def __init__(self, monitoring_interval: int = 30, history_size: int = 1000):
        logger.info(f"[CLOUD-MONITOR] Initializing performance monitor (interval: {monitoring_interval}s)...")

        # Monitoring configuration
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.running = False
        self.monitor_thread = None

        # Performance data storage
        self.performance_history = deque(maxlen=history_size)
        self.cloud_history = deque(maxlen=history_size)
        self.alerts = {}
        self.alert_callbacks = []

        # Resource limits and thresholds
        self.memory_limit_mb = 1024  # Qdrant Cloud free tier
        self.cpu_limit_percent = 50.0  # 0.5 vCPU equivalent

        # Performance thresholds
        self.thresholds = {
            'memory_warning': 0.75,    # 75% of memory limit
            'memory_critical': 0.85,   # 85% of memory limit
            'memory_emergency': 0.95,  # 95% of memory limit
            'cpu_warning': 0.70,       # 70% of CPU limit
            'cpu_critical': 0.85,      # 85% of CPU limit
            'response_time_warning': 5000,   # 5 seconds
            'response_time_critical': 10000,  # 10 seconds
            'error_rate_warning': 0.05,      # 5% error rate
            'error_rate_critical': 0.10      # 10% error rate
        }

        # Cloud-specific metrics
        self.cloud_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': deque(maxlen=1000),  # Last 1000 response times
            'qdrant_connection_tests': deque(maxlen=100),  # Last 100 connection tests
            'component_status': {}
        }

        # Performance optimization tracking
        self.optimization_suggestions = []
        self.last_optimization_check = datetime.now()

        logger.info("[OK] Performance monitor initialized")

    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if self.running:
            logger.warning("[WARN] Performance monitoring already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("[OK] Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        logger.info("[MONITOR] Stopping performance monitoring...")

        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("[OK] Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system performance metrics
                performance_metrics = self._collect_performance_metrics()
                self.performance_history.append(performance_metrics)

                # Collect cloud-specific metrics
                cloud_metrics = self._collect_cloud_metrics()
                self.cloud_history.append(cloud_metrics)

                # Check for performance alerts
                self._check_performance_alerts(performance_metrics, cloud_metrics)

                # Update optimization suggestions
                if (datetime.now() - self.last_optimization_check).seconds > 300:  # Every 5 minutes
                    self._update_optimization_suggestions()
                    self.last_optimization_check = datetime.now()

                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"[ERROR] Performance monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect system performance metrics"""
        try:
            process = psutil.Process()

            # CPU and memory metrics
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_rss_mb = memory_info.rss / (1024 * 1024)
            memory_vms_mb = memory_info.vms / (1024 * 1024)
            memory_percent = (memory_rss_mb / self.memory_limit_mb) * 100

            # I/O metrics
            io_counters = process.io_counters()
            disk_io_read_mb = io_counters.read_bytes / (1024 * 1024)
            disk_io_write_mb = io_counters.write_bytes / (1024 * 1024)

            # Network metrics
            net_io = psutil.net_io_counters()
            network_io_sent_mb = net_io.bytes_sent / (1024 * 1024)
            network_io_recv_mb = net_io.bytes_recv / (1024 * 1024)

            # System metrics
            active_threads = process.num_threads()
            open_files = process.num_fds() if hasattr(process, 'num_fds') else 0

            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_rss_mb=memory_rss_mb,
                memory_vms_mb=memory_vms_mb,
                memory_percent=memory_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_io_sent_mb=network_io_sent_mb,
                network_io_recv_mb=network_io_recv_mb,
                active_threads=active_threads,
                open_files=open_files
            )

        except Exception as e:
            logger.error(f"[ERROR] Failed to collect performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0, memory_rss_mb=0.0, memory_vms_mb=0.0,
                memory_percent=0.0, disk_io_read_mb=0.0, disk_io_write_mb=0.0,
                network_io_sent_mb=0.0, network_io_recv_mb=0.0,
                active_threads=0, open_files=0
            )

    def _collect_cloud_metrics(self) -> CloudMetrics:
        """Collect cloud-specific performance metrics"""
        try:
            # Calculate request metrics
            total_requests = self.cloud_metrics['total_requests']
            successful_requests = self.cloud_metrics['successful_requests']
            failed_requests = self.cloud_metrics['failed_requests']

            # Calculate response time metrics
            response_times = list(self.cloud_metrics['response_times'])
            if response_times:
                average_response_time_ms = np.mean(response_times)
                p95_response_time_ms = np.percentile(response_times, 95)
            else:
                average_response_time_ms = 0.0
                p95_response_time_ms = 0.0

            # Test Qdrant connection
            qdrant_connection_status = self._test_qdrant_connection()

            # Count active components
            active_components = len(self.cloud_metrics['component_status'])

            # Calculate efficiency scores
            memory_efficiency_score = self._calculate_memory_efficiency()
            storage_efficiency_score = self._calculate_storage_efficiency()

            return CloudMetrics(
                timestamp=datetime.now(),
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time_ms=average_response_time_ms,
                p95_response_time_ms=p95_response_time_ms,
                qdrant_connection_status=qdrant_connection_status,
                active_components=active_components,
                memory_efficiency_score=memory_efficiency_score,
                storage_efficiency_score=storage_efficiency_score
            )

        except Exception as e:
            logger.error(f"[ERROR] Failed to collect cloud metrics: {e}")
            return CloudMetrics(
                timestamp=datetime.now(),
                total_requests=0, successful_requests=0, failed_requests=0,
                average_response_time_ms=0.0, p95_response_time_ms=0.0,
                qdrant_connection_status=False, active_components=0,
                memory_efficiency_score=0.0, storage_efficiency_score=0.0
            )

    def _test_qdrant_connection(self) -> bool:
        """Test connection to Qdrant Cloud"""
        try:
            from qdrant_manager import get_qdrant_manager
            qdrant = get_qdrant_manager()
            status = qdrant.test_connection()

            # Track connection test results
            self.cloud_metrics['qdrant_connection_tests'].append(status)

            return status
        except Exception:
            self.cloud_metrics['qdrant_connection_tests'].append(False)
            return False

    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        try:
            if not self.performance_history:
                return 0.0

            latest_metrics = self.performance_history[-1]
            memory_usage_ratio = latest_metrics.memory_percent / 100.0

            # Efficiency score: 100% at 50% usage, decreasing to 0% at 100%
            if memory_usage_ratio <= 0.5:
                return 100.0
            else:
                return max(0.0, 100.0 - (memory_usage_ratio - 0.5) * 200)

        except Exception:
            return 0.0

    def _calculate_storage_efficiency(self) -> float:
        """Calculate storage efficiency score"""
        try:
            from cloud_storage_optimizer import get_cloud_storage_optimizer

            storage_optimizer = get_cloud_storage_optimizer()
            health_report = storage_optimizer.get_storage_health_report()

            storage_usage = health_report.get('storage_usage', {})
            usage_percent = storage_usage.get('usage_percent', 0.0)

            # Similar to memory efficiency: optimal at 50% usage
            if usage_percent <= 50:
                return 100.0
            else:
                return max(0.0, 100.0 - (usage_percent - 50) * 2)

        except Exception:
            return 0.0

    def _check_performance_alerts(self, performance: PerformanceMetrics, cloud: CloudMetrics):
        """Check for performance issues and generate alerts"""
        try:
            # Memory alerts
            memory_ratio = performance.memory_percent / 100.0
            if memory_ratio >= self.thresholds['memory_emergency']:
                self._create_alert(
                    AlertLevel.EMERGENCY,
                    "Memory Emergency",
                    f"Memory usage is critical: {performance.memory_percent:.1f}%",
                    memory_ratio,
                    self.thresholds['memory_emergency']
                )
            elif memory_ratio >= self.thresholds['memory_critical']:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "Memory Critical",
                    f"Memory usage is high: {performance.memory_percent:.1f}%",
                    memory_ratio,
                    self.thresholds['memory_critical']
                )
            elif memory_ratio >= self.thresholds['memory_warning']:
                self._create_alert(
                    AlertLevel.WARNING,
                    "Memory Warning",
                    f"Memory usage is elevated: {performance.memory_percent:.1f}%",
                    memory_ratio,
                    self.thresholds['memory_warning']
                )

            # CPU alerts
            cpu_ratio = performance.cpu_percent / self.cpu_limit_percent
            if cpu_ratio >= self.thresholds['cpu_critical']:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "CPU Critical",
                    f"CPU usage is high: {performance.cpu_percent:.1f}%",
                    cpu_ratio,
                    self.thresholds['cpu_critical']
                )
            elif cpu_ratio >= self.thresholds['cpu_warning']:
                self._create_alert(
                    AlertLevel.WARNING,
                    "CPU Warning",
                    f"CPU usage is elevated: {performance.cpu_percent:.1f}%",
                    cpu_ratio,
                    self.thresholds['cpu_warning']
                )

            # Response time alerts
            if cloud.average_response_time_ms >= self.thresholds['response_time_critical']:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "Response Time Critical",
                    f"Average response time is very high: {cloud.average_response_time_ms:.0f}ms",
                    cloud.average_response_time_ms,
                    self.thresholds['response_time_critical']
                )
            elif cloud.average_response_time_ms >= self.thresholds['response_time_warning']:
                self._create_alert(
                    AlertLevel.WARNING,
                    "Response Time Warning",
                    f"Response time is elevated: {cloud.average_response_time_ms:.0f}ms",
                    cloud.average_response_time_ms,
                    self.thresholds['response_time_warning']
                )

            # Error rate alerts
            total_requests = cloud.total_requests
            if total_requests > 0:
                error_rate = cloud.failed_requests / total_requests
                if error_rate >= self.thresholds['error_rate_critical']:
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        "Error Rate Critical",
                        f"Error rate is critical: {error_rate:.1%}",
                        error_rate,
                        self.thresholds['error_rate_critical']
                    )
                elif error_rate >= self.thresholds['error_rate_warning']:
                    self._create_alert(
                        AlertLevel.WARNING,
                        "Error Rate Warning",
                        f"Error rate is elevated: {error_rate:.1%}",
                        error_rate,
                        self.thresholds['error_rate_warning']
                    )

            # Qdrant connection alerts
            if not cloud.qdrant_connection_status:
                recent_tests = list(self.cloud_metrics['qdrant_connection_tests'])[-10:]  # Last 10 tests
                failure_rate = 1.0 - (sum(recent_tests) / len(recent_tests)) if recent_tests else 1.0

                if failure_rate >= 0.8:  # 80% failure rate
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        "Qdrant Connection Critical",
                        f"Qdrant connection failures: {failure_rate:.1%}",
                        failure_rate,
                        0.2
                    )

        except Exception as e:
            logger.error(f"[ERROR] Alert checking failed: {e}")

    def _create_alert(self, level: AlertLevel, title: str, message: str, value: float, threshold: float):
        """Create a new performance alert"""
        try:
            alert_id = f"{level.value}_{title.lower().replace(' ', '_')}_{int(time.time())}"

            # Check if similar alert already exists and is not resolved
            existing_alert = None
            for existing in self.alerts.values():
                if (not existing.resolved and
                    existing.level == level and
                    existing.title == title and
                    (datetime.now() - existing.timestamp).seconds < 300):  # Within 5 minutes
                    existing_alert = existing
                    break

            if existing_alert:
                # Update existing alert
                existing_alert.metrics_value = value
                existing_alert.timestamp = datetime.now()
            else:
                # Create new alert
                alert = Alert(
                    id=alert_id,
                    level=level,
                    title=title,
                    message=message,
                    timestamp=datetime.now(),
                    metrics_value=value,
                    threshold=threshold
                )

                self.alerts[alert_id] = alert
                logger.warning(f"[ALERT] {level.value.upper()}: {title} - {message}")

                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"[ERROR] Alert callback failed: {e}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to create alert: {e}")

    def record_request(self, success: bool, response_time_ms: float):
        """Record a request for performance tracking"""
        self.cloud_metrics['total_requests'] += 1

        if success:
            self.cloud_metrics['successful_requests'] += 1
        else:
            self.cloud_metrics['failed_requests'] += 1

        self.cloud_metrics['response_times'].append(response_time_ms)

    def update_component_status(self, component_name: str, status: Dict[str, Any]):
        """Update status of a specific component"""
        self.cloud_metrics['component_status'][component_name] = {
            'status': status,
            'last_updated': datetime.now().isoformat()
        }

    def _update_optimization_suggestions(self):
        """Update performance optimization suggestions"""
        try:
            suggestions = []

            if not self.performance_history:
                return

            latest_metrics = self.performance_history[-1]
            latest_cloud = self.cloud_history[-1] if self.cloud_history else None

            # Memory optimization suggestions
            memory_ratio = latest_metrics.memory_percent / 100.0
            if memory_ratio > 0.8:
                suggestions.append({
                    'priority': 'high',
                    'category': 'memory',
                    'suggestion': 'Memory usage is high. Consider increasing cleanup frequency or reducing batch sizes.',
                    'current_value': f"{memory_ratio:.1%}",
                    'target_value': '< 80%'
                })
            elif memory_ratio > 0.6:
                suggestions.append({
                    'priority': 'medium',
                    'category': 'memory',
                    'suggestion': 'Monitor memory usage closely. Enable more aggressive garbage collection.',
                    'current_value': f"{memory_ratio:.1%}",
                    'target_value': '< 60%'
                })

            # CPU optimization suggestions
            cpu_ratio = latest_metrics.cpu_percent / self.cpu_limit_percent
            if cpu_ratio > 0.7:
                suggestions.append({
                    'priority': 'medium',
                    'category': 'cpu',
                    'suggestion': 'CPU usage is elevated. Consider optimizing algorithm efficiency or reducing concurrent tasks.',
                    'current_value': f"{cpu_ratio:.1%}",
                    'target_value': '< 70%'
                })

            # Response time suggestions
            if latest_cloud and latest_cloud.average_response_time_ms > 3000:
                suggestions.append({
                    'priority': 'high',
                    'category': 'performance',
                    'suggestion': 'Response times are slow. Check for bottlenecks in processing pipelines.',
                    'current_value': f"{latest_cloud.average_response_time_ms:.0f}ms",
                    'target_value': '< 3000ms'
                })

            # Error rate suggestions
            if latest_cloud:
                total_requests = latest_cloud.total_requests
                if total_requests > 0:
                    error_rate = latest_cloud.failed_requests / total_requests
                    if error_rate > 0.05:
                        suggestions.append({
                            'priority': 'high',
                            'category': 'reliability',
                            'suggestion': 'Error rate is elevated. Review error logs and implement better error handling.',
                            'current_value': f"{error_rate:.1%}",
                            'target_value': '< 5%'
                        })

            if not suggestions:
                suggestions.append({
                    'priority': 'info',
                    'category': 'general',
                    'suggestion': 'System performance is optimal. Continue current configuration.',
                    'current_value': 'Good',
                    'target_value': 'Maintain current performance'
                })

            self.optimization_suggestions = suggestions

        except Exception as e:
            logger.error(f"[ERROR] Failed to update optimization suggestions: {e}")

    def get_performance_status(self) -> PerformanceStatus:
        """Get overall system performance status"""
        try:
            if not self.performance_history:
                return PerformanceStatus.GOOD

            latest_metrics = self.performance_history[-1]
            latest_cloud = self.cloud_history[-1] if self.cloud_history else None

            # Calculate overall score
            memory_ratio = latest_metrics.memory_percent / 100.0
            cpu_ratio = latest_metrics.cpu_percent / self.cpu_limit_percent

            # Start with base score of 100
            score = 100.0

            # Memory impact (40% weight)
            if memory_ratio > 0.95:
                score -= 40
            elif memory_ratio > 0.85:
                score -= 30
            elif memory_ratio > 0.75:
                score -= 20
            elif memory_ratio > 0.6:
                score -= 10

            # CPU impact (30% weight)
            if cpu_ratio > 0.9:
                score -= 30
            elif cpu_ratio > 0.8:
                score -= 20
            elif cpu_ratio > 0.7:
                score -= 10

            # Response time impact (20% weight)
            if latest_cloud:
                response_ratio = latest_cloud.average_response_time_ms / 5000  # 5 seconds as baseline
                if response_ratio > 2:
                    score -= 20
                elif response_ratio > 1:
                    score -= 10

            # Error rate impact (10% weight)
            if latest_cloud and latest_cloud.total_requests > 0:
                error_rate = latest_cloud.failed_requests / latest_cloud.total_requests
                if error_rate > 0.1:
                    score -= 10
                elif error_rate > 0.05:
                    score -= 5

            # Determine status based on score
            if score >= 85:
                return PerformanceStatus.EXCELLENT
            elif score >= 70:
                return PerformanceStatus.GOOD
            elif score >= 50:
                return PerformanceStatus.FAIR
            elif score >= 30:
                return PerformanceStatus.POOR
            else:
                return PerformanceStatus.CRITICAL

        except Exception:
            return PerformanceStatus.UNKNOWN

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        try:
            current_status = self.get_performance_status()

            # Get latest metrics
            latest_performance = self.performance_history[-1] if self.performance_history else None
            latest_cloud = self.cloud_history[-1] if self.cloud_history else None

            # Calculate trends (comparing last 10 to previous 10)
            performance_trend = self._calculate_performance_trend()

            # Active alerts
            active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]

            dashboard = {
                'status': current_status.value,
                'monitoring_active': self.running,
                'last_updated': datetime.now().isoformat(),
                'current_metrics': {
                    'performance': asdict(latest_performance) if latest_performance else None,
                    'cloud': asdict(latest_cloud) if latest_cloud else None
                },
                'trends': performance_trend,
                'alerts': {
                    'total_count': len(active_alerts),
                    'by_level': {
                        level.value: len([a for a in active_alerts if a.level == level])
                        for level in AlertLevel
                    },
                    'recent': [asdict(alert) for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:10]]
                },
                'optimization_suggestions': self.optimization_suggestions,
                'resource_utilization': {
                    'memory': {
                        'current_mb': latest_performance.memory_rss_mb if latest_performance else 0,
                        'limit_mb': self.memory_limit_mb,
                        'percent': latest_performance.memory_percent if latest_performance else 0
                    },
                    'cpu': {
                        'current_percent': latest_performance.cpu_percent if latest_performance else 0,
                        'limit_percent': self.cpu_limit_percent
                    }
                },
                'cloud_health': {
                    'qdrant_connection': latest_cloud.qdrant_connection_status if latest_cloud else False,
                    'active_components': latest_cloud.active_components if latest_cloud else 0,
                    'memory_efficiency': latest_cloud.memory_efficiency_score if latest_cloud else 0,
                    'storage_efficiency': latest_cloud.storage_efficiency_score if latest_cloud else 0
                }
            }

            return dashboard

        except Exception as e:
            logger.error(f"[ERROR] Failed to generate performance dashboard: {e}")
            return {'error': str(e), 'status': 'unknown'}

    def _calculate_performance_trend(self) -> Dict[str, str]:
        """Calculate performance trends"""
        try:
            if len(self.performance_history) < 20:
                return {'overall': 'insufficient_data'}

            # Compare last 10 to previous 10
            recent = list(self.performance_history)[-10:]
            previous = list(self.performance_history)[-20:-10]

            recent_avg_memory = np.mean([m.memory_percent for m in recent])
            previous_avg_memory = np.mean([m.memory_percent for m in previous])

            recent_avg_cpu = np.mean([m.cpu_percent for m in recent])
            previous_avg_cpu = np.mean([m.cpu_percent for m in previous])

            trends = {}

            # Memory trend
            if recent_avg_memory > previous_avg_memory * 1.1:
                trends['memory'] = 'increasing'
            elif recent_avg_memory < previous_avg_memory * 0.9:
                trends['memory'] = 'decreasing'
            else:
                trends['memory'] = 'stable'

            # CPU trend
            if recent_avg_cpu > previous_avg_cpu * 1.1:
                trends['cpu'] = 'increasing'
            elif recent_avg_cpu < previous_avg_cpu * 0.9:
                trends['cpu'] = 'decreasing'
            else:
                trends['cpu'] = 'stable'

            # Overall trend
            if (trends['memory'] == 'increasing' and trends['cpu'] == 'increasing'):
                trends['overall'] = 'degrading'
            elif (trends['memory'] == 'decreasing' and trends['cpu'] == 'decreasing'):
                trends['overall'] = 'improving'
            else:
                trends['overall'] = 'stable'

            return trends

        except Exception:
            return {'overall': 'error'}

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for new alerts"""
        self.alert_callbacks.append(callback)

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            logger.info(f"[ALERT] Resolved: {alert_id}")

    def get_metrics_export(self, format_type: str = 'json') -> str:
        """Export performance metrics in specified format"""
        try:
            dashboard = self.get_performance_dashboard()

            if format_type.lower() == 'json':
                return json.dumps(dashboard, indent=2, default=str)
            elif format_type.lower() == 'csv':
                # Simple CSV export of key metrics
                if self.performance_history:
                    metrics_data = []
                    for metrics in self.performance_history:
                        metrics_data.append({
                            'timestamp': metrics.timestamp.isoformat(),
                            'cpu_percent': metrics.cpu_percent,
                            'memory_rss_mb': metrics.memory_rss_mb,
                            'memory_percent': metrics.memory_percent
                        })
                    return pd.DataFrame(metrics_data).to_csv(index=False)
                else:
                    return "No metrics data available"
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to export metrics: {e}")
            return f"Error: {str(e)}"

# Global performance monitor instance
cloud_performance_monitor = CloudPerformanceMonitor()

def get_cloud_performance_monitor() -> CloudPerformanceMonitor:
    """Get the global cloud performance monitor instance"""
    return cloud_performance_monitor

if __name__ == "__main__":
    # Test the performance monitor
    monitor = CloudPerformanceMonitor(monitoring_interval=5)  # 5 second intervals for testing

    try:
        monitor.start_monitoring()
        print("Performance monitoring started...")

        # Simulate some requests
        for i in range(10):
            monitor.record_request(i % 9 != 0, 100 + i * 50)  # 1 failure, varying response times
            time.sleep(1)

        # Get dashboard
        dashboard = monitor.get_performance_dashboard()
        print("\nPerformance Dashboard:")
        print(json.dumps(dashboard, indent=2, default=str))

        # Export metrics
        print("\nMetrics Export (JSON):")
        print(monitor.get_metrics_export('json')[:500] + "...")  # First 500 chars

    finally:
        monitor.stop_monitoring()
        print("\nPerformance monitoring stopped.")