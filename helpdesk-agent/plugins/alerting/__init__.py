"""Alerting plugin — volume anomaly detection and topic clustering."""

import datetime as dt
import logging

from config import OPSGENIE_API_KEY
from core.pipeline import get_plugin_config
from db import get_db_conn, get_last_run_date, set_last_run_date
from plugins._protocol import BasePlugin
from plugins.alerting.anomaly import CategoryAnomalyDetector, ControlChartDetector
from plugins.alerting.anomaly_alert import fire_anomaly_alert, fire_content_anomaly_alerts
from plugins.alerting.cluster_alert import process_cluster_alerts
from plugins.alerting.clustering import TopicClusterer, run_all_windows
from plugins.alerting.isolation_forest import (
    IsolationForestDetector,
    store_control_chart_score,
)

log = logging.getLogger(__name__)

TOPIC_CLUSTER_JOB = "topic_cluster"

try:
    from ingest.oauth2lo import get_cloud_base_url as _get_cloud_base_url
    _cloud_id = _get_cloud_base_url("jsm").rstrip("/").split("/")[-1]
except Exception:
    _cloud_id = "default"

__all__ = ["plugin", "CategoryAnomalyDetector", "ControlChartDetector", "TopicClusterer"]


class AlertingPlugin(BasePlugin):
    """Control-chart anomaly detection and daily topic clustering."""

    name = "alerting"

    def on_schedule(self) -> None:
        cfg = get_plugin_config("alerting")
        sigma = float(cfg.get("anomaly_sigma", 2.0))
        window_days = int(cfg.get("cluster_window_days", 3))
        cluster_k = int(cfg.get("cluster_k", 10))
        velocity_threshold = float(cfg.get("cluster_velocity_threshold", 0.3))
        min_cluster_size = int(cfg.get("hdbscan_min_cluster_size", 3))
        clustering_approach = str(cfg.get("clustering_approach", "auto"))

        detector = ControlChartDetector(sigma=sigma)
        cat_detector = CategoryAnomalyDetector(sigma=sigma)
        if_detector = IsolationForestDetector(
            contamination=float(cfg.get("if_contamination", 0.05)),
            fpr_threshold=float(cfg.get("if_fpr_threshold", 0.05)),
            fpr_window_days=int(cfg.get("if_window_days", 7)),
        )
        clusterer = TopicClusterer(velocity_threshold=velocity_threshold)
        today = dt.date.today()

        with get_db_conn() as conn:
            baseline_count = detector.build_baseline(conn)
            log.info("[alerting] rebuilt baseline for %d segments", baseline_count)

            cat_baseline_count = cat_detector.build_category_baselines(conn)
            log.info(
                "[alerting] rebuilt category baselines for %d (category, segment) pairs",
                cat_baseline_count,
            )

            cc_result = detector.score(conn, window_minutes=60)
            # Always persist control-chart telemetry for live FPR measurement,
            # even when IF is currently active.
            store_control_chart_score(conn, cc_result, window_minutes=60)
            if if_detector.is_activated(conn):
                if if_detector.fit(conn):
                    result = if_detector.score(conn, window_minutes=60)
                    log.info(
                        "[alerting] isolation forest active — segment=%s count=%d z=%.2f anomaly=%s",
                        result.segment,
                        result.ticket_count,
                        result.zscore,
                        result.is_anomaly,
                    )
                else:
                    result = cc_result
                    log.warning(
                        "[alerting] isolation forest activated but unfitted; "
                        "falling back to control chart for this cycle"
                    )
            else:
                result = cc_result
                log.info(
                    "[alerting] control chart active — segment=%s count=%d z=%.2f anomaly=%s",
                    result.segment,
                    result.ticket_count,
                    result.zscore,
                    result.is_anomaly,
                )
            detector.store_score(conn, result, window_minutes=60)

            if (
                result.is_anomaly
                and cfg.get("anomaly_alerts_enabled")
                and OPSGENIE_API_KEY
            ):
                if fire_anomaly_alert(result):
                    log.info(
                        "[alerting] OpsGenie anomaly alert sent segment=%s",
                        result.segment,
                    )

            _job_key = f"{TOPIC_CLUSTER_JOB}:{_cloud_id}"
            last_cluster_run = get_last_run_date(conn, _job_key)
            if last_cluster_run == today:
                log.debug("[alerting] topic_cluster already ran today — skipping")
                return

            log.info(
                "[alerting] running all 6 clustering windows (k=%d, approach=%s)",
                cluster_k,
                clustering_approach,
            )
            window_results = run_all_windows(
                conn,
                k=cluster_k,
                min_cluster_size=min_cluster_size,
                clustering_approach=clustering_approach,
            )
            set_last_run_date(conn, _job_key, today)
            total_written = sum(window_results.values())
            log.info(
                "[alerting] all windows complete — %d total clusters across %d windows: %s",
                total_written,
                len(window_results),
                ", ".join(f"{k}={v}" for k, v in window_results.items()),
            )

            # Fire cluster alerts using the short-window (daily) deltas, which
            # most closely match the old incident-window behaviour.
            if window_results.get("daily", 0) > 0:
                from db import get_db_conn as _get_db_conn
                daily_clusterer = TopicClusterer(velocity_threshold=velocity_threshold)
                prev_daily = daily_clusterer.load_previous_clusters(
                    conn, today.isoformat(), run_type="daily"
                )
                daily_current = daily_clusterer.cluster(
                    conn, window_days=1, k=cluster_k,
                    min_cluster_size=min_cluster_size,
                    clustering_approach=clustering_approach,
                )
                if daily_current:
                    daily_deltas = daily_clusterer.compute_delta(daily_current, prev_daily)
                    alerted = process_cluster_alerts(
                        conn,
                        daily_deltas,
                        velocity_threshold=velocity_threshold,
                        run_date=today.isoformat(),
                        cfg=cfg,
                    )
                    log.info("[alerting] cluster alerts — %d fired (daily window)", alerted)

            content_anomalies = cat_detector.score_categories(
                conn, window_days=window_days
            )
            topic_spikes = cat_detector.detect_topic_spikes(
                conn, window_days=window_days
            )
            novel_cats = cat_detector.detect_novel_categories(
                conn, window_days=window_days
            )
            all_content_anomalies = content_anomalies + topic_spikes + novel_cats
            log.info(
                "[alerting] content anomaly scan — %d category spikes, %d topic spikes, "
                "%d novel categories",
                sum(1 for r in content_anomalies if r.is_spike),
                len(topic_spikes),
                len(novel_cats),
            )

            if (
                all_content_anomalies
                and cfg.get("anomaly_alerts_enabled")
                and OPSGENIE_API_KEY
            ):
                content_sent = fire_content_anomaly_alerts(all_content_anomalies)
                log.info(
                    "[alerting] fired %d content anomaly OpsGenie alerts",
                    content_sent,
                )


plugin = AlertingPlugin()
