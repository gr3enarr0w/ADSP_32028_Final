"""Tests for plugins/alerting/clustering.py — ANTSE-318 / ANTSE-320."""

import datetime as dt
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from db import get_db_conn, init_db


@pytest.fixture(autouse=True)
def use_memory_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("db.DB_PATH", db_path):
        init_db()
        yield


TOPIC_TEMPLATES = [
    ("access login permission for jira project", "access / login issues"),
    ("bot service account oauth token alias", "bot / service accounts"),
    ("confluence wiki space pages access", "confluence / wiki access"),
    ("automation export scripting rule trigger", "automation and export"),
    ("custom field workflow issue type template", "JIRA configuration"),
    ("group rover membership contributing team", "group / access provisioning"),
    ("project scheme category workflow admin", "project admin / scheme changes"),
]


def _seed_cluster_tickets(conn, count_per_topic: int = 4, days_ago: int = 1):
    base = datetime.now(timezone.utc) - timedelta(days=days_ago)
    idx = 0
    for text, _label in TOPIC_TEMPLATES:
        for i in range(count_per_topic):
            idx += 1
            ts = (base + timedelta(hours=idx)).isoformat()
            conn.execute(
                """
                INSERT INTO tickets (ticket_key, summary, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (f"CL-{idx}", f"Ticket {idx}", text, ts),
            )


class TestTopicClusterer:
    def test_cluster_returns_labeled_results(self):
        from plugins.alerting.clustering import TopicClusterer

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=5)
            clusterer = TopicClusterer()
            results = clusterer.cluster(conn, window_days=3, k=7)

        assert len(results) > 0
        assert all(r.label for r in results)
        assert all(r.size > 0 for r in results)
        assert all(r.ticket_keys for r in results)
        assert all(r.approach for r in results)

    def test_compute_delta_new_cluster(self):
        from plugins.alerting.clustering import ClusterResult, TopicClusterer

        clusterer = TopicClusterer(velocity_threshold=0.3)
        current = [
            ClusterResult(cluster_id=0, ticket_keys=["A-1", "A-2"], label="access", size=2),
        ]
        previous = [
            ClusterResult(cluster_id=0, ticket_keys=["B-1", "B-2"], label="other", size=2),
        ]
        deltas = clusterer.compute_delta(current, previous)
        assert len(deltas) == 1
        assert deltas[0].is_new is True
        assert deltas[0].growth_rate == 1.0

    def test_compute_delta_growing_cluster(self):
        from plugins.alerting.clustering import ClusterResult, TopicClusterer

        clusterer = TopicClusterer(velocity_threshold=0.3)
        current = [
            ClusterResult(
                cluster_id=1,
                ticket_keys=["T-1", "T-2", "T-3", "T-4"],
                label="access",
                size=4,
            ),
        ]
        previous = [
            ClusterResult(cluster_id=1, ticket_keys=["T-1", "T-2"], label="access", size=2),
        ]
        deltas = clusterer.compute_delta(current, previous)
        assert deltas[0].is_new is False
        assert deltas[0].growth_rate == 1.0

    def test_write_and_load_clusters(self):
        from plugins.alerting.clustering import DeltaResult, TopicClusterer

        clusterer = TopicClusterer()
        run_date = "2026-06-01"
        deltas = [
            DeltaResult(
                cluster_id=0,
                label="access / login issues",
                size=5,
                is_new=True,
                growth_rate=1.0,
                ticket_keys=["CL-1", "CL-2"],
            )
        ]
        with get_db_conn() as conn:
            clusterer.write_clusters(conn, run_date, deltas)
            loaded = clusterer.load_previous_clusters(conn, "2026-06-02")

        assert len(loaded) == 1
        assert loaded[0].ticket_keys == ["CL-1", "CL-2"]

    def test_load_previous_clusters_run_type_isolation(self):
        from plugins.alerting.clustering import DeltaResult, TopicClusterer

        clusterer = TopicClusterer()
        run_date = "2026-06-01"
        incident_delta = DeltaResult(
            cluster_id=1, label="incident cluster", size=3,
            is_new=True, growth_rate=1.0, ticket_keys=["I-1", "I-2"],
        )
        trend_delta = DeltaResult(
            cluster_id=1, label="trend cluster", size=10,
            is_new=False, growth_rate=0.5, ticket_keys=["T-1", "T-2", "T-3"],
        )
        with get_db_conn() as conn:
            clusterer.write_clusters(conn, run_date, [incident_delta], run_type="incident")
            clusterer.write_clusters(conn, run_date, [trend_delta], run_type="trend")
            loaded_incident = clusterer.load_previous_clusters(conn, "2026-06-02", run_type="incident")
            loaded_trend = clusterer.load_previous_clusters(conn, "2026-06-02", run_type="trend")

        assert len(loaded_incident) == 1
        assert loaded_incident[0].label == "incident cluster"
        assert loaded_incident[0].ticket_keys == ["I-1", "I-2"]

        assert len(loaded_trend) == 1
        assert loaded_trend[0].label == "trend cluster"
        assert loaded_trend[0].ticket_keys == ["T-1", "T-2", "T-3"]

    def test_write_clusters_invalid_run_type_raises(self):
        from plugins.alerting.clustering import DeltaResult, TopicClusterer

        clusterer = TopicClusterer()
        with get_db_conn() as conn:
            with pytest.raises(ValueError, match="Invalid run_type"):
                clusterer.write_clusters(
                    conn, "2026-06-05",
                    [DeltaResult(1, "label", 1, True, 1.0, ["T-1"])],
                    run_type="Incident",
                )

    def test_write_clusters_json_ticket_keys(self):
        from plugins.alerting.clustering import DeltaResult, TopicClusterer

        clusterer = TopicClusterer()
        with get_db_conn() as conn:
            clusterer.write_clusters(
                conn,
                "2026-06-03",
                [
                    DeltaResult(
                        cluster_id=2,
                        label="bot accounts",
                        size=3,
                        is_new=False,
                        growth_rate=0.5,
                        ticket_keys=["X-1", "X-2", "X-3"],
                    )
                ],
            )
            row = conn.execute(
                "SELECT ticket_keys, is_new FROM ticket_clusters WHERE run_date = ? AND cluster_id = 2",
                ("2026-06-03",),
            ).fetchone()
        assert json.loads(row["ticket_keys"]) == ["X-1", "X-2", "X-3"]
        assert row["is_new"] == 0


class TestDailyGuard:
    def test_on_schedule_skips_second_run_same_day(self):
        from plugins.alerting import TOPIC_CLUSTER_JOB, plugin

        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO job_state (job_name, last_run_date) VALUES (?, ?)
                """,
                (TOPIC_CLUSTER_JOB, dt.date.today().isoformat()),
            )
            before = conn.execute("SELECT COUNT(*) FROM ticket_clusters").fetchone()[0]

        with patch("plugins.alerting.set_last_run_date") as mock_set:
            with patch("plugins.alerting.TopicClusterer.cluster") as mock_cluster:
                plugin.on_schedule()
                mock_cluster.assert_not_called()
                mock_set.assert_not_called()

        with get_db_conn() as conn:
            after = conn.execute("SELECT COUNT(*) FROM ticket_clusters").fetchone()[0]
        assert after == before


class TestHdbscanHelpers:
    def test_fit_hdbscan_returns_labels(self):
        from plugins.alerting.clustering import _fit_hdbscan, _vectorize

        texts = [f"access login jira permission issue number {i}" for i in range(12)]
        texts += [f"confluence wiki space page access number {i}" for i in range(12)]
        _, matrix = _vectorize(texts)
        model, labels = _fit_hdbscan(matrix, min_cluster_size=3)
        assert model is not None
        assert labels is not None
        assert len(labels) == matrix.shape[0]
        assert len({int(label) for label in labels if label != -1}) >= 1

    def test_ensemble_labels_fallback_for_noise(self):
        from plugins.alerting.clustering import _ensemble_labels

        hdbscan = np.array([0, 0, -1, 1, -1])
        kmeans = np.array([5, 6, 7, 8, 9])
        combined = _ensemble_labels(hdbscan, kmeans)
        assert combined.tolist() == [0, 0, 7, 1, 9]

    def test_benchmark_approaches_returns_all_three(self):
        from plugins.alerting.clustering import _vectorize, benchmark_approaches

        texts = [f"access login jira permission issue number {i}" for i in range(12)]
        texts += [f"confluence wiki space page access number {i}" for i in range(12)]
        texts += [f"bot service account oauth token alias number {i}" for i in range(12)]
        tickets = [{"ticket_key": f"T-{i}", "text": t, "summary": t} for i, t in enumerate(texts)]
        vectorizer, matrix = _vectorize(texts)
        feature_names = vectorizer.get_feature_names_out()
        benchmarks, labels = benchmark_approaches(
            tickets, matrix, feature_names, k=5, min_cluster_size=3
        )
        assert set(benchmarks) == {"kmeans", "hdbscan", "ensemble"}
        assert set(labels) == {"kmeans", "hdbscan", "ensemble"}
        for approach in benchmarks:
            assert benchmarks[approach].approach == approach

    def test_pick_best_approach_uses_composite_score(self):
        from plugins.alerting.clustering import ApproachMetrics, pick_best_approach

        benchmarks = {
            "kmeans": ApproachMetrics(
                approach="kmeans",
                silhouette=0.05,
                noise_fraction=0.0,
                cluster_count=5,
                coherent_count=2,
                incoherent_count=3,
                coherence_ratio=0.4,
                composite_score=0.26,
                valid=True,
            ),
            "hdbscan": ApproachMetrics(
                approach="hdbscan",
                silhouette=0.08,
                noise_fraction=0.15,
                cluster_count=4,
                coherent_count=3,
                incoherent_count=1,
                coherence_ratio=0.75,
                composite_score=0.482,
                valid=True,
            ),
            "ensemble": ApproachMetrics(
                approach="ensemble",
                silhouette=0.06,
                noise_fraction=0.05,
                cluster_count=6,
                coherent_count=4,
                incoherent_count=2,
                coherence_ratio=0.667,
                composite_score=0.424,
                valid=True,
            ),
        }
        assert pick_best_approach(benchmarks) == "hdbscan"

    def test_pick_best_approach_tiebreak_prefers_ensemble(self):
        from plugins.alerting.clustering import ApproachMetrics, pick_best_approach

        score = 0.5
        benchmarks = {
            "kmeans": ApproachMetrics("kmeans", 0.1, 0.0, 3, 1, 2, 0.33, score, True),
            "hdbscan": ApproachMetrics("hdbscan", 0.1, 0.1, 3, 1, 2, 0.33, score, True),
            "ensemble": ApproachMetrics("ensemble", 0.1, 0.05, 3, 1, 2, 0.33, score, True),
        }
        assert pick_best_approach(benchmarks) == "ensemble"

    def test_select_approach_forced_modes(self):
        from plugins.alerting.clustering import _select_approach

        assert _select_approach("kmeans") == "kmeans"
        assert _select_approach("hdbscan") == "hdbscan"
        assert _select_approach("ensemble") == "ensemble"

    def test_labels_valid_requires_two_distinct_clusters(self):
        from plugins.alerting.clustering import _labels_valid

        assert _labels_valid(np.array([0, 0, 1, 1])) is True
        assert _labels_valid(np.array([0, 0, 0, 0])) is False
        assert _labels_valid(np.array([-1, -1, -1])) is False

    def test_cluster_coherence_intra_known_matrix(self):
        from plugins.alerting.clustering import _cluster_coherence_intra

        # Three unit vectors: pair (0,1) cos=1, (0,2) cos=0, (1,2) cos=0 → mean 1/3
        matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        score = _cluster_coherence_intra(matrix, np.array([0, 1, 2]))
        assert score == pytest.approx(1.0 / 3.0)

    def test_cluster_coherence_intra_singleton(self):
        from plugins.alerting.clustering import _cluster_coherence_intra

        matrix = np.array([[1.0, 0.0]])
        assert _cluster_coherence_intra(matrix, np.array([0])) == 1.0

    def test_evaluate_labels_intra_similarity_mode(self):
        from plugins.alerting.clustering import _evaluate_labels, _vectorize

        texts = [f"access login jira permission issue number {i}" for i in range(8)]
        texts += [f"confluence wiki space page access number {i}" for i in range(8)]
        tickets = [{"ticket_key": f"T-{i}", "text": t, "summary": t} for i, t in enumerate(texts)]
        vectorizer, matrix = _vectorize(texts)
        feature_names = vectorizer.get_feature_names_out()
        labels = np.array([0] * 8 + [1] * 8)
        metrics = _evaluate_labels(
            labels,
            tickets,
            matrix,
            feature_names,
            approach="kmeans",
            coherence_mode="intra_similarity",
        )
        assert metrics.valid is True
        assert 0.0 <= metrics.coherence_ratio <= 1.0
        assert metrics.composite_score > float("-inf")
        assert metrics.coherent_count + metrics.incoherent_count == metrics.cluster_count

    def test_trend_composite_score_weights(self):
        from plugins.alerting.clustering import _trend_composite_score

        assert _trend_composite_score(0.4, 0.2) == pytest.approx(0.3)


class TestApproachSelection:
    def test_cluster_forced_kmeans(self):
        from plugins.alerting.clustering import TopicClusterer

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=5)
            clusterer = TopicClusterer()
            results = clusterer.cluster(conn, window_days=3, k=7, clustering_approach="kmeans")

        assert len(results) > 0
        assert all(r.approach == "kmeans" for r in results)

    def test_cluster_forced_ensemble(self):
        from plugins.alerting.clustering import TopicClusterer

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=5)
            clusterer = TopicClusterer()
            results = clusterer.cluster(conn, window_days=3, k=7, clustering_approach="ensemble")

        assert len(results) > 0
        assert all(r.approach == "ensemble" for r in results)

    def test_cluster_auto_uses_benchmark_winner(self):
        from plugins.alerting.clustering import ApproachMetrics, TopicClusterer

        fake_benchmarks = {
            "kmeans": ApproachMetrics("kmeans", 0.01, 0.0, 3, 1, 2, 0.33, 0.1, True),
            "hdbscan": ApproachMetrics("hdbscan", 0.02, 0.1, 3, 1, 2, 0.33, 0.15, True),
            "ensemble": ApproachMetrics("ensemble", 0.03, 0.05, 4, 3, 1, 0.75, 0.5, True),
        }
        fake_labels = {
            "kmeans": np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
            "hdbscan": np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
            "ensemble": np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        }

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=5)
            clusterer = TopicClusterer()
            with patch(
                "plugins.alerting.clustering.benchmark_approaches",
                return_value=(fake_benchmarks, fake_labels),
            ):
                with patch(
                    "plugins.alerting.clustering.pick_best_approach",
                    return_value="ensemble",
                ):
                    results = clusterer.cluster(conn, window_days=3, k=7, clustering_approach="auto")

        assert len(results) > 0
        assert all(r.approach == "ensemble" for r in results)

    def test_cluster_forced_hdbscan_not_blocked_by_k(self):
        from plugins.alerting.clustering import ApproachMetrics, TopicClusterer

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=5)
            clusterer = TopicClusterer()
            fake_benchmarks = {
                "hdbscan": ApproachMetrics(
                    "hdbscan", 0.2, 0.1, 3, 2, 1, 0.67, 0.482, True
                )
            }
            fake_labels = {"hdbscan": np.array([0] * 12 + [1] * 12 + [2] * 11)}
            with patch(
                "plugins.alerting.clustering.benchmark_approaches",
                return_value=(fake_benchmarks, fake_labels),
            ) as mock_benchmark:
                results = clusterer.cluster(
                    conn,
                    window_days=3,
                    k=100,
                    min_cluster_size=3,
                    clustering_approach="hdbscan",
                )

        mock_benchmark.assert_called_once()
        assert len(results) > 0
        assert all(r.approach == "hdbscan" for r in results)

    def test_cluster_window_type_trend_uses_intra_similarity(self):
        from plugins.alerting.clustering import ApproachMetrics, TopicClusterer

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=5)
            clusterer = TopicClusterer()
            with patch(
                "plugins.alerting.clustering.benchmark_approaches"
            ) as mock_benchmark:
                mock_benchmark.return_value = (
                    {
                        "hdbscan": ApproachMetrics(
                            "hdbscan", 0.35, 0.1, 4, 3, 1, 0.55, 0.45, True
                        )
                    },
                    {"hdbscan": np.array([0] * 12 + [1] * 12 + [2] * 11)},
                )
                clusterer.cluster(conn, window_days=30, k=7, window_type="trend")

        mock_benchmark.assert_called_once()
        _, kwargs = mock_benchmark.call_args
        assert kwargs["coherence_mode"] == "intra_similarity"

    def test_write_clusters_run_type(self):
        from plugins.alerting.clustering import DeltaResult, TopicClusterer

        clusterer = TopicClusterer()
        with get_db_conn() as conn:
            clusterer.write_clusters(
                conn,
                "2026-06-04",
                [
                    DeltaResult(
                        cluster_id=1,
                        label="mixed",
                        size=2,
                        is_new=True,
                        growth_rate=1.0,
                        ticket_keys=["T-1"],
                    )
                ],
                run_type="trend",
            )
            row = conn.execute(
                """
                SELECT run_type FROM ticket_clusters
                WHERE run_date = ? AND cluster_id = 1
                """,
                ("2026-06-04",),
            ).fetchone()
        assert row["run_type"] == "trend"


class TestClusteringBenchmark:
    def test_benchmark_run_on_seeded_corpus(self):
        from analysis.clustering_benchmark import run

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=6)

        results = run(window_days=3, k=7, min_cluster_size=3, anchor_mode="now")

        assert results["ticket_count"] >= 35
        assert set(results["approaches"]) == {"kmeans", "hdbscan", "ensemble"}
        for approach in ("kmeans", "hdbscan", "ensemble"):
            row = results["approaches"][approach]
            assert row["cluster_count"] > 0
            assert row["valid"] is True
            assert row["composite_score"] is not None
        assert results["selected_approach"] in ("kmeans", "hdbscan", "ensemble")

    def test_benchmark_run_trend_window_type(self):
        from analysis.clustering_benchmark import run

        with get_db_conn() as conn:
            _seed_cluster_tickets(conn, count_per_topic=6)

        results = run(
            window_days=30,
            k=7,
            min_cluster_size=3,
            anchor_mode="now",
            window_type="trend",
        )

        assert results["window_type"] == "trend"
        assert results["coherence_mode"] == "intra_similarity"
        assert results["acceptance_bar"] == 0.25


class TestConfig:
    def test_cluster_defaults_from_pipeline(self):
        from pathlib import Path
        from core.pipeline import get_plugin_config, load_pipeline_config

        load_pipeline_config(Path(__file__).resolve().parents[3] / "pipeline.yaml")
        cfg = get_plugin_config("alerting")
        assert cfg["cluster_window_days"] == 3
        assert cfg["trend_window_days"] == 30
        assert cfg["trend_alerts_enabled"] is True
        assert cfg["cluster_k"] == 10
        assert cfg["cluster_velocity_threshold"] == 0.3
        assert cfg["cluster_alerts_enabled"] is True
        assert cfg["opsgenie_enabled"] is True
        assert cfg["hdbscan_min_cluster_size"] == 3
        assert cfg["clustering_approach"] == "auto"


class TestSqliteMigrations:
    def test_init_db_rebuilds_ticket_clusters_primary_key(self, tmp_path):
        db_path = str(tmp_path / "migrated.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE ticket_clusters (
                cluster_id INTEGER NOT NULL,
                run_date TEXT NOT NULL,
                ticket_keys TEXT,
                label TEXT,
                size INTEGER,
                is_new INTEGER DEFAULT 0,
                growth_rate REAL,
                PRIMARY KEY (cluster_id, run_date)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO ticket_clusters (cluster_id, run_date, ticket_keys, label, size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (1, "2026-06-04", "[]", "incident", 1),
        )
        conn.commit()
        conn.close()

        with patch("db.DB_PATH", db_path):
            init_db()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        pk_cols = [
            row["name"]
            for row in conn.execute("PRAGMA table_info(ticket_clusters)").fetchall()
            if row["pk"] > 0
        ]
        assert pk_cols == ["cluster_id", "run_date", "run_type"]

        conn.execute(
            """
            INSERT INTO ticket_clusters (cluster_id, run_date, run_type, ticket_keys, label, size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "2026-06-04", "trend", "[]", "trend", 1),
        )
        conn.commit()

        rows = conn.execute(
            """
            SELECT run_type FROM ticket_clusters
            WHERE cluster_id = ? AND run_date = ?
            ORDER BY run_type
            """,
            (1, "2026-06-04"),
        ).fetchall()
        conn.close()

        assert [row["run_type"] for row in rows] == ["incident", "trend"]

    def test_init_db_already_migrated_no_op(self, tmp_path):
        """init_db() on an already-migrated schema leaves data intact."""
        db_path = str(tmp_path / "already_migrated.db")
        with patch("db.DB_PATH", db_path):
            init_db()
            # Seed a row so we can verify it survives a second init
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT INTO ticket_clusters
                    (cluster_id, run_date, run_type, ticket_keys, label, size)
                VALUES (1, '2026-06-05', 'incident', '[]', 'test', 2)
                """
            )
            conn.commit()
            conn.close()
            # Second init_db() must not wipe the table or change the schema
            init_db()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT label FROM ticket_clusters WHERE cluster_id = 1 AND run_date = '2026-06-05'"
        ).fetchone()
        pk_cols = [
            r["name"]
            for r in conn.execute("PRAGMA table_info(ticket_clusters)").fetchall()
            if r["pk"] > 0
        ]
        conn.close()

        assert row is not None and row["label"] == "test"
        assert pk_cols == ["cluster_id", "run_date", "run_type"]
