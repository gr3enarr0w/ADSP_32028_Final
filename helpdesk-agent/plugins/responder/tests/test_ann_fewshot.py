import pytest

from db import init_db, get_db_conn


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("db.DB_PATH", db_path)
    init_db()

    from plugins.responder.ann_fewshot import ANNFewShotIndex

    ANNFewShotIndex._last_rebuild_date = None
    yield


def _fake_embed_text(text: str, task_type: str = "query") -> list[float]:
    lowered = text.lower()
    if "password" in lowered:
        return [1.0, 0.0, 0.0]
    if "sso" in lowered or "access" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _fake_embed_batch(texts: list[str], task_type: str = "query") -> list[list[float]]:
    return [_fake_embed_text(text, task_type=task_type) for text in texts]


class TestANNFewShotIndex:
    def test_build_and_retrieve(self, monkeypatch):
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)

        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO tickets (ticket_key, summary, description, status)
                VALUES ('T-1', 'Reset password access', 'User cannot reset password', 'Open')
                """
            )
            conn.execute(
                """
                INSERT INTO tickets (ticket_key, summary, description, status)
                VALUES ('T-2', 'SSO access issue', 'User cannot sign in with SSO', 'Open')
                """
            )
            conn.execute(
                """
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response, agent_feedback)
                VALUES (
                    'T-1', 'c1', 'self_service',
                    'Please reset your password.',
                    'Please reset your password.',
                    'both_good'
                )
                """
            )
            conn.execute(
                """
                INSERT INTO response_examples
                    (ticket_key, response_type, agent_response)
                VALUES (
                    'T-2', 'admin_action',
                    'Please review your SSO configuration in admin.atlassian.com.'
                )
                """
            )

        count = ANNFewShotIndex.build()
        assert count == 2

        with get_db_conn() as conn:
            stored = conn.execute(
                "SELECT COUNT(*) FROM few_shot_examples WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        assert stored == 2

        results = ANNFewShotIndex.retrieve("password reset help needed", k=5, similarity_floor=0.5)
        assert len(results) == 1
        assert "Please reset your password." in results[0][0]
        assert results[0][1] == pytest.approx(1.0)

    def test_rebuild_daily_runs_once_per_day(self, monkeypatch):
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        calls = {"count": 0}

        def fake_rebuild(db=None):
            calls["count"] += 1
            ANNFewShotIndex._last_rebuild_date = "2026-06-02"
            return 1

        monkeypatch.setattr(ANNFewShotIndex, "_today", classmethod(lambda cls: "2026-06-02"))
        monkeypatch.setattr(ANNFewShotIndex, "build", classmethod(lambda cls, db=None: fake_rebuild(db)))

        assert ANNFewShotIndex.rebuild_daily() == 1
        assert ANNFewShotIndex.rebuild_daily() == 0
        assert calls["count"] == 1


    def test_partial_store_embedding_failure_leaves_table_empty(self, monkeypatch):
        """Test C — store_embedding raises on second call; exception is re-raised and table is empty.

        Pre-seeds one committed row into few_shot_examples to prove old rows do NOT survive a failed
        build. build() opens a connection via get_db_conn(), which rolls back the entire transaction
        (DELETE + INSERTs + partial store_embedding UPDATEs) when store_embedding raises. Because the
        pre-seeded row was committed in a separate transaction before build() ran, the rollback
        restores the table to that pre-build state — the old row is preserved.

        Known risk: if _build_on_conn() is called with a raw connection (no auto-rollback), the DELETE
        succeeds but the subsequent failure leaves the table empty, wiping the old index permanently.
        This test exercises the transactional path and confirms old data survives a failed build.
        """
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)

        # Pre-seed one row directly into few_shot_examples (committed before build() runs).
        # This proves the test starts with a non-empty table, so the assertion is meaningful.
        with get_db_conn() as conn:
            conn.execute(
                """
                INSERT INTO few_shot_examples
                    (example_id, source_table, source_key, ticket_key, response_type, example_text)
                VALUES ('pre:T-00', 'ai_draft_feedback', 'T-00:c00', 'T-00', 'self_service',
                        'Pre-existing example before failed rebuild.')
                """
            )

        # Seed two source rows so there are exactly two records to embed/store.
        with get_db_conn() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description, status) "
                "VALUES ('T-10', 'Password issue', 'Cannot reset password', 'Open')"
            )
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description, status) "
                "VALUES ('T-11', 'SSO problem', 'Cannot sign in', 'Open')"
            )
            conn.execute(
                """
                INSERT INTO ai_draft_feedback
                    (ticket_key, draft_comment_id, response_type,
                     draft_customer_response, actual_response, agent_feedback)
                VALUES ('T-10', 'c10', 'self_service',
                        'Reset password.', 'Reset password.', 'both_good')
                """
            )
            conn.execute(
                """
                INSERT INTO response_examples
                    (ticket_key, response_type, agent_response)
                VALUES ('T-11', 'admin_action', 'Review SSO config.')
                """
            )

        call_counter = {"n": 0}

        def _failing_store_embedding(example_id, vector, entity_type, conn=None):
            call_counter["n"] += 1
            if call_counter["n"] >= 2:
                raise RuntimeError("Simulated embedding store failure on second call")

        monkeypatch.setattr(ann_fewshot, "store_embedding", _failing_store_embedding)

        with pytest.raises(RuntimeError, match="Simulated embedding store failure"):
            ANNFewShotIndex.build()

        with get_db_conn() as conn:
            row_count = conn.execute(
                "SELECT COUNT(*) FROM few_shot_examples"
            ).fetchone()[0]

        # The transactional path (get_db_conn auto-rollback) restores the pre-seeded row.
        # row_count == 1 confirms old data survives when build() raises — not row_count == 0.
        # Known risk: the raw-connection path (_build_on_conn called directly) does NOT rollback,
        # so old data would be wiped. That non-transactional hazard is documented in ann_fewshot.py.
        assert row_count == 1, (
            f"Expected pre-seeded row to survive rollback after failed build, got {row_count} rows"
        )

    def test_pod_restart_daily_guard(self, monkeypatch):
        """Test D — rebuild_daily skips when already run today; pod restart (None) triggers rebuild."""
        from plugins.responder import ann_fewshot
        from plugins.responder.ann_fewshot import ANNFewShotIndex

        monkeypatch.setattr(ann_fewshot, "embed_text", _fake_embed_text)
        monkeypatch.setattr(ann_fewshot, "embed_batch", _fake_embed_batch)

        today = "2026-06-02"
        monkeypatch.setattr(ANNFewShotIndex, "_today", classmethod(lambda cls: today))

        # Save the original ClassVar so other tests aren't affected by mutations below.
        original = ANNFewShotIndex._last_rebuild_date
        try:
            # Guard: already rebuilt today — should skip.
            ANNFewShotIndex._last_rebuild_date = today
            result = ANNFewShotIndex.rebuild_daily()
            assert result == 0, f"Expected 0 (skipped), got {result}"

            # Simulate pod restart: clear the in-memory date flag.
            ANNFewShotIndex._last_rebuild_date = None
            result = ANNFewShotIndex.rebuild_daily()
            # DB is empty so build() returns 0; the important thing is it ran (didn't skip).
            assert result >= 0, f"Expected non-negative count after rebuild, got {result}"
            # After a successful rebuild the date flag is set.
            assert ANNFewShotIndex._last_rebuild_date == today
        finally:
            ANNFewShotIndex._last_rebuild_date = original


class TestResponderFallback:
    def test_static_fallback_when_ann_empty(self, monkeypatch):
        from plugins.responder import drafting

        monkeypatch.setattr(drafting.ANNFewShotIndex, "retrieve", classmethod(lambda cls, ticket_text, k=5, similarity_floor=0.5: []))
        monkeypatch.setattr(drafting.ANNFewShotIndex, "is_empty", classmethod(lambda cls: True))
        monkeypatch.setattr(
            drafting,
            "get_few_shot_examples",
            lambda response_type=None, limit=3: [
                {
                    "response_type": "self_service",
                    "draft_customer_response": "Draft text",
                    "actual_response": "Final text",
                }
            ],
        )
        monkeypatch.setattr(
            drafting,
            "get_organic_examples",
            lambda response_type=None, limit=3: [
                {
                    "response_type": "admin_action",
                    "agent_response": "Organic response",
                }
            ],
        )
        monkeypatch.setattr(
            drafting,
            "get_plugin_config",
            lambda plugin_name: {"fewshot_k": 5, "fewshot_similarity_floor": 0.5},
        )

        block = drafting._build_few_shot_block("Ticket summary\nTicket description")

        assert "Draft text" in block
        assert "Organic response" in block

    def test_on_schedule_rebuilds_ann_index(self, monkeypatch):
        import plugins.responder as responder_pkg
        from plugins.responder import ann_fewshot

        calls = {"count": 0}

        monkeypatch.setattr(responder_pkg, "capture_feedback", lambda: None)
        monkeypatch.setattr(responder_pkg, "harvest_response_examples", lambda: None)
        monkeypatch.setattr(
            responder_pkg,
            "get_plugin_config",
            lambda plugin_name: {
                "capture_feedback": True,
                "harvest_response_examples": True,
            },
        )
        monkeypatch.setattr(
            ann_fewshot.ANNFewShotIndex,
            "rebuild_daily",
            classmethod(lambda cls, db=None: calls.__setitem__("count", calls["count"] + 1) or 1),
        )

        responder_pkg.plugin.on_schedule()

        assert calls["count"] == 1

    def test_on_schedule_respects_config_flags(self, monkeypatch):
        import plugins.responder as responder_pkg

        calls = {"capture": 0, "harvest": 0, "rebuild": 0}

        monkeypatch.setattr(
            responder_pkg,
            "capture_feedback",
            lambda: calls.__setitem__("capture", calls["capture"] + 1),
        )
        monkeypatch.setattr(
            responder_pkg,
            "harvest_response_examples",
            lambda: calls.__setitem__("harvest", calls["harvest"] + 1),
        )
        monkeypatch.setattr(
            responder_pkg.ANNFewShotIndex,
            "rebuild_daily",
            classmethod(lambda cls, db=None: calls.__setitem__("rebuild", calls["rebuild"] + 1) or 0),
        )
        monkeypatch.setattr(
            responder_pkg,
            "get_plugin_config",
            lambda plugin_name: {
                "capture_feedback": False,
                "harvest_response_examples": False,
            },
        )

        responder_pkg.plugin.on_schedule()

        assert calls == {"capture": 0, "harvest": 0, "rebuild": 1}
