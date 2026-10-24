"""Model switches must never mix vector spaces or discard a working index."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app

GEMMA = "google/embeddinggemma-2"
BGE = "BAAI/bge-small-en-v1.5"


class EmbeddingSelectionTests(unittest.TestCase):
    def setUp(self):
        self.selected = app._SELECTED_EMBEDDING_MODEL
        app._SELECTED_EMBEDDING_MODEL = None
        app.DOCUMENT_INDEX.clear()
        self.environment = patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local", "LOCAL_EMBEDDING_MODEL": GEMMA})
        self.environment.start()

    def tearDown(self):
        app.DOCUMENT_INDEX.clear()
        app._SELECTED_EMBEDDING_MODEL = self.selected
        self.environment.stop()

    def test_each_model_uses_its_calibrated_gate(self):
        self.assertIs(app.runtime_settings(), app.DEFAULT_SETTINGS)
        token = app._EMBEDDING_OVERRIDE.set(BGE)
        try:
            self.assertEqual(app.runtime_settings().embedding_min_similarity, 0.555)
            self.assertEqual(app.embedding_config(), ("local", BGE, "", ""))
            self.assertEqual(app.DEFAULT_SETTINGS.embedding_min_similarity, 0.63)
        finally:
            app._EMBEDDING_OVERRIDE.reset(token)

    def test_model_selection_without_a_pdf_is_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            store = app.PersistentDocumentStore(Path(temp))
            try:
                with patch("app.document_store", return_value=store), patch("app.local_embedder"):
                    result = app.switch_embedding_model(BGE)
                self.assertEqual(result["embedding_model"], BGE)
                self.assertEqual(store.selected_embedding_model(), BGE)
                self.assertFalse(result["document"]["loaded"])
            finally:
                store.close()

    def test_switches_rebuild_vectors_in_both_directions(self):
        chunks = [app.DocumentChunk(number=1, text="A useful passage.", page=1)]
        def vectors(texts, *args, **kwargs):
            return [[1.0, 0.0, 0.0] if app.embedding_config()[1] == GEMMA else [1.0, 0.0] for _ in texts]
        with tempfile.TemporaryDirectory() as temp:
            store = app.PersistentDocumentStore(Path(temp))
            try:
                with patch("app.document_store", return_value=store), patch("app.local_embedder"), patch("app.parse_pdf", return_value=(chunks, 1, set())), patch("app.embed_texts", side_effect=vectors):
                    app.index_document("paper.pdf", b"%PDF-test", app.DEFAULT_SETTINGS, store)
                    for model, dimensions in [(BGE, 2), (GEMMA, 3)]:
                        result = app.switch_embedding_model(model)
                        self.assertEqual(result["document"]["embedding_model"], model)
                        restored = store.load_latest()
                        self.assertEqual(restored.embedding_model, model)
                        self.assertEqual(len(restored.embeddings[0]), dimensions)
                        self.assertEqual(store.selected_embedding_model(), model)
            finally:
                store.close()

    def test_failed_embedding_keeps_previous_model_and_document(self):
        chunks = [app.DocumentChunk(number=1, text="A useful passage.", page=1)]
        with tempfile.TemporaryDirectory() as temp:
            store = app.PersistentDocumentStore(Path(temp))
            try:
                with patch("app.parse_pdf", return_value=(chunks, 1, set())), patch("app.embed_texts", return_value=[[1.0, 0.0, 0.0]]):
                    app.index_document("paper.pdf", b"%PDF-test", app.DEFAULT_SETTINGS, store)
                previous = app.DOCUMENT_INDEX.status()
                with patch("app.document_store", return_value=store), patch("app.local_embedder"), patch("app.parse_pdf", return_value=(chunks, 1, set())), patch("app.embed_texts", side_effect=RuntimeError("model unavailable")):
                    with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                        app.switch_embedding_model(BGE)
                self.assertEqual(app.embedding_config()[1], GEMMA)
                self.assertEqual(app.DOCUMENT_INDEX.status(), previous)
                self.assertEqual(store.load_latest().embedding_model, GEMMA)
                self.assertIsNone(app._EMBEDDING_OVERRIDE.get())
            finally:
                store.close()

    def test_unavailable_model_does_not_change_selection(self):
        with patch("app.document_store"), patch("app.local_embedder", side_effect=RuntimeError("missing weights")):
            with self.assertRaisesRegex(RuntimeError, "missing weights"):
                app.switch_embedding_model(BGE)
        self.assertEqual(app.embedding_config()[1], GEMMA)

    def test_unknown_model_is_rejected_before_loading(self):
        with patch("app.local_embedder") as loader:
            for model in ["untrusted/model", None, [GEMMA]]:
                with self.assertRaises(ValueError):
                    app.switch_embedding_model(model)
            loader.assert_not_called()

    def test_endpoint_rejects_invalid_payload(self):
        for data in [[], {"model": "untrusted/model"}]:
            payload = json.dumps(data).encode()
            handler = app.ChatHandler.__new__(app.ChatHandler)
            handler.path = "/api/embedding-model"
            handler.headers = {"Content-Length": str(len(payload))}
            handler.rfile = io.BytesIO(payload)
            handler.send_json = Mock()
            handler.do_POST()
            self.assertEqual(handler.send_json.call_args.args[0], 400)
