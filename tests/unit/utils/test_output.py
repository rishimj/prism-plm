"""Tests for output utilities."""
import csv
import json
from pathlib import Path

import pytest

from src.utils.output import (
    save_cluster_statistics,
    save_description_csv,
    save_evaluation_csv,
    save_explanation_csv,
    save_go_enrichment_results,
    save_representative_sequences,
)


class TestSaveDescriptionCsv:
    """Tests for save_description_csv function."""

    def test_creates_csv_file(self, tmp_output_dir, sample_descriptions):
        """Should create a CSV file."""
        filepath = save_description_csv(
            descriptions=sample_descriptions,
            model_name="test_model",
            target_model="esm2_t6_8M",
            layer_id=18,
            unit_id=100,
            output_dir=tmp_output_dir / "descriptions",
        )
        assert filepath.exists()
        assert filepath.suffix == ".csv"

    def test_csv_has_correct_headers(self, tmp_output_dir, sample_descriptions):
        """CSV should have correct column headers."""
        filepath = save_description_csv(
            descriptions=sample_descriptions,
            model_name="test_model",
            target_model="esm2_t6_8M",
            layer_id=18,
            unit_id=100,
            output_dir=tmp_output_dir / "descriptions",
        )

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "layer" in headers
        assert "unit" in headers
        assert "description" in headers
        assert "mean_activation" in headers
        assert "highlights" in headers

    def test_csv_contains_all_descriptions(self, tmp_output_dir, sample_descriptions):
        """CSV should contain all descriptions."""
        filepath = save_description_csv(
            descriptions=sample_descriptions,
            model_name="test_model",
            target_model="esm2_t6_8M",
            layer_id=18,
            unit_id=100,
            output_dir=tmp_output_dir / "descriptions",
        )

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(sample_descriptions)

    def test_filename_contains_layer_and_unit(self, tmp_output_dir, sample_descriptions):
        """Filename should contain layer and unit IDs."""
        filepath = save_description_csv(
            descriptions=sample_descriptions,
            model_name="test_model",
            target_model="esm2_t6_8M",
            layer_id=18,
            unit_id=100,
            output_dir=tmp_output_dir / "descriptions",
        )

        assert "layer-18" in filepath.name
        assert "unit-100" in filepath.name

    def test_creates_nested_directories(self, tmp_path, sample_descriptions):
        """Should create nested directories if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c"
        assert not deep_path.exists()

        save_description_csv(
            descriptions=sample_descriptions,
            model_name="test_model",
            target_model="esm2_t6_8M",
            layer_id=18,
            unit_id=100,
            output_dir=deep_path,
        )

        assert deep_path.exists()

    def test_handles_empty_descriptions(self, tmp_output_dir):
        """Should handle empty description list."""
        filepath = save_description_csv(
            descriptions=[],
            model_name="test_model",
            target_model="esm2_t6_8M",
            layer_id=18,
            unit_id=100,
            output_dir=tmp_output_dir / "descriptions",
        )

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 0

    def test_custom_timestamp(self, tmp_output_dir, sample_descriptions):
        """Should use custom timestamp if provided."""
        filepath = save_description_csv(
            descriptions=sample_descriptions,
            model_name="test_model",
            target_model="esm2_t6_8M",
            layer_id=18,
            unit_id=100,
            output_dir=tmp_output_dir / "descriptions",
            timestamp="2024-01-01_12-00-00",
        )

        assert "2024-01-01_12-00-00" in filepath.name


class TestSaveExplanationCsv:
    """Tests for save_explanation_csv function."""

    def test_creates_csv_file(self, tmp_output_dir):
        """Should create a CSV file."""
        explanations = [
            {"layer": 18, "unit": 100, "description": "test", "input_success": True}
        ]
        filepath = save_explanation_csv(
            explanations=explanations,
            model_name="esm2_t6_8M",
            method_name="output-centric",
            layers=[0, 18, 36],
            n_samples=60,
            output_dir=tmp_output_dir,
        )
        assert filepath.exists()

    def test_filename_contains_layers(self, tmp_output_dir):
        """Filename should contain layer information."""
        explanations = [{"layer": 18, "unit": 100}]
        filepath = save_explanation_csv(
            explanations=explanations,
            model_name="esm2_t6_8M",
            method_name="test",
            layers=[0, 18, 36],
            n_samples=60,
            output_dir=tmp_output_dir,
        )

        assert "layer0" in filepath.name
        assert "layer18" in filepath.name
        assert "layer36" in filepath.name

    def test_handles_empty_explanations(self, tmp_output_dir):
        """Should handle empty explanations list."""
        filepath = save_explanation_csv(
            explanations=[],
            model_name="esm2_t6_8M",
            method_name="test",
            layers=[0],
            n_samples=0,
            output_dir=tmp_output_dir,
        )
        assert filepath.exists()


class TestSaveEvaluationCsv:
    """Tests for save_evaluation_csv function."""

    def test_creates_csv_file(self, tmp_output_dir, sample_evaluation_results):
        """Should create a CSV file."""
        filepath = save_evaluation_csv(
            results=sample_evaluation_results,
            method_name="cosy",
            target_model="esm2_t6_8M",
            text_gen_model="llama-3.1",
            eval_gen_model="llama-3.1",
            aggregation="mean",
            dataset="uniref50",
            n_samples=1000,
            output_dir=tmp_output_dir,
        )
        assert filepath.exists()

    def test_filename_format(self, tmp_output_dir, sample_evaluation_results):
        """Filename should follow PRISM format."""
        filepath = save_evaluation_csv(
            results=sample_evaluation_results,
            method_name="cosy",
            target_model="esm2_t6_8M",
            text_gen_model="llama-3.1",
            eval_gen_model="llama-3.1",
            aggregation="mean",
            dataset="uniref50",
            n_samples=1000,
            output_dir=tmp_output_dir,
        )

        assert "cosy-evaluation" in filepath.name
        assert "target-" in filepath.name
        assert "textgen-" in filepath.name
        assert "evalgen-" in filepath.name


class TestSaveClusterStatistics:
    """Tests for save_cluster_statistics function."""

    def test_creates_json_file(self, tmp_output_dir, cluster_statistics):
        """Should create a JSON file."""
        filepath = save_cluster_statistics(
            statistics=cluster_statistics,
            experiment_name="test_exp",
            output_dir=tmp_output_dir,
        )
        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_json_is_valid(self, tmp_output_dir, cluster_statistics):
        """JSON should be valid and loadable."""
        filepath = save_cluster_statistics(
            statistics=cluster_statistics,
            experiment_name="test_exp",
            output_dir=tmp_output_dir,
        )

        with open(filepath, "r") as f:
            loaded = json.load(f)

        assert loaded["n_clusters"] == cluster_statistics["n_clusters"]


class TestSaveRepresentativeSequences:
    """Tests for save_representative_sequences function."""

    def test_creates_text_file(self, tmp_output_dir, valid_sequences):
        """Should create a text file."""
        sequences_by_cluster = {
            0: valid_sequences[:2],
            1: valid_sequences[2:4],
        }
        filepath = save_representative_sequences(
            sequences=sequences_by_cluster,
            experiment_name="test_exp",
            output_dir=tmp_output_dir,
        )
        assert filepath.exists()
        assert filepath.suffix == ".txt"

    def test_contains_cluster_info(self, tmp_output_dir, valid_sequences):
        """File should contain cluster information."""
        sequences_by_cluster = {
            0: valid_sequences[:2],
            1: valid_sequences[2:4],
        }
        filepath = save_representative_sequences(
            sequences=sequences_by_cluster,
            experiment_name="test_exp",
            output_dir=tmp_output_dir,
        )

        with open(filepath, "r") as f:
            content = f.read()

        assert "CLUSTER 0" in content
        assert "CLUSTER 1" in content


class TestSaveGoEnrichmentResults:
    """Tests for save_go_enrichment_results function."""

    def test_creates_csv_file(self, tmp_output_dir):
        """Should create a CSV file."""
        results = [
            {"go_id": "GO:0003677", "term": "DNA binding", "p_value": 0.001},
            {"go_id": "GO:0005634", "term": "nucleus", "p_value": 0.01},
        ]
        filepath = save_go_enrichment_results(
            enrichment_results=results,
            cluster_id=0,
            experiment_name="test_exp",
            output_dir=tmp_output_dir,
        )
        assert filepath.exists()
        assert filepath.suffix == ".csv"

    def test_filename_contains_cluster_id(self, tmp_output_dir):
        """Filename should contain cluster ID."""
        results = [{"go_id": "GO:0003677", "p_value": 0.001}]
        filepath = save_go_enrichment_results(
            enrichment_results=results,
            cluster_id=5,
            experiment_name="test_exp",
            output_dir=tmp_output_dir,
        )
        assert "cluster5" in filepath.name






