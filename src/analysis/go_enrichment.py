"""GO Enrichment Analysis for PRISM-Bio.

This module provides Gene Ontology enrichment analysis for protein clusters.
Supports both SwissProt (extracting GO from dataset columns) and UniProt
(loading from GAF annotation files).
"""
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.utils.logging_utils import get_logger, LogContext

logger = get_logger("analysis.go_enrichment")

# GO ontology URL
GO_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
# Use project directory for cache to avoid home directory quota issues
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "go"


def download_go_ontology(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Download GO ontology OBO file if not present.

    Args:
        cache_dir: Directory to cache downloaded files
        force_download: Force re-download even if file exists

    Returns:
        Path to the OBO file

    Raises:
        RuntimeError: If download fails
    """
    import requests

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    obo_path = cache_dir / "go-basic.obo"

    if obo_path.exists() and not force_download:
        logger.info(f"Using cached GO ontology: {obo_path}")
        return obo_path

    logger.info(f"Downloading GO ontology from {GO_OBO_URL}")

    try:
        response = requests.get(GO_OBO_URL, timeout=60)
        response.raise_for_status()

        with open(obo_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        logger.info(f"GO ontology downloaded to: {obo_path}")
        return obo_path

    except requests.RequestException as e:
        logger.error(f"Failed to download GO ontology: {e}")
        raise RuntimeError(f"Failed to download GO ontology: {e}") from e


def extract_go_annotations_swissprot(
    sequences: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Extract GO annotations from SwissProt dataset columns.

    SwissProt datasets typically have GO annotations in columns like:
    - "Gene Ontology (biological process)"
    - "Gene Ontology (molecular function)"
    - "Gene Ontology (cellular component)"

    The format is usually semicolon-separated GO terms with names,
    e.g., "cytoplasm [GO:0005737]; nucleus [GO:0005634]"

    Args:
        sequences: List of sequence dicts from SwissProt dataset

    Returns:
        Dict mapping protein_id to list of GO term IDs (e.g., ["GO:0005737", "GO:0005634"])
    """
    logger.info("Extracting GO annotations from SwissProt dataset")

    # GO columns in SwissProt
    go_columns = [
        "Gene Ontology (biological process)",
        "Gene Ontology (molecular function)",
        "Gene Ontology (cellular component)",
    ]

    # Regex to extract GO IDs (format: GO:XXXXXXX)
    go_pattern = re.compile(r"GO:\d{7}")

    annotations = defaultdict(list)
    proteins_with_go = 0
    total_go_terms = 0

    for seq in sequences:
        protein_id = seq.get("id", "")
        if not protein_id:
            continue

        go_terms = set()

        for col in go_columns:
            value = seq.get(col, "")
            if value and isinstance(value, str):
                # Extract all GO IDs from the column
                matches = go_pattern.findall(value)
                go_terms.update(matches)

        if go_terms:
            annotations[protein_id] = list(go_terms)
            proteins_with_go += 1
            total_go_terms += len(go_terms)

    logger.info(f"Extracted GO annotations for {proteins_with_go}/{len(sequences)} proteins")
    logger.info(f"Total GO terms found: {total_go_terms}")

    return dict(annotations)


def load_go_annotations_uniprot(
    annotation_file: str,
) -> Dict[str, List[str]]:
    """Load GO annotations from UniProt GAF (Gene Association Format) file.

    GAF format is tab-separated with columns:
    0: DB (database)
    1: DB_Object_ID (protein ID)
    2: DB_Object_Symbol
    3: Qualifier
    4: GO_ID
    ...

    Args:
        annotation_file: Path to GAF annotation file

    Returns:
        Dict mapping protein_id to list of GO term IDs

    Raises:
        FileNotFoundError: If annotation file doesn't exist
    """
    logger.info(f"Loading GO annotations from GAF file: {annotation_file}")

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"GAF annotation file not found: {annotation_file}")

    annotations = defaultdict(list)
    line_count = 0
    proteins_with_go = 0

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comments
            if line.startswith("!"):
                continue

            line_count += 1
            parts = line.strip().split("\t")

            if len(parts) < 5:
                continue

            # GAF columns
            db = parts[0]
            protein_id = parts[1]
            go_id = parts[4]

            # Validate GO ID format
            if not go_id.startswith("GO:"):
                continue

            # Handle UniProtKB IDs
            if db in ("UniProtKB", "UniProt"):
                # Use the accession number as ID
                if go_id not in annotations[protein_id]:
                    annotations[protein_id].append(go_id)

    proteins_with_go = len(annotations)
    total_go_terms = sum(len(terms) for terms in annotations.values())

    logger.info(f"Loaded {line_count} annotation lines")
    logger.info(f"Found annotations for {proteins_with_go} proteins")
    logger.info(f"Total GO terms: {total_go_terms}")

    return dict(annotations)


def perform_go_enrichment(
    study_protein_ids: List[str],
    population_protein_ids: List[str],
    protein_to_go: Dict[str, List[str]],
    obo_path: str,
    p_value_threshold: float = 0.05,
    correction_method: str = "fdr_bh",
    min_genes: int = 3,
    namespaces: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Perform GO enrichment analysis using goatools.

    Args:
        study_protein_ids: Protein IDs in the study set (cluster)
        population_protein_ids: All protein IDs (background)
        protein_to_go: Mapping of protein ID to GO term IDs
        obo_path: Path to GO OBO file
        p_value_threshold: P-value threshold for significance
        correction_method: Multiple testing correction method
        min_genes: Minimum genes per GO term
        namespaces: GO namespaces to include (default: all)

    Returns:
        List of enriched GO term dicts with statistics
    """
    from goatools.obo_parser import GODag
    from goatools.go_enrichment import GOEnrichmentStudy
    from goatools.anno.genetogo_reader import Gene2GoReader

    logger.info("Performing GO enrichment analysis")
    logger.debug(f"  Study set size: {len(study_protein_ids)}")
    logger.debug(f"  Population size: {len(population_protein_ids)}")
    logger.debug(f"  P-value threshold: {p_value_threshold}")
    logger.debug(f"  Correction method: {correction_method}")

    if namespaces is None:
        namespaces = ["biological_process", "molecular_function", "cellular_component"]

    # Load GO ontology
    logger.debug(f"Loading GO DAG from: {obo_path}")
    go_dag = GODag(obo_path)

    # Build gene-to-GO associations for goatools
    # Format: {gene_id: set of GO IDs}
    gene2go = {}
    for protein_id in population_protein_ids:
        go_terms = protein_to_go.get(protein_id, [])
        if go_terms:
            gene2go[protein_id] = set(go_terms)

    if not gene2go:
        logger.warning("No GO annotations found for population proteins")
        return []

    # Filter study set to proteins with GO annotations
    study_set = set(study_protein_ids) & set(gene2go.keys())
    population_set = set(population_protein_ids) & set(gene2go.keys())

    if len(study_set) < min_genes:
        logger.warning(f"Study set too small ({len(study_set)} < {min_genes} min_genes)")
        return []

    logger.debug(f"  Proteins with GO in study: {len(study_set)}")
    logger.debug(f"  Proteins with GO in population: {len(population_set)}")

    # Create GOEnrichmentStudy
    methods = []
    if correction_method == "fdr_bh":
        methods = ["fdr_bh"]
    elif correction_method == "bonferroni":
        methods = ["bonferroni"]
    else:
        methods = ["fdr_bh"]  # Default

    goeaobj = GOEnrichmentStudy(
        population_set,
        gene2go,
        go_dag,
        propagate_counts=True,
        alpha=p_value_threshold,
        methods=methods,
    )

    # Run enrichment
    results = goeaobj.run_study(study_set)

    # Filter and format results
    enriched_terms = []

    for r in results:
        # Filter by namespace
        if r.NS not in ["BP", "MF", "CC"]:
            continue

        namespace_map = {"BP": "biological_process", "MF": "molecular_function", "CC": "cellular_component"}
        full_namespace = namespace_map.get(r.NS, r.NS)

        if full_namespace not in namespaces:
            continue

        # Filter by p-value (use corrected p-value if available)
        p_val = r.p_fdr_bh if hasattr(r, "p_fdr_bh") and r.p_fdr_bh is not None else r.p_uncorrected
        if p_val > p_value_threshold:
            continue

        # Filter by minimum genes
        if r.study_count < min_genes:
            continue

        enriched_terms.append({
            "go_id": r.GO,
            "go_name": r.name,
            "namespace": full_namespace,
            "p_value": r.p_uncorrected,
            "p_value_corrected": p_val,
            "study_count": r.study_count,
            "study_total": len(study_set),
            "population_count": r.pop_count,
            "population_total": len(population_set),
            "enrichment_ratio": (r.study_count / len(study_set)) / (r.pop_count / len(population_set)) if r.pop_count > 0 else 0,
        })

    # Sort by corrected p-value
    enriched_terms.sort(key=lambda x: x["p_value_corrected"])

    logger.info(f"Found {len(enriched_terms)} significantly enriched GO terms")

    return enriched_terms


def get_top_go_terms(
    enrichment_results: List[Dict[str, Any]],
    n_terms: int = 5,
    per_namespace: bool = False,
) -> List[str]:
    """Extract top enriched GO term names for description.

    Args:
        enrichment_results: Results from perform_go_enrichment()
        n_terms: Number of top terms to return
        per_namespace: If True, return n_terms per namespace

    Returns:
        List of GO term names
    """
    if not enrichment_results:
        return []

    if per_namespace:
        # Get top N per namespace
        by_namespace = defaultdict(list)
        for result in enrichment_results:
            by_namespace[result["namespace"]].append(result)

        terms = []
        for ns in ["biological_process", "molecular_function", "cellular_component"]:
            ns_results = by_namespace.get(ns, [])
            for result in ns_results[:n_terms]:
                terms.append(result["go_name"])

        return terms
    else:
        # Get overall top N
        return [r["go_name"] for r in enrichment_results[:n_terms]]


def run_cluster_go_enrichment(
    cluster_sequences: Dict[int, List[Dict[str, Any]]],
    all_sequences: List[Dict[str, Any]],
    config: Any,
    experiment_name: str,
    output_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    """Run GO enrichment for all clusters.

    Args:
        cluster_sequences: Dict mapping cluster_id to list of sequence dicts
        all_sequences: All sequences in the dataset
        config: Configuration object with go_enrichment settings
        experiment_name: Name of the experiment
        output_dir: Output directory for enrichment results

    Returns:
        Dict mapping cluster_id to enrichment results dict with keys:
        - "enriched_terms": List of enriched GO terms
        - "description": GO-based description string
        - "top_terms": List of top GO term names
    """
    from src.utils.output import save_go_enrichment_results

    logger.info("=" * 80)
    logger.info("GO ENRICHMENT ANALYSIS")
    logger.info("=" * 80)

    go_config = config.go_enrichment

    # Check if enrichment is enabled
    if not go_config.enabled:
        logger.info("GO enrichment is disabled in config")
        return {}

    # Extract GO annotations based on dataset type
    dataset_name = config.dataset.hf_dataset_name.lower() if hasattr(config.dataset, "hf_dataset_name") else ""

    if "swissprot" in dataset_name or "swiss" in dataset_name:
        logger.info("Extracting GO annotations from SwissProt dataset")
        protein_to_go = extract_go_annotations_swissprot(all_sequences)
    elif go_config.annotation_file:
        logger.info(f"Loading GO annotations from file: {go_config.annotation_file}")
        protein_to_go = load_go_annotations_uniprot(go_config.annotation_file)
    else:
        logger.warning("No GO annotations available. Skipping enrichment.")
        logger.warning("For UniProt datasets, provide annotation_file in config.")
        return {}

    if not protein_to_go:
        logger.warning("No GO annotations extracted. Skipping enrichment.")
        return {}

    # Get or download GO ontology
    obo_path = go_config.obo_file
    if obo_path is None:
        try:
            obo_path = str(download_go_ontology())
        except Exception as e:
            logger.error(f"Failed to download GO ontology: {e}")
            return {}

    # Get all protein IDs (population)
    all_protein_ids = [seq.get("id", "") for seq in all_sequences if seq.get("id")]

    # Run enrichment for each cluster
    cluster_results = {}

    for cluster_id, cluster_seqs in cluster_sequences.items():
        logger.info(f"\n--- Cluster {cluster_id} GO Enrichment ---")

        # Get cluster protein IDs
        cluster_protein_ids = [seq.get("id", "") for seq in cluster_seqs if seq.get("id")]

        if not cluster_protein_ids:
            logger.warning(f"No protein IDs for cluster {cluster_id}")
            continue

        try:
            # Run enrichment
            enriched_terms = perform_go_enrichment(
                study_protein_ids=cluster_protein_ids,
                population_protein_ids=all_protein_ids,
                protein_to_go=protein_to_go,
                obo_path=obo_path,
                p_value_threshold=go_config.p_value_threshold,
                correction_method=go_config.correction_method,
                min_genes=go_config.min_genes,
                namespaces=go_config.namespaces,
            )

            # Get top terms for description
            top_terms = get_top_go_terms(enriched_terms, n_terms=3)

            # Create description
            if top_terms:
                description = f"Cluster {cluster_id}: {', '.join(top_terms)} ({len(cluster_seqs)} samples)"
            else:
                description = f"Cluster {cluster_id} with {len(cluster_seqs)} samples (no enriched GO terms)"

            cluster_results[cluster_id] = {
                "enriched_terms": enriched_terms,
                "description": description,
                "top_terms": top_terms,
            }

            # Log results
            logger.info(f"Cluster {cluster_id}: {len(enriched_terms)} enriched GO terms")
            if top_terms:
                logger.info(f"  Top terms: {', '.join(top_terms)}")

            # Save detailed enrichment results
            if output_dir and enriched_terms:
                save_go_enrichment_results(
                    enrichment_results=enriched_terms,
                    cluster_id=cluster_id,
                    experiment_name=experiment_name,
                    output_dir=output_dir,
                )

        except Exception as e:
            logger.error(f"GO enrichment failed for cluster {cluster_id}: {e}")
            cluster_results[cluster_id] = {
                "enriched_terms": [],
                "description": f"Cluster {cluster_id} with {len(cluster_seqs)} samples (enrichment failed)",
                "top_terms": [],
            }

    logger.info(f"\nGO enrichment complete for {len(cluster_results)} clusters")

    return cluster_results

