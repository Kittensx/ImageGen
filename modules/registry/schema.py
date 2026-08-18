from __future__ import annotations

REGISTRY_SCHEMA_VERSION = 8

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    extension TEXT,
    file_size INTEGER,
    modified_time REAL,
    created_time REAL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    exists_on_disk INTEGER NOT NULL DEFAULT 1,

    quick_fingerprint TEXT,
    sha256 TEXT,
    blake3 TEXT,

    asset_type TEXT NOT NULL DEFAULT 'unclassified_asset',
    format_type TEXT NOT NULL DEFAULT 'other',
    architecture TEXT NOT NULL DEFAULT '',
    architecture_state TEXT NOT NULL DEFAULT 'observed_unclassified',

    checkpoint_kind TEXT NOT NULL DEFAULT 'unknown',
    has_unet INTEGER NOT NULL DEFAULT 0,
    has_vae INTEGER NOT NULL DEFAULT 0,
    has_text_encoder INTEGER NOT NULL DEFAULT 0,
    has_text_encoder_2 INTEGER NOT NULL DEFAULT 0,

    library_root TEXT,
    managed_category TEXT,
    path_kind TEXT NOT NULL DEFAULT 'external',

    key_count INTEGER,
    metadata_json TEXT,
    notes TEXT,
    location_state TEXT NOT NULL DEFAULT 'available'
);

CREATE INDEX IF NOT EXISTS idx_assets_filename ON assets(filename);
CREATE INDEX IF NOT EXISTS idx_assets_extension ON assets(extension);
CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON assets(sha256);
CREATE INDEX IF NOT EXISTS idx_assets_blake3 ON assets(blake3);
CREATE INDEX IF NOT EXISTS idx_assets_asset_type ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_assets_architecture ON assets(architecture);
CREATE INDEX IF NOT EXISTS idx_assets_library_root ON assets(library_root);
CREATE INDEX IF NOT EXISTS idx_assets_managed_category ON assets(managed_category);
CREATE INDEX IF NOT EXISTS idx_assets_path_kind ON assets(path_kind);

CREATE TABLE IF NOT EXISTS asset_inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    inspected_at TEXT NOT NULL,
    inspector_version TEXT,
    key_count INTEGER,
    prefix_summary_json TEXT,
    example_keys_json TEXT,
    dtype_summary_json TEXT,
    tensor_shape_summary_json TEXT,
    result_json TEXT,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_asset_inspections_asset_id ON asset_inspections(asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_inspections_inspected_at ON asset_inspections(inspected_at);

CREATE TABLE IF NOT EXISTS asset_components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    snapshot_at TEXT NOT NULL,
    snapshot_version TEXT NOT NULL,
    component_role TEXT NOT NULL,
    source_prefixes_json TEXT,
    tensor_count INTEGER NOT NULL,
    total_bytes INTEGER NOT NULL,
    component_sha256 TEXT NOT NULL,
    structure_sha256 TEXT NOT NULL,
    dtype_summary_json TEXT,
    tensor_manifest_json TEXT,
    metadata_json TEXT,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
    UNIQUE(asset_id, snapshot_version, component_role)
);

CREATE INDEX IF NOT EXISTS idx_asset_components_asset_id ON asset_components(asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_components_role ON asset_components(component_role);
CREATE INDEX IF NOT EXISTS idx_asset_components_sha256 ON asset_components(component_sha256);
CREATE INDEX IF NOT EXISTS idx_asset_components_structure_sha256 ON asset_components(structure_sha256);

-- Phase 01 normalized component identity. Existing asset_components rows remain
-- authoritative occurrence evidence and are backfilled into these tables without
-- reading or hashing model payload bytes.
CREATE TABLE IF NOT EXISTS component_identities (
    component_sha256 TEXT PRIMARY KEY,
    structure_sha256 TEXT,
    total_bytes INTEGER,
    tensor_count INTEGER,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_component_identities_structure ON component_identities(structure_sha256);

CREATE TABLE IF NOT EXISTS component_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_sha256 TEXT NOT NULL,
    asset_id INTEGER NOT NULL,
    component_role TEXT NOT NULL,
    source_form TEXT NOT NULL DEFAULT 'unknown',
    embedded_state TEXT NOT NULL DEFAULT 'unknown',
    provider_family TEXT,
    provider_version TEXT,
    availability_state TEXT NOT NULL DEFAULT 'unknown',
    locator_json TEXT,
    scan_timestamp TEXT,
    scanner_version TEXT,
    snapshot_version TEXT NOT NULL DEFAULT '',
    metadata_json TEXT,
    FOREIGN KEY(component_sha256) REFERENCES component_identities(component_sha256) ON DELETE CASCADE,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
    UNIQUE(component_sha256, asset_id, component_role, snapshot_version)
);

CREATE INDEX IF NOT EXISTS idx_component_sources_sha256 ON component_sources(component_sha256);
CREATE INDEX IF NOT EXISTS idx_component_sources_asset_id ON component_sources(asset_id);
CREATE INDEX IF NOT EXISTS idx_component_sources_role ON component_sources(component_role);
CREATE INDEX IF NOT EXISTS idx_component_sources_family ON component_sources(provider_family);
CREATE INDEX IF NOT EXISTS idx_component_sources_form ON component_sources(source_form);
CREATE INDEX IF NOT EXISTS idx_component_sources_availability ON component_sources(availability_state);

-- The following Phase 01 tables establish durable boundaries for later phases.
-- Their detailed behavior remains owned by the referenced future phase documents.
CREATE TABLE IF NOT EXISTS model_blueprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    blueprint_id TEXT NOT NULL UNIQUE,
    family_id TEXT NOT NULL,
    provider_version TEXT,
    source_asset_id INTEGER,
    source_file_sha256 TEXT,
    composition_sha256 TEXT,
    blueprint_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_asset_id) REFERENCES assets(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_model_blueprints_family ON model_blueprints(family_id);
CREATE INDEX IF NOT EXISTS idx_model_blueprints_source_asset ON model_blueprints(source_asset_id);
CREATE INDEX IF NOT EXISTS idx_model_blueprints_composition ON model_blueprints(composition_sha256);

CREATE TABLE IF NOT EXISTS saved_compositions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    composition_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    family_id TEXT NOT NULL,
    provider_version TEXT,
    composition_sha256 TEXT NOT NULL,
    composition_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_saved_compositions_family ON saved_compositions(family_id);
CREATE INDEX IF NOT EXISTS idx_saved_compositions_sha ON saved_compositions(composition_sha256);

CREATE TABLE IF NOT EXISTS component_policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_scope TEXT NOT NULL,
    base_component_sha256 TEXT,
    component_sha256 TEXT NOT NULL,
    component_role TEXT,
    policy_action TEXT NOT NULL,
    policy_source TEXT NOT NULL DEFAULT 'user',
    reason TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(policy_scope, base_component_sha256, component_sha256, component_role, policy_action)
);

CREATE INDEX IF NOT EXISTS idx_component_policies_component ON component_policies(component_sha256);
CREATE INDEX IF NOT EXISTS idx_component_policies_base ON component_policies(base_component_sha256);

CREATE TABLE IF NOT EXISTS component_validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id TEXT,
    provider_version TEXT,
    base_component_sha256 TEXT,
    component_sha256 TEXT NOT NULL,
    composition_sha256 TEXT,
    component_role TEXT,
    validation_state TEXT NOT NULL,
    validation_stage TEXT,
    validation_result TEXT,
    blocking_state TEXT NOT NULL DEFAULT 'advisory',
    evidence_type TEXT NOT NULL,
    evidence_json TEXT,
    environment_json TEXT,
    evidence_artifact TEXT,
    error_category TEXT,
    error_message TEXT,
    runtime_version TEXT,
    validated_at TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_component_validations_component ON component_validations(component_sha256);
CREATE INDEX IF NOT EXISTS idx_component_validations_base ON component_validations(base_component_sha256);
CREATE INDEX IF NOT EXISTS idx_component_validations_family ON component_validations(family_id);

-- Component decomposition is intentionally gated by evidence rather than hash identity
-- alone. This role-specific table is a soft splitter preflight: validated digital parity
-- can recommend a split, while callers may explicitly override the recommendation.
CREATE TABLE IF NOT EXISTS model_split_eligibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    model_sha256 TEXT,
    family_id TEXT,
    component_role TEXT NOT NULL,
    component_sha256 TEXT NOT NULL,
    eligibility_state TEXT NOT NULL,
    gate_mode TEXT NOT NULL DEFAULT 'recommended',
    digital_parity_status TEXT NOT NULL DEFAULT 'untested',
    parity_validation_id INTEGER,
    evidence_artifact TEXT,
    recommendation_reason TEXT,
    evidence_json TEXT,
    validated_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
    FOREIGN KEY(parity_validation_id) REFERENCES component_validations(id) ON DELETE SET NULL,
    UNIQUE(asset_id, component_role, component_sha256)
);

CREATE INDEX IF NOT EXISTS idx_model_split_eligibility_asset ON model_split_eligibility(asset_id);
CREATE INDEX IF NOT EXISTS idx_model_split_eligibility_component ON model_split_eligibility(component_sha256);
CREATE INDEX IF NOT EXISTS idx_model_split_eligibility_state ON model_split_eligibility(eligibility_state);
CREATE INDEX IF NOT EXISTS idx_model_split_eligibility_family ON model_split_eligibility(family_id);

-- Indexes for columns introduced by registry schema v7 are created by the v7
-- migration in AssetRegistry._migrate_component_validation_evidence(). Keeping
-- them out of unconditional bootstrap SQL allows pre-v7 databases to open long
-- enough for the migration to add those columns first.

CREATE TABLE IF NOT EXISTS registry_metrics (
    metric_key TEXT PRIMARY KEY,
    metric_value_json TEXT NOT NULL,
    calculated_at TEXT NOT NULL,
    calculation_version TEXT NOT NULL
);

-- ML-F01 reserves compact provider-defined analytical evidence. Individual tensor
-- hashes remain inside the manifest JSON rather than being normalized into millions
-- of database rows. ML-F02 owns population of these records.
CREATE TABLE IF NOT EXISTS component_analysis_manifests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_sha256 TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    component_role TEXT NOT NULL,
    layout_version INTEGER NOT NULL,
    algorithm_version TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(component_sha256) REFERENCES component_identities(component_sha256) ON DELETE CASCADE,
    UNIQUE(component_sha256, provider_id, component_role, layout_version, algorithm_version)
);

CREATE INDEX IF NOT EXISTS idx_component_analysis_component ON component_analysis_manifests(component_sha256);
CREATE INDEX IF NOT EXISTS idx_component_analysis_family ON component_analysis_manifests(family_id);
CREATE INDEX IF NOT EXISTS idx_component_analysis_manifest_sha ON component_analysis_manifests(manifest_sha256);

-- Generic analytical/provenance relationship home. Compatibility evidence remains
-- in component_validations; ML-F03/ML-F04 own relationship behavior and taxonomy.
CREATE TABLE IF NOT EXISTS component_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_component_sha256 TEXT NOT NULL,
    target_component_sha256 TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    evidence_kind TEXT NOT NULL,
    evidence_version TEXT NOT NULL,
    evidence_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_component_sha256) REFERENCES component_identities(component_sha256) ON DELETE CASCADE,
    FOREIGN KEY(target_component_sha256) REFERENCES component_identities(component_sha256) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_component_relationships_source ON component_relationships(source_component_sha256);
CREATE INDEX IF NOT EXISTS idx_component_relationships_target ON component_relationships(target_component_sha256);
CREATE INDEX IF NOT EXISTS idx_component_relationships_type ON component_relationships(relationship_type);

-- ML-F04 durable analytical/provenance relationship boundary. This table is
-- intentionally separate from component_validations and component_policies and
-- supports N-way relationships through normalized participants.
CREATE TABLE IF NOT EXISTS component_relationship_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_key TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    evidence_source TEXT NOT NULL,
    evidence_kind TEXT NOT NULL,
    evidence_version TEXT NOT NULL,
    provider_id TEXT,
    family_id TEXT,
    algorithm_id TEXT,
    algorithm_version TEXT,
    layout_version INTEGER,
    authoritative INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by_id INTEGER,
    evidence_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(superseded_by_id) REFERENCES component_relationship_evidence(id) ON DELETE SET NULL,
    UNIQUE(relationship_key, evidence_source, evidence_kind, evidence_version)
);

CREATE INDEX IF NOT EXISTS idx_relationship_evidence_key ON component_relationship_evidence(relationship_key);
CREATE INDEX IF NOT EXISTS idx_relationship_evidence_type ON component_relationship_evidence(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationship_evidence_source ON component_relationship_evidence(evidence_source);
CREATE INDEX IF NOT EXISTS idx_relationship_evidence_status ON component_relationship_evidence(status);

CREATE TABLE IF NOT EXISTS component_relationship_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_evidence_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    participant_role TEXT NOT NULL,
    component_sha256 TEXT,
    composition_id TEXT,
    blueprint_id TEXT,
    weight REAL,
    metadata_json TEXT,
    FOREIGN KEY(relationship_evidence_id) REFERENCES component_relationship_evidence(id) ON DELETE CASCADE,
    FOREIGN KEY(component_sha256) REFERENCES component_identities(component_sha256) ON DELETE CASCADE,
    UNIQUE(relationship_evidence_id, position)
);

CREATE INDEX IF NOT EXISTS idx_relationship_participants_component ON component_relationship_participants(component_sha256);
CREATE INDEX IF NOT EXISTS idx_relationship_participants_evidence ON component_relationship_participants(relationship_evidence_id);

CREATE TABLE IF NOT EXISTS registry_schema_meta (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS load_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    loaded_at TEXT NOT NULL,
    status TEXT NOT NULL,
    device TEXT,
    precision TEXT,
    load_time_ms INTEGER,
    error_message TEXT,
    context_json TEXT,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_load_history_asset_id ON load_history(asset_id);
CREATE INDEX IF NOT EXISTS idx_load_history_loaded_at ON load_history(loaded_at);

CREATE TABLE IF NOT EXISTS asset_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_asset_id INTEGER NOT NULL,
    target_asset_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence REAL,
    metadata_json TEXT,
    FOREIGN KEY(source_asset_id) REFERENCES assets(id) ON DELETE CASCADE,
    FOREIGN KEY(target_asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_asset_relationships_source ON asset_relationships(source_asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_relationships_target ON asset_relationships(target_asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_relationships_type ON asset_relationships(relationship_type);
"""

__all__ = ["REGISTRY_SCHEMA_VERSION", "SCHEMA_SQL"]
