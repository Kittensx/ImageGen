from __future__ import annotations

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

    asset_type TEXT NOT NULL DEFAULT 'unknown',
    format_type TEXT NOT NULL DEFAULT 'other',
    architecture TEXT NOT NULL DEFAULT 'unknown',

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
    notes TEXT
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