/* Region Builder startup. Loaded last after every feature method is defined. */
function initRegionBuilder() {
  const query = new URLSearchParams(window.location.search);
  imageGenTarget = String(query.get('target') || imageGenTarget || 'positive');
  applyBuilderDimensions(query.get('width') || 512, query.get('height') || 512, { rebuildPrompt: false });
  regions = [];
  sel = -1;
  rebuild();
  renderGuides();
  updateEditor();
  bindHostEvents();
  bindInputEvents();
  setInteractionMode('select', { toastMessage: false });
  postToHost({ type: 'imagegen-region-builder-ready' });
  window.__REGION_BUILDER_BOOT_READY__ = true;
}

try {
  initRegionBuilder();
} catch (error) {
  console.error('Region Builder initialization failed:', error);
  if (typeof showRegionBuilderBootFailure === 'function') {
    showRegionBuilderBootFailure(error?.message || String(error));
  }
}
