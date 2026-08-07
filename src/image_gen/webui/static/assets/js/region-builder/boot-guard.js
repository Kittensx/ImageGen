/* Visible startup diagnostics for the Region Builder runtime. */
window.__REGION_BUILDER_BOOT_READY__ = false;

function showRegionBuilderBootFailure(message) {
  if (window.__REGION_BUILDER_BOOT_READY__) return;
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = `Region Builder startup failed: ${message || 'check the browser console'}`;
  toast.classList.add('show', 'boot-error');
}

window.addEventListener('error', (event) => {
  const source = String(event.filename || '');
  if (source.includes('/assets/js/region-builder/')) {
    showRegionBuilderBootFailure(event.message || `Unable to load ${source}`);
  }
});

window.addEventListener('unhandledrejection', (event) => {
  showRegionBuilderBootFailure(event.reason?.message || String(event.reason || 'Unhandled startup error'));
});

window.setTimeout(() => {
  if (!window.__REGION_BUILDER_BOOT_READY__) {
    showRegionBuilderBootFailure('runtime did not finish initializing');
  }
}, 5000);
