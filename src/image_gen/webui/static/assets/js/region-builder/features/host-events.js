/* IMAGE_GEN parent/opener messaging. */
function bindHostEvents() {
  window.addEventListener('message', (event) => {
    if (event.origin !== window.location.origin) return;
    const payload = event.data || {};
    if (payload.type === 'imagegen-region-builder-init') {
      imageGenTarget = String(payload.target || 'positive');
      applyBuilderDimensions(payload.width || 512, payload.height || 512, { rebuildPrompt: false });
      const current = String(payload.prompt || '');
      if (current.includes('REGION{')) {
        document.getElementById('promptText').value = current;
        importPrompt();
      } else {
        rebuild();
      }
      return;
    }
    if (payload.type === 'imagegen-region-builder-resync') {
      imageGenTarget = String(payload.target || imageGenTarget || 'positive');
      applyBuilderDimensions(payload.width || 512, payload.height || 512);
      toast(`Resolution synchronized to ${getRes().w} × ${getRes().h}; review and Apply again`);
    }
  });
}
