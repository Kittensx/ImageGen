/* Canvas viewport transform behavior. */
function updateViewTransform() {
  const vc = document.getElementById('viewContainer');
  if (vc) vc.style.transform = `translate(${viewPanX}px, ${viewPanY}px) scale(${viewZoom})`;
}

function resetView() {
  viewZoom = 1.0; viewPanX = 0; viewPanY = 0;
  updateViewTransform();
}
