const DEFAULT_BATCH_SIZE = 50;

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

export function createProgressiveAssetGrid({
  grid,
  createCard,
  batchSize = DEFAULT_BATCH_SIZE,
  onProgress = null,
  getItemKey = (item) => item?.asset_id,
} = {}) {
  if (!grid || typeof createCard !== "function") {
    return {
      setItems() {},
      refresh() {},
      refreshItem() { return false; },
      loadNext() {},
      reset() {},
      destroy() {},
      get renderedCount() { return 0; },
      get totalCount() { return 0; },
    };
  }

  const normalizedBatchSize = Math.max(1, Number(batchSize) || DEFAULT_BATCH_SIZE);
  let items = [];
  let renderedCount = 0;
  let observer = null;
  let fallbackListener = null;

  const sentinel = document.createElement("div");
  sentinel.className = "asset-grid-load-sentinel";
  sentinel.setAttribute("aria-hidden", "true");

  const emitProgress = () => {
    if (typeof onProgress !== "function") return;
    onProgress({
      shown: renderedCount,
      total: items.length,
      hasMore: renderedCount < items.length,
      batchSize: normalizedBatchSize,
    });
  };

  const syncSentinel = () => {
    sentinel.classList.toggle("is-hidden", renderedCount >= items.length);
    if (!sentinel.isConnected) grid.append(sentinel);
  };

  const appendRange = (targetCount) => {
    const nextCount = Math.min(items.length, Math.max(renderedCount, targetCount));
    if (nextCount <= renderedCount) {
      syncSentinel();
      emitProgress();
      return;
    }

    const fragment = document.createDocumentFragment();
    for (let index = renderedCount; index < nextCount; index += 1) {
      fragment.append(createCard(items[index], index));
    }
    grid.insertBefore(fragment, sentinel);
    renderedCount = nextCount;
    syncSentinel();
    emitProgress();
  };

  const loadNext = () => {
    appendRange(renderedCount + normalizedBatchSize);
  };

  const renderWindow = (targetCount, { preserveScroll = false } = {}) => {
    const previousScrollTop = grid.scrollTop;
    grid.replaceChildren(sentinel);
    renderedCount = 0;
    appendRange(targetCount);
    if (preserveScroll) {
      requestAnimationFrame(() => {
        grid.scrollTop = Math.min(previousScrollTop, Math.max(0, grid.scrollHeight - grid.clientHeight));
      });
    } else {
      grid.scrollTop = 0;
    }
  };

  const setItems = (nextItems, { reset = true } = {}) => {
    const previousRenderedCount = renderedCount;
    items = safeArray(nextItems);
    const targetCount = reset
      ? Math.min(normalizedBatchSize, items.length)
      : Math.min(items.length, Math.max(normalizedBatchSize, previousRenderedCount));
    renderWindow(targetCount, { preserveScroll: !reset });
  };

  const refresh = (nextItems = items) => {
    setItems(nextItems, { reset: false });
  };

  const refreshItem = (key, nextItem = null) => {
    if (typeof getItemKey !== "function") return false;
    const normalizedKey = String(key ?? "");
    if (!normalizedKey) return false;
    const index = items.findIndex((item, itemIndex) => String(getItemKey(item, itemIndex) ?? "") === normalizedKey);
    if (index < 0) return false;
    if (nextItem != null) items[index] = nextItem;

    // Items beyond the progressive render window will use the updated record
    // when they are eventually created, so no DOM work is needed yet.
    if (index >= renderedCount) return true;
    const current = grid.children[index];
    if (!current || current === sentinel) return false;
    current.replaceWith(createCard(items[index], index));
    emitProgress();
    return true;
  };

  const reset = () => {
    setItems(items, { reset: true });
  };

  grid.append(sentinel);

  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      if (renderedCount >= items.length) return;
      loadNext();
    }, {
      root: null,
      rootMargin: "320px 0px",
      threshold: 0.01,
    });
    observer.observe(sentinel);
  } else {
    fallbackListener = () => {
      if (renderedCount >= items.length) return;
      const rect = sentinel.getBoundingClientRect();
      if (rect.top <= window.innerHeight + 320) loadNext();
    };
    grid.addEventListener("scroll", fallbackListener, { passive: true });
    window.addEventListener("scroll", fallbackListener, { passive: true });
  }

  emitProgress();

  return {
    setItems,
    refresh,
    refreshItem,
    loadNext,
    reset,
    destroy() {
      observer?.disconnect();
      if (fallbackListener) {
        grid.removeEventListener("scroll", fallbackListener);
        window.removeEventListener("scroll", fallbackListener);
      }
    },
    get renderedCount() {
      return renderedCount;
    },
    get totalCount() {
      return items.length;
    },
  };
}

export { DEFAULT_BATCH_SIZE };
