export async function loadFragments(root = document) {
  const placeholders = [...root.querySelectorAll("[data-fragment]")];
  await Promise.all(placeholders.map(async (placeholder) => {
    const response = await fetch(placeholder.dataset.fragment, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" },
    });
    if (!response.ok) {
      throw new Error(`Unable to load UI fragment: ${placeholder.dataset.fragment}`);
    }
    const template = document.createElement("template");
    template.innerHTML = await response.text();
    placeholder.replaceWith(template.content);
  }));
}
