// Local, offline color-name catalog based on the CSS Color Module named colors.
// Custom IMAGE_GEN palette names are listed first so exact theme presets retain
// their familiar names. Nearest matching uses OKLab distance for better
// perceptual results than raw RGB distance.
const COLOR_NAME_CATALOG = Object.freeze([
  Object.freeze({"name": "Sky Blue", "hex": "#179ee7"}),
  Object.freeze({"name": "Ocean", "hex": "#0284c7"}),
  Object.freeze({"name": "Teal", "hex": "#0d9488"}),
  Object.freeze({"name": "Emerald", "hex": "#16a34a"}),
  Object.freeze({"name": "Violet", "hex": "#7c3aed"}),
  Object.freeze({"name": "Rose", "hex": "#e11d48"}),
  Object.freeze({"name": "Amber", "hex": "#d97706"}),
  Object.freeze({"name": "Charcoal", "hex": "#111d29"}),
  Object.freeze({"name": "Midnight", "hex": "#111827"}),
  Object.freeze({"name": "Slate", "hex": "#1e293b"}),
  Object.freeze({"name": "Navy", "hex": "#101b33"}),
  Object.freeze({"name": "Espresso", "hex": "#251914"}),
  Object.freeze({"name": "Plum", "hex": "#241625"}),
  Object.freeze({"name": "Forest", "hex": "#12251c"}),
  Object.freeze({"name": "Alice Blue", "hex": "#f0f8ff"}),
  Object.freeze({"name": "Antique White", "hex": "#faebd7"}),
  Object.freeze({"name": "Aqua", "hex": "#00ffff"}),
  Object.freeze({"name": "Aquamarine", "hex": "#7fffd4"}),
  Object.freeze({"name": "Azure", "hex": "#f0ffff"}),
  Object.freeze({"name": "Beige", "hex": "#f5f5dc"}),
  Object.freeze({"name": "Bisque", "hex": "#ffe4c4"}),
  Object.freeze({"name": "Black", "hex": "#000000"}),
  Object.freeze({"name": "Blanched Almond", "hex": "#ffebcd"}),
  Object.freeze({"name": "Blue", "hex": "#0000ff"}),
  Object.freeze({"name": "Blue Violet", "hex": "#8a2be2"}),
  Object.freeze({"name": "Brown", "hex": "#a52a2a"}),
  Object.freeze({"name": "Burly Wood", "hex": "#deb887"}),
  Object.freeze({"name": "Cadet Blue", "hex": "#5f9ea0"}),
  Object.freeze({"name": "Chartreuse", "hex": "#7fff00"}),
  Object.freeze({"name": "Chocolate", "hex": "#d2691e"}),
  Object.freeze({"name": "Coral", "hex": "#ff7f50"}),
  Object.freeze({"name": "Cornflower Blue", "hex": "#6495ed"}),
  Object.freeze({"name": "Cornsilk", "hex": "#fff8dc"}),
  Object.freeze({"name": "Crimson", "hex": "#dc143c"}),
  Object.freeze({"name": "Dark Blue", "hex": "#00008b"}),
  Object.freeze({"name": "Dark Cyan", "hex": "#008b8b"}),
  Object.freeze({"name": "Dark Goldenrod", "hex": "#b8860b"}),
  Object.freeze({"name": "Dark Gray", "hex": "#a9a9a9"}),
  Object.freeze({"name": "Dark Green", "hex": "#006400"}),
  Object.freeze({"name": "Dark Khaki", "hex": "#bdb76b"}),
  Object.freeze({"name": "Dark Magenta", "hex": "#8b008b"}),
  Object.freeze({"name": "Dark Olive Green", "hex": "#556b2f"}),
  Object.freeze({"name": "Dark Orange", "hex": "#ff8c00"}),
  Object.freeze({"name": "Dark Orchid", "hex": "#9932cc"}),
  Object.freeze({"name": "Dark Red", "hex": "#8b0000"}),
  Object.freeze({"name": "Dark Salmon", "hex": "#e9967a"}),
  Object.freeze({"name": "Dark Sea Green", "hex": "#8fbc8f"}),
  Object.freeze({"name": "Dark Slate Blue", "hex": "#483d8b"}),
  Object.freeze({"name": "Dark Slate Gray", "hex": "#2f4f4f"}),
  Object.freeze({"name": "Dark Turquoise", "hex": "#00ced1"}),
  Object.freeze({"name": "Dark Violet", "hex": "#9400d3"}),
  Object.freeze({"name": "Deep Pink", "hex": "#ff1493"}),
  Object.freeze({"name": "Deep Sky Blue", "hex": "#00bfff"}),
  Object.freeze({"name": "Dim Gray", "hex": "#696969"}),
  Object.freeze({"name": "Dodger Blue", "hex": "#1e90ff"}),
  Object.freeze({"name": "Fire Brick", "hex": "#b22222"}),
  Object.freeze({"name": "Floral White", "hex": "#fffaf0"}),
  Object.freeze({"name": "Forest Green", "hex": "#228b22"}),
  Object.freeze({"name": "Fuchsia", "hex": "#ff00ff"}),
  Object.freeze({"name": "Gainsboro", "hex": "#dcdcdc"}),
  Object.freeze({"name": "Ghost White", "hex": "#f8f8ff"}),
  Object.freeze({"name": "Gold", "hex": "#ffd700"}),
  Object.freeze({"name": "Goldenrod", "hex": "#daa520"}),
  Object.freeze({"name": "Gray", "hex": "#808080"}),
  Object.freeze({"name": "Green", "hex": "#008000"}),
  Object.freeze({"name": "Green Yellow", "hex": "#adff2f"}),
  Object.freeze({"name": "Honeydew", "hex": "#f0fff0"}),
  Object.freeze({"name": "Hot Pink", "hex": "#ff69b4"}),
  Object.freeze({"name": "Indian Red", "hex": "#cd5c5c"}),
  Object.freeze({"name": "Indigo", "hex": "#4b0082"}),
  Object.freeze({"name": "Ivory", "hex": "#fffff0"}),
  Object.freeze({"name": "Khaki", "hex": "#f0e68c"}),
  Object.freeze({"name": "Lavender", "hex": "#e6e6fa"}),
  Object.freeze({"name": "Lavender Blush", "hex": "#fff0f5"}),
  Object.freeze({"name": "Lawn Green", "hex": "#7cfc00"}),
  Object.freeze({"name": "Lemon Chiffon", "hex": "#fffacd"}),
  Object.freeze({"name": "Light Blue", "hex": "#add8e6"}),
  Object.freeze({"name": "Light Coral", "hex": "#f08080"}),
  Object.freeze({"name": "Light Cyan", "hex": "#e0ffff"}),
  Object.freeze({"name": "Light Goldenrod Yellow", "hex": "#fafad2"}),
  Object.freeze({"name": "Light Gray", "hex": "#d3d3d3"}),
  Object.freeze({"name": "Light Green", "hex": "#90ee90"}),
  Object.freeze({"name": "Light Pink", "hex": "#ffb6c1"}),
  Object.freeze({"name": "Light Salmon", "hex": "#ffa07a"}),
  Object.freeze({"name": "Light Sea Green", "hex": "#20b2aa"}),
  Object.freeze({"name": "Light Sky Blue", "hex": "#87cefa"}),
  Object.freeze({"name": "Light Slate Gray", "hex": "#778899"}),
  Object.freeze({"name": "Light Steel Blue", "hex": "#b0c4de"}),
  Object.freeze({"name": "Light Yellow", "hex": "#ffffe0"}),
  Object.freeze({"name": "Lime", "hex": "#00ff00"}),
  Object.freeze({"name": "Lime Green", "hex": "#32cd32"}),
  Object.freeze({"name": "Linen", "hex": "#faf0e6"}),
  Object.freeze({"name": "Maroon", "hex": "#800000"}),
  Object.freeze({"name": "Medium Aquamarine", "hex": "#66cdaa"}),
  Object.freeze({"name": "Medium Blue", "hex": "#0000cd"}),
  Object.freeze({"name": "Medium Orchid", "hex": "#ba55d3"}),
  Object.freeze({"name": "Medium Purple", "hex": "#9370db"}),
  Object.freeze({"name": "Medium Sea Green", "hex": "#3cb371"}),
  Object.freeze({"name": "Medium Slate Blue", "hex": "#7b68ee"}),
  Object.freeze({"name": "Medium Spring Green", "hex": "#00fa9a"}),
  Object.freeze({"name": "Medium Turquoise", "hex": "#48d1cc"}),
  Object.freeze({"name": "Medium Violet Red", "hex": "#c71585"}),
  Object.freeze({"name": "Midnight Blue", "hex": "#191970"}),
  Object.freeze({"name": "Mint Cream", "hex": "#f5fffa"}),
  Object.freeze({"name": "Misty Rose", "hex": "#ffe4e1"}),
  Object.freeze({"name": "Moccasin", "hex": "#ffe4b5"}),
  Object.freeze({"name": "Navajo White", "hex": "#ffdead"}),
  Object.freeze({"name": "Navy", "hex": "#000080"}),
  Object.freeze({"name": "Old Lace", "hex": "#fdf5e6"}),
  Object.freeze({"name": "Olive", "hex": "#808000"}),
  Object.freeze({"name": "Olive Drab", "hex": "#6b8e23"}),
  Object.freeze({"name": "Orange", "hex": "#ffa500"}),
  Object.freeze({"name": "Orange Red", "hex": "#ff4500"}),
  Object.freeze({"name": "Orchid", "hex": "#da70d6"}),
  Object.freeze({"name": "Pale Goldenrod", "hex": "#eee8aa"}),
  Object.freeze({"name": "Pale Green", "hex": "#98fb98"}),
  Object.freeze({"name": "Pale Turquoise", "hex": "#afeeee"}),
  Object.freeze({"name": "Pale Violet Red", "hex": "#db7093"}),
  Object.freeze({"name": "Papaya Whip", "hex": "#ffefd5"}),
  Object.freeze({"name": "Peach Puff", "hex": "#ffdab9"}),
  Object.freeze({"name": "Peru", "hex": "#cd853f"}),
  Object.freeze({"name": "Pink", "hex": "#ffc0cb"}),
  Object.freeze({"name": "Plum", "hex": "#dda0dd"}),
  Object.freeze({"name": "Powder Blue", "hex": "#b0e0e6"}),
  Object.freeze({"name": "Purple", "hex": "#800080"}),
  Object.freeze({"name": "Rebecca Purple", "hex": "#663399"}),
  Object.freeze({"name": "Red", "hex": "#ff0000"}),
  Object.freeze({"name": "Rosy Brown", "hex": "#bc8f8f"}),
  Object.freeze({"name": "Royal Blue", "hex": "#4169e1"}),
  Object.freeze({"name": "Saddle Brown", "hex": "#8b4513"}),
  Object.freeze({"name": "Salmon", "hex": "#fa8072"}),
  Object.freeze({"name": "Sandy Brown", "hex": "#f4a460"}),
  Object.freeze({"name": "Sea Green", "hex": "#2e8b57"}),
  Object.freeze({"name": "Seashell", "hex": "#fff5ee"}),
  Object.freeze({"name": "Sienna", "hex": "#a0522d"}),
  Object.freeze({"name": "Silver", "hex": "#c0c0c0"}),
  Object.freeze({"name": "Sky Blue", "hex": "#87ceeb"}),
  Object.freeze({"name": "Slate Blue", "hex": "#6a5acd"}),
  Object.freeze({"name": "Slate Gray", "hex": "#708090"}),
  Object.freeze({"name": "Snow", "hex": "#fffafa"}),
  Object.freeze({"name": "Spring Green", "hex": "#00ff7f"}),
  Object.freeze({"name": "Steel Blue", "hex": "#4682b4"}),
  Object.freeze({"name": "Tan", "hex": "#d2b48c"}),
  Object.freeze({"name": "Teal", "hex": "#008080"}),
  Object.freeze({"name": "Thistle", "hex": "#d8bfd8"}),
  Object.freeze({"name": "Tomato", "hex": "#ff6347"}),
  Object.freeze({"name": "Turquoise", "hex": "#40e0d0"}),
  Object.freeze({"name": "Violet", "hex": "#ee82ee"}),
  Object.freeze({"name": "Wheat", "hex": "#f5deb3"}),
  Object.freeze({"name": "White", "hex": "#ffffff"}),
  Object.freeze({"name": "White Smoke", "hex": "#f5f5f5"}),
  Object.freeze({"name": "Yellow", "hex": "#ffff00"}),
  Object.freeze({"name": "Yellow Green", "hex": "#9acd32"}),
]);

function normalizeHex(value) {
  const text = String(value || "").trim();
  const short = /^#?([0-9a-f]{3})$/i.exec(text);
  if (short) return `#${short[1].split("").map((item) => item + item).join("")}`.toLowerCase();
  const full = /^#?([0-9a-f]{6})$/i.exec(text);
  return full ? `#${full[1].toLowerCase()}` : null;
}

function hexToRgb(hex) {
  const value = normalizeHex(hex);
  if (!value) return null;
  return [
    Number.parseInt(value.slice(1, 3), 16) / 255,
    Number.parseInt(value.slice(3, 5), 16) / 255,
    Number.parseInt(value.slice(5, 7), 16) / 255,
  ];
}

function srgbToLinear(channel) {
  return channel <= 0.04045
    ? channel / 12.92
    : ((channel + 0.055) / 1.055) ** 2.4;
}

function rgbToOklab(rgb) {
  const [r, g, b] = rgb.map(srgbToLinear);
  const l = (0.4122214708 * r) + (0.5363325363 * g) + (0.0514459929 * b);
  const m = (0.2119034982 * r) + (0.6806995451 * g) + (0.1073969566 * b);
  const s = (0.0883024619 * r) + (0.2817188376 * g) + (0.6299787005 * b);
  const lRoot = Math.cbrt(l);
  const mRoot = Math.cbrt(m);
  const sRoot = Math.cbrt(s);
  return [
    (0.2104542553 * lRoot) + (0.7936177850 * mRoot) - (0.0040720468 * sRoot),
    (1.9779984951 * lRoot) - (2.4285922050 * mRoot) + (0.4505937099 * sRoot),
    (0.0259040371 * lRoot) + (0.7827717662 * mRoot) - (0.8086757660 * sRoot),
  ];
}

const CATALOG_WITH_LAB = COLOR_NAME_CATALOG.map((entry) => ({
  ...entry,
  lab: rgbToOklab(hexToRgb(entry.hex)),
}));

export function nearestColorName(value) {
  const hex = normalizeHex(value);
  if (!hex) return "Custom Color";
  const target = rgbToOklab(hexToRgb(hex));
  let nearest = CATALOG_WITH_LAB[0];
  let nearestDistance = Number.POSITIVE_INFINITY;
  CATALOG_WITH_LAB.forEach((entry) => {
    const deltaL = target[0] - entry.lab[0];
    const deltaA = target[1] - entry.lab[1];
    const deltaB = target[2] - entry.lab[2];
    const distance = (deltaL * deltaL) + (deltaA * deltaA) + (deltaB * deltaB);
    if (distance < nearestDistance) {
      nearest = entry;
      nearestDistance = distance;
    }
  });
  return nearest.name;
}

export function colorNameCatalogSize() {
  return COLOR_NAME_CATALOG.length;
}
