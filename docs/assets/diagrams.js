/*
 * Renders ```mermaid blocks with the official Mermaid library (so the SVG sits
 * in the normal DOM, unlike Material's closed shadow root), follows the site's
 * light/dark scheme, and opens any diagram or chart in a pan/zoom viewer.
 */
(function () {
  const MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  let mermaidMod = null, counter = 0;

  const isDark = () => document.body.getAttribute("data-md-color-scheme") === "slate";

  async function mermaid() {
    if (!mermaidMod) mermaidMod = (await import(MERMAID)).default;
    mermaidMod.initialize({
      startOnLoad: false, securityLevel: "strict",
      theme: isDark() ? "dark" : "default",
      fontFamily: "var(--md-text-font-family), system-ui, sans-serif",
    });
    return mermaidMod;
  }

  const BOX = /[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF\u2190-\u21FF]/;   // box drawing, blocks, shapes, arrows
  function tagAsciiDiagrams() {
    for (const code of document.querySelectorAll(".md-typeset .highlight pre > code")) {
      const block = code.closest(".highlight");
      if (block.classList.contains("ascii-diagram") || block.className.match(/language-(?!text)/)) continue;
      const text = code.textContent;
      if ((text.match(new RegExp(BOX.source, "g")) || []).length >= 6) {
        block.classList.add("ascii-diagram");
        block.setAttribute("tabindex", "0");
        block.setAttribute("aria-label", "Text diagram. Press Enter to open a zoomable view.");
      }
    }
  }

  async function inlineCharts() {
    const imgs = document.querySelectorAll('.md-typeset img[src*="figures/"][src$=".svg"]');
    await Promise.all([...imgs].map(async (img) => {
      try {
        const text = await fetch(img.src).then((r) => r.text());
        const svg = new DOMParser().parseFromString(text, "image/svg+xml").documentElement;
        if (svg.nodeName !== "svg") return;
        svg.classList.add("chart");
        svg.setAttribute("role", "img");
        svg.setAttribute("aria-label", img.alt);
        svg.removeAttribute("width"); svg.removeAttribute("height");
        img.replaceWith(svg);
      } catch (e) { /* keep the <img>: it still works, just follows the OS theme */ }
    }));
  }

  async function renderAll() {
    tagAsciiDiagrams();
    inlineCharts();
    const sources = document.querySelectorAll("pre.diagram-source");
    if (!sources.length) return;
    const m = await mermaid();
    for (const pre of sources) {
      const code = pre.textContent;
      let host = pre.nextElementSibling;
      if (!host || !host.classList.contains("diagram")) {
        host = document.createElement("div");
        host.className = "diagram";
        host.setAttribute("role", "button");
        host.setAttribute("tabindex", "0");
        host.setAttribute("aria-label", "Diagram. Press Enter to open a zoomable view.");
        pre.after(host);
      }
      try {
        const { svg } = await m.render("dg-" + (++counter), code);
        host.innerHTML = svg;
      } catch (e) {
        host.textContent = "Diagram failed to render.";
        console.error(e);
      }
    }
  }

  /* ---------------- pan / zoom viewer ---------------- */
  function openViewer(node) {
    const overlay = document.createElement("div");
    overlay.className = "zoom-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.innerHTML =
      '<div class="zoom-bar"><span class="hint">Scroll or pinch to zoom · drag to pan · Esc to close</span>' +
      '<button data-a="out" aria-label="Zoom out">−</button>' +
      '<button data-a="in" aria-label="Zoom in">+</button>' +
      '<button data-a="fit">Fit</button>' +
      '<button data-a="close" aria-label="Close">Close ✕</button></div>' +
      '<div class="zoom-stage"><div class="zoom-content"></div></div>';
    const stage = overlay.querySelector(".zoom-stage");
    const content = overlay.querySelector(".zoom-content");
    const clone = node.cloneNode(true);
    clone.removeAttribute("style");
    clone.querySelectorAll("button, .md-clipboard").forEach((b) => b.remove());
    content.appendChild(clone);
    document.body.appendChild(overlay);
    document.documentElement.style.overflow = "hidden";

    // natural size of the drawing
    let w, h;
    if (clone.tagName === "IMG") { w = node.naturalWidth || node.width; h = node.naturalHeight || node.height; }
    else if (clone.tagName === "PRE") {
      clone.classList.add("zoom-pre");
      w = clone.scrollWidth; h = clone.scrollHeight;   // measured in the viewer's own font size
    }
    else {
      const vb = clone.viewBox && clone.viewBox.baseVal;
      w = vb && vb.width ? vb.width : node.getBoundingClientRect().width;
      h = vb && vb.height ? vb.height : node.getBoundingClientRect().height;
    }
    clone.style.width = w + "px"; clone.style.height = h + "px";

    let s = 1, x = 0, y = 0;
    const apply = () => { content.style.transform = `translate(${x}px,${y}px) scale(${s})`; };
    const fit = () => {
      const r = stage.getBoundingClientRect();
      s = Math.min((r.width - 40) / w, (r.height - 40) / h);
      x = (r.width - w * s) / 2; y = (r.height - h * s) / 2; apply();
    };
    const zoomAt = (factor, cx, cy) => {
      const ns = Math.min(Math.max(s * factor, 0.1), 20);
      x = cx - (cx - x) * (ns / s); y = cy - (cy - y) * (ns / s); s = ns; apply();
    };
    fit();

    stage.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = stage.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    const pts = new Map(); let last = null, lastDist = null;
    stage.addEventListener("pointerdown", (e) => {
      e.preventDefault();                          // no text selection while dragging
      stage.setPointerCapture(e.pointerId); pts.set(e.pointerId, e);
      stage.classList.add("dragging"); last = { x: e.clientX, y: e.clientY };
    });
    stage.addEventListener("pointermove", (e) => {
      if (!pts.has(e.pointerId)) return;
      pts.set(e.pointerId, e);
      if (pts.size === 2) {                         // pinch zoom
        const [a, b] = [...pts.values()];
        const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
        const r = stage.getBoundingClientRect();
        if (lastDist) zoomAt(d / lastDist, (a.clientX + b.clientX) / 2 - r.left, (a.clientY + b.clientY) / 2 - r.top);
        lastDist = d; return;
      }
      x += e.clientX - last.x; y += e.clientY - last.y; last = { x: e.clientX, y: e.clientY }; apply();
    });
    const up = (e) => { pts.delete(e.pointerId); lastDist = null; if (!pts.size) stage.classList.remove("dragging");
                        const p = [...pts.values()][0]; if (p) last = { x: p.clientX, y: p.clientY }; };
    stage.addEventListener("pointerup", up); stage.addEventListener("pointercancel", up);

    const close = () => {
      overlay.remove(); document.documentElement.style.overflow = "";
      document.removeEventListener("keydown", onKey); node.closest(".diagram, p")?.focus?.();
    };
    const onKey = (e) => {
      if (e.key === "Escape") close();
      else if (e.key === "+" || e.key === "=") zoomAt(1.25, stage.clientWidth / 2, stage.clientHeight / 2);
      else if (e.key === "-") zoomAt(0.8, stage.clientWidth / 2, stage.clientHeight / 2);
      else if (e.key === "0") fit();
    };
    document.addEventListener("keydown", onKey);
    overlay.querySelector(".zoom-bar").addEventListener("click", (e) => {
      const a = e.target.dataset.a; if (!a) return;
      const cx = stage.clientWidth / 2, cy = stage.clientHeight / 2;
      if (a === "in") zoomAt(1.25, cx, cy); else if (a === "out") zoomAt(0.8, cx, cy);
      else if (a === "fit") fit(); else close();
    });
    overlay.querySelector('[data-a="close"]').focus();
  }

  function target(e) {
    if (e.target.closest(".zoom-overlay")) return null;
    const d = e.target.closest(".md-typeset .diagram");
    if (d) return d.querySelector("svg");
    const img = e.target.closest('.md-typeset img[src*="figures/"], .md-typeset svg.chart');
    if (img) return img;
    const ascii = e.target.closest(".md-typeset .ascii-diagram");
    if (ascii && !e.target.closest("button, .md-clipboard")) return ascii.querySelector("pre");
    return null;
  }
  document.addEventListener("click", (e) => { const t = target(e); if (t) openViewer(t); });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const a = document.activeElement;
    const d = a && a.closest && a.closest(".diagram");
    if (d && d.querySelector("svg")) return openViewer(d.querySelector("svg"));
    const t = a && a.closest && a.closest(".ascii-diagram");
    if (t) openViewer(t.querySelector("pre"));
  });

  // Material's instant navigation swaps pages without reloading: hook into it.
  if (window.document$) document$.subscribe(renderAll); else document.addEventListener("DOMContentLoaded", renderAll);

  // Re-render in the matching theme when the reader toggles light/dark.
  new MutationObserver(() => { mermaidMod = null; renderAll(); })
    .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
})();
