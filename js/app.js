/* =============================================================
   app.js  —  engine (no need to edit this file)
   Reads SITE and SECTIONS from data.js and builds the page.
   ============================================================= */

(function () {
  "use strict";

  /* --- tiny helpers --- */
  const $  = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const esc = (t) =>
    String(t).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  const isEmpty = (v) => !v || String(v).trim() === "";

  const paragraphs = (text) =>
    String(text)
      .split(/\n\s*\n|\n/)
      .filter((p) => p.trim() !== "")
      .map((p) => `<p>${esc(p)}</p>`)
      .join("");

  const todo = (msg) => `<div class="todo"><span>${esc(msg)}</span></div>`;

  // render \( ... \) inline math (KaTeX) inside prose text
  const inlineMath = (s) => {
    let out = "", last = 0, m;
    const re = /\\\(([\s\S]+?)\\\)/g;
    while ((m = re.exec(s)) !== null) {
      out += esc(s.slice(last, m.index));
      out += `<span data-math="${esc(m[1])}"></span>`;
      last = re.lastIndex;
    }
    out += esc(s.slice(last));
    return out;
  };

  // prose split into paragraphs, with inline \(...\) math rendered
  const richProse = (text) =>
    String(text)
      .split(/\n\s*\n|\n/)
      .filter((p) => p.trim() !== "")
      .map((p) => `<p>${inlineMath(p)}</p>`)
      .join("");

  /* --- render math with KaTeX --- */
  function renderMath(root) {
    (root || document).querySelectorAll("[data-math]").forEach((el) => {
      if (el.dataset.mathDone) return;
      const tex = el.getAttribute("data-math");
      try {
        katex.render(tex, el, { throwOnError: false, displayMode: false });
        el.dataset.mathDone = "1";
      } catch (e) {
        el.textContent = tex;
      }
    });
  }

  /* --- fixed text from SITE --- */
  function fillSite() {
    $$("[data-field]").forEach((el) => {
      const key = el.getAttribute("data-field");
      if (SITE[key] !== undefined && !isEmpty(SITE[key])) {
        el.textContent = SITE[key];
      } else if (key === "introLead") {
        el.outerHTML = todo("introLead — write a short project introduction in data.js.");
      }
    });
    document.title = (SITE.heroTitle || "Functions") + " — MHF4U Photo Album";
  }

  /* =============================================================
     Hero background — animated function curves on a faint grid
     ============================================================= */
  function buildHeroCurves() {
    const W = 1200, H = 700;
    const xMin = -12, xMax = 12, yMin = -7, yMax = 7;
    const mapX = (x) => ((x - xMin) / (xMax - xMin)) * W;
    const mapY = (y) => H - ((y - yMin) / (yMax - yMin)) * H;

    // build an SVG path, breaking the line where the curve leaves the frame
    const pathFor = (fn, step) => {
      let d = "", pen = false;
      for (let x = xMin; x <= xMax; x += (step || 0.12)) {
        const y = fn(x);
        if (!isFinite(y) || y < yMin || y > yMax) { pen = false; continue; }
        const sx = mapX(x).toFixed(1), sy = mapY(y).toFixed(1);
        d += (pen ? "L" : "M") + sx + " " + sy + " ";
        pen = true;
      }
      return d.trim();
    };

    const curves = [
      { f: (x) => 4 * Math.sin(1.05 * x),        c: "#818cf8", w: 1.7 },
      { f: (x) => 0.016 * x * x * x,             c: "#a78bfa", w: 1.7 },
      { f: (x) => 4 - 0.075 * x * x,             c: "#6366f1", w: 1.6 },
      { f: (x) => 0.5 * Math.pow(1.55, x),       c: "#c4b5fd", w: 1.6 },
      { f: (x) => (x === 0 ? NaN : 4.6 / x),     c: "#a5b4fc", w: 1.5 },
      { f: (x) => (x > 0 ? 2.5 * Math.log(x) : NaN), c: "#8b5cf6", w: 1.5 },
    ];

    // faint grid lines
    let grid = "";
    for (let gx = xMin; gx <= xMax; gx += 2)
      grid += `<line x1="${mapX(gx)}" y1="0" x2="${mapX(gx)}" y2="${H}"/>`;
    for (let gy = yMin; gy <= yMax; gy += 2)
      grid += `<line x1="0" y1="${mapY(gy)}" x2="${W}" y2="${mapY(gy)}"/>`;

    const axes =
      `<line x1="0" y1="${mapY(0)}" x2="${W}" y2="${mapY(0)}"/>` +
      `<line x1="${mapX(0)}" y1="0" x2="${mapX(0)}" y2="${H}"/>`;

    const paths = curves
      .map(
        (c, i) =>
          `<path class="hero-curve" d="${pathFor(c.f)}" stroke="${c.c}" ` +
          `stroke-width="${c.w}" fill="none" stroke-linecap="round" ` +
          `style="filter: drop-shadow(0 0 8px ${c.c}88);" data-i="${i}"/>`
      )
      .join("");

    $("#heroCurves").innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid slice">
         <g stroke="rgba(255,255,255,0.05)" stroke-width="1">${grid}</g>
         <g stroke="rgba(255,255,255,0.12)" stroke-width="1.2">${axes}</g>
         ${paths}
       </svg>`;

    // draw-on animation
    $$(".hero-curve").forEach((p, i) => {
      const len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
      p.style.transition = "stroke-dashoffset 2.2s ease " + (i * 0.28) + "s";
      requestAnimationFrame(() =>
        requestAnimationFrame(() => { p.style.strokeDashoffset = "0"; })
      );
    });
  }

  /* --- navigation --- */
  function buildNav() {
    const list = $("#navList");
    list.innerHTML = SECTIONS.map(
      (s, i) => `<li><a href="#fn-${i + 1}" data-nav="${i + 1}">${esc(s.navLabel)}</a></li>`
    ).join("");

    const toggle = $("#navToggle");
    toggle.addEventListener("click", () => list.classList.toggle("is-open"));
    list.addEventListener("click", (e) => {
      if (e.target.tagName === "A") list.classList.remove("is-open");
    });
  }

  /* --- overview grid (9 cards) --- */
  function buildOverview() {
    $("#overviewGrid").innerHTML = SECTIONS.map((s, i) => {
      const n = String(i + 1).padStart(2, "0");
      const eq = isEmpty(s.parentEq)
        ? `<div class="ov-card__eq ov-card__eq--empty">Parent function — TBD</div>`
        : `<div class="ov-card__eq" data-math="${esc(s.parentEq)}"></div>`;
      return `<a class="ov-card" href="#fn-${i + 1}">
        <span class="ov-card__num">FUNCTION ${n}</span>
        <span class="ov-card__type">${esc(s.type)}</span>
        ${eq}
      </a>`;
    }).join("");
  }

  /* --- photo card --- */
  function photoCard(s) {
    if (isEmpty(s.image)) {
      return `<div class="media-card">
        <div class="media-card__label">Photograph</div>
        <div class="media-card__photo--empty"><span>No photo set yet</span>
          <span>add an image path to "image" in data.js</span></div>
      </div>`;
    }
    const alt = esc(s.imageAlt || s.title || "project photo");
    return `<div class="media-card">
      <div class="media-card__label">Photograph</div>
      <img class="media-card__photo" src="${esc(s.image)}" alt="${alt}"
        onerror="this.outerHTML='<div class=\\'media-card__photo--empty\\'><span>Image not found</span><span>${esc(
          s.image
        )}</span></div>'">
    </div>`;
  }

  /* --- Desmos graph card --- */
  function desmosCard(s) {
    const id = isEmpty(s.desmosId) ? "" : encodeURIComponent(s.desmosId);
    const link = id
      ? `<div class="media-card__foot">
          <a href="https://www.desmos.com/calculator/${id}" target="_blank" rel="noopener">
            Open this graph in Desmos &#8599;
          </a>
        </div>`
      : "";

    /* screenshot mode — a static image shown like the embed */
    if (!isEmpty(s.desmosImage)) {
      const shot = `<img class="media-card__shot" src="${esc(s.desmosImage)}"
        alt="Desmos graph screenshot"
        onerror="this.outerHTML='<div class=\\'media-card__frame--empty\\'><span>Screenshot not found</span><span>${esc(
          s.desmosImage
        )}</span></div>'">`;
      const visual = id
        ? `<a href="https://www.desmos.com/calculator/${id}" target="_blank" rel="noopener">${shot}</a>`
        : shot;
      return `<div class="media-card">
        <div class="media-card__label">Desmos Graph</div>
        ${visual}
        ${link}
      </div>`;
    }

    /* embed mode — interactive iframe */
    if (id) {
      return `<div class="media-card">
        <div class="media-card__label">Desmos Graph</div>
        <iframe class="media-card__frame"
          src="https://www.desmos.com/calculator/${id}?embed"
          allowfullscreen title="Desmos graph"></iframe>
        ${link}
      </div>`;
    }

    /* nothing added yet */
    return `<div class="media-card">
      <div class="media-card__label">Desmos Graph</div>
      <div class="media-card__frame--empty"><span>Desmos graph not added yet</span>
        <span>add "desmosId" or "desmosImage" in data.js</span></div>
    </div>`;
  }

  /* --- graph title + axis labels --- */
  function graphMeta(s) {
    const row = (key, val, hint) =>
      `<div class="graph-meta__row">
        <span class="graph-meta__key">${key}</span>
        <span>${
          isEmpty(val) ? `<span class="graph-meta__hint">${esc(hint)}</span>` : esc(val)
        }</span>
      </div>`;
    return `<div class="graph-meta">
      ${row("Title", s.graphTitle, "[dependent variable] as a function of [independent variable]")}
      ${row("x-axis", s.xLabel, "horizontal-axis label with units")}
      ${row("y-axis", s.yLabel, "vertical-axis label with units")}
    </div>`;
  }

  /* --- equations --- */
  function equations(s) {
    const parent = isEmpty(s.parentEq)
      ? todo("parentEq — add the parent function in data.js.")
      : `<div class="eq-card__math" data-math="${esc(s.parentEq)}"></div>`;
    const model = isEmpty(s.modelEq)
      ? todo("modelEq — build and add your transformed equation in data.js.")
      : `<div class="eq-card__math" data-math="${esc(s.modelEq)}"></div>`;
    return `<div class="equations">
      <div class="eq-card">
        <div class="eq-card__label">Parent Function</div>${parent}
      </div>
      <div class="eq-card eq-card--model">
        <div class="eq-card__label">Transformed Model</div>${model}
      </div>
    </div>`;
  }

  /* --- domain / range --- */
  function domainRange(s) {
    const item = (label, val) =>
      `<div class="dr-item">
        <div class="dr-item__label">${label}</div>
        ${
          isEmpty(val)
            ? `<div class="dr-item__value dr-item__value--empty">add notation in data.js</div>`
            : `<div class="dr-item__value" data-math="${esc(val)}"></div>`
        }
      </div>`;
    return `<div class="dr-row">${item("Domain", s.domain)}${item("Range", s.range)}</div>`;
  }

  /* --- one function section --- */
  function sectionHTML(s, i) {
    const n = i + 1;
    const nn = String(n).padStart(2, "0");

    const title = isEmpty(s.title)
      ? `<h2 class="fn__title" style="color:var(--accent)">[Add a section title in data.js]</h2>`
      : `<h2 class="fn__title">${esc(s.title)}</h2>`;
    const subtitle = isEmpty(s.subtitle)
      ? ""
      : `<p class="fn__subtitle">${esc(s.subtitle)}</p>`;

    const context = isEmpty(s.context)
      ? todo("context — add the personal story and real-world meaning of this photo (data.js).")
      : `<div class="fn__context">${richProse(s.context)}</div>`;

    const transforms = `<div class="prose-block">
      <div class="prose-block__head">Transformations</div>
      ${
        isEmpty(s.transformsText)
          ? todo("transformsText — describe, in your own words, every transformation applied to the parent function (data.js).")
          : richProse(s.transformsText)
      }
    </div>`;

    const dr = `<div class="prose-block">
      <div class="prose-block__head">Domain &amp; Range</div>
      ${domainRange(s)}
      ${
        isEmpty(s.domainRangeText)
          ? todo("domainRangeText — state the domain and range and explain, in your own words, why they are restricted (data.js).")
          : richProse(s.domainRangeText)
      }
    </div>`;

    return `<section class="fn reveal" id="fn-${n}">
      <div class="fn__ghost">${nn}</div>
      <div class="fn__inner">
        <div class="fn__head">
          <span class="fn__num">${nn}</span>
          <span class="fn__type">${esc(s.type)}</span>
        </div>
        ${title}
        ${subtitle}
        ${context}
        <div class="fn__media">${photoCard(s)}${desmosCard(s)}</div>
        ${graphMeta(s)}
        ${equations(s)}
        ${transforms}
        ${dr}
      </div>
    </section>`;
  }

  /* --- scroll behaviours: progress bar, reveal, active nav --- */
  function setupScroll() {
    const bar = $("#progress");
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
    };
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    // reveal on scroll
    const revIO = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            revIO.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    $$(".reveal").forEach((el) => revIO.observe(el));

    // active nav link
    const navLinks = {};
    $$("[data-nav]").forEach((a) => (navLinks[a.dataset.nav] = a));
    const actIO = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          const id = e.target.id.replace("fn-", "");
          if (e.isIntersecting) {
            Object.values(navLinks).forEach((a) => a.classList.remove("is-active"));
            if (navLinks[id]) navLinks[id].classList.add("is-active");
          }
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    $$(".fn").forEach((el) => actIO.observe(el));
  }

  /* --- Hero section interactive effects --- */
  function setupHeroEffects() {
    const hero = $("#top");
    if (!hero) return;
    
    hero.addEventListener("mousemove", (e) => {
      const rect = hero.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      hero.style.setProperty("--mouse-x", `${x}px`);
      hero.style.setProperty("--mouse-y", `${y}px`);
    });
  }

  /* --- build everything --- */
  function build() {
    fillSite();
    buildNav();
    buildOverview();
    $("#sections").innerHTML = SECTIONS.map(sectionHTML).join("");
    renderMath(document);
    buildHeroCurves();
    setupScroll();
    setupHeroEffects();
  }

  document.addEventListener("DOMContentLoaded", build);
})();
