/* MLAtoms — theme, search, mermaid, copy buttons, table of contents. */
(() => {
  "use strict";

  const root = document.body.dataset.root || "./";
  const docEl = document.documentElement;
  const THEME_KEY = "mlatoms-theme";

  const ICON_SUN =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.2 5.2l1.6 1.6M17.2 17.2l1.6 1.6M18.8 5.2l-1.6 1.6M6.8 17.2l-1.6 1.6"/></svg>';
  const ICON_MOON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.5 14.2A8.6 8.6 0 0 1 9.8 3.5a8.6 8.6 0 1 0 10.7 10.7Z"/></svg>';

  const prefersDark = () =>
    window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

  const currentTheme = () => docEl.dataset.theme || (prefersDark() ? "dark" : "light");

  /* -------------------------------------------------------------- theme -- */

  const themeBtn = document.querySelector("[data-theme-toggle]");

  function paintThemeButton() {
    if (!themeBtn) return;
    const dark = currentTheme() === "dark";
    const icon = themeBtn.querySelector("span[aria-hidden]");
    if (icon) icon.innerHTML = dark ? ICON_MOON : ICON_SUN;
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    themeBtn.setAttribute("aria-pressed", String(dark));
  }

  function setTheme(theme) {
    docEl.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (err) {
      /* private mode: stay in-session only */
    }
    paintThemeButton();
    paintMermaid();
  }

  paintThemeButton();
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      setTheme(currentTheme() === "dark" ? "light" : "dark");
    });
  }

  /* -------------------------------------------------------------- copy -- */

  const copyLabel = { idle: "Copy", done: "Copied" };

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise((resolve, reject) => {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy") ? resolve() : reject(new Error("copy failed"));
      } catch (err) {
        reject(err);
      } finally {
        document.body.removeChild(area);
      }
    });
  }

  function addCopyButtons() {
    document.querySelectorAll("pre.highlight").forEach((pre) => {
      if (pre.classList.contains("mermaid-source") || pre.dataset.copy === "off") return;
      const code = pre.querySelector("code");
      if (!code) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "copy-btn";
      button.textContent = copyLabel.idle;
      button.setAttribute("aria-label", "Copy this code block");
      button.addEventListener("click", () => {
        copyText(code.innerText).then(
          () => {
            button.textContent = copyLabel.done;
            button.classList.add("is-done");
            window.setTimeout(() => {
              button.textContent = copyLabel.idle;
              button.classList.remove("is-done");
            }, 1400);
          },
          () => {
            button.textContent = "Failed";
            window.setTimeout(() => {
              button.textContent = copyLabel.idle;
            }, 1400);
          }
        );
      });
      pre.appendChild(button);
    });
  }

  /* ----------------------------------------------------------- mermaid -- */

  const diagrams = [];
  let mermaidApi = null;
  let mermaidFailed = false;

  async function paintMermaid() {
    const dark = currentTheme() === "dark";
    if (mermaidApi) {
      mermaidApi.initialize(mermaidConfig(mermaidApi, dark));
    }
    diagrams.forEach((diagram) => {
      if (diagram.rendered) diagram.host.remove();
      const host = document.createElement("div");
      host.className = "mermaid";
      host.textContent = diagram.code;
      diagram.host = host;
      diagram.rendered = false;
      diagram.source.after(host);
    });
    if (!mermaidApi) return;
    for (const diagram of diagrams) {
      try {
        await mermaidApi.run({ nodes: [diagram.host] });
        diagram.rendered = true;
        diagram.source.hidden = true;
      } catch (err) {
        diagram.source.hidden = false;
        diagram.host.remove();
        diagram.rendered = false;
      }
    }
  }

  function mermaidConfig(api, dark) {
    const palette = dark
      ? {
          background: "#14161a",
          primaryColor: "#1b1e23",
          primaryTextColor: "#ecebe6",
          primaryBorderColor: "#34383f",
          lineColor: "#83868e",
          secondaryColor: "#1b1e23",
          tertiaryColor: "#14161a",
          noteBkgColor: "#1b1e23",
          noteTextColor: "#ecebe6",
          noteBorderColor: "#34383f",
          textColor: "#ecebe6",
        }
      : {
          background: "#f4f2ec",
          primaryColor: "#f4f2ec",
          primaryTextColor: "#161614",
          primaryBorderColor: "#d2cec1",
          lineColor: "#7c7a72",
          secondaryColor: "#f4f2ec",
          tertiaryColor: "#fbfaf7",
          noteBkgColor: "#ece9e0",
          noteTextColor: "#161614",
          noteBorderColor: "#d2cec1",
          textColor: "#161614",
        };
    return {
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      darkMode: dark,
      fontFamily:
        '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif',
      themeVariables: Object.assign(palette, { fontSize: "15px" }),
    };
  }

  async function initMermaid() {
    const sources = Array.from(document.querySelectorAll("pre.mermaid-source"));
    if (!sources.length) return;
    sources.forEach((source) => {
      diagrams.push({ source, code: (source.textContent || "").trim(), host: null, rendered: false });
    });
    try {
      const module = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
      mermaidApi = module.default || module;
      mermaidApi.initialize(mermaidConfig(mermaidApi, currentTheme() === "dark"));
      await paintMermaid();
    } catch (err) {
      mermaidFailed = true;
      diagrams.forEach((diagram) => {
        diagram.source.hidden = false;
        if (diagram.host) diagram.host.remove();
        diagram.host = null;
        diagram.rendered = false;
      });
    }
  }

  /* ------------------------------------------------------------ search -- */

  const dialog = document.querySelector("dialog.search");
  const form = dialog && dialog.querySelector(".search__form");
  const input = dialog && dialog.querySelector("input[type=search]");
  const list = dialog && dialog.querySelector(".search__results");
  const status = dialog && dialog.querySelector(".search__status");
  const HINT = status ? status.textContent : "";

  let index = null;
  let indexPromise = null;
  let active = -1;
  let entries = [];
  let renderToken = 0;

  function loadIndex() {
    if (index) return Promise.resolve(index);
    if (!indexPromise) {
      indexPromise = fetch(root + "search-index.json")
        .then((response) => (response.ok ? response.json() : []))
        .then((data) => {
          index = Array.isArray(data) ? data : [];
          return index;
        })
        .catch(() => {
          index = [];
          return index;
        });
    }
    return indexPromise;
  }

  const normalize = (text) => (text || "").toLowerCase();

  function countOccurrences(haystack, needle) {
    let count = 0;
    let at = haystack.indexOf(needle);
    while (at !== -1) {
      count += 1;
      at = haystack.indexOf(needle, at + needle.length);
    }
    return count;
  }

  function scoreEntry(entry, terms) {
    const title = normalize(entry.t);
    const body = normalize(entry.b);
    const desc = normalize(entry.d);
    const variants = normalize(entry.v);
    let total = 0;
    for (const term of terms) {
      let score = 0;
      if (title.includes(term)) score += title.startsWith(term) ? 16 : 10;
      if (variants.includes(term)) score += 7;
      if (desc.includes(term)) score += 4;
      const hits = countOccurrences(body, term);
      if (hits) score += Math.min(hits, 5);
      if (!score) return 0;
      total += score;
    }
    return total * (entry.w || 0.7);
  }

  function snippet(entry, terms) {
    const body = entry.b || "";
    const text = terms.some((term) => body.toLowerCase().includes(term)) ? body : entry.d || body;
    const lower = text.toLowerCase();
    let at = -1;
    for (const term of terms) {
      const found = lower.indexOf(term);
      if (found !== -1 && (at === -1 || found < at)) at = found;
    }
    if (at === -1) return text.slice(0, 132) + (text.length > 132 ? "…" : "");
    const start = Math.max(0, at - 48);
    const end = Math.min(text.length, at + 104);
    return (start > 0 ? "…" : "") + text.slice(start, end).trim() + (end < text.length ? "…" : "");
  }

  function markTerms(container, text, terms) {
    const lower = text.toLowerCase();
    let cursor = 0;
    const hits = [];
    terms.forEach((term) => {
      let at = lower.indexOf(term);
      while (at !== -1) {
        hits.push([at, at + term.length]);
        at = lower.indexOf(term, at + term.length);
      }
    });
    hits.sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const hit of hits) {
      const last = merged[merged.length - 1];
      if (last && hit[0] <= last[1]) last[1] = Math.max(last[1], hit[1]);
      else merged.push([hit[0], hit[1]]);
    }
    for (const [from, to] of merged) {
      if (from > cursor) container.appendChild(document.createTextNode(text.slice(cursor, from)));
      const mark = document.createElement("mark");
      mark.textContent = text.slice(from, to);
      container.appendChild(mark);
      cursor = to;
    }
    if (cursor < text.length) container.appendChild(document.createTextNode(text.slice(cursor)));
  }

  function setActive(next) {
    const links = list ? Array.from(list.querySelectorAll("a")) : [];
    if (!links.length) return;
    active = (next + links.length) % links.length;
    links.forEach((link, i) => link.classList.toggle("is-active", i === active));
    links[active].scrollIntoView({ block: "nearest" });
  }

  function renderResults(results, terms) {
    if (!list || !status) return;
    list.textContent = "";
    entries = results;
    active = -1;
    if (!results.length) {
      status.textContent = "No matches. Try an algorithm name, a variant, or a chapter.";
      return;
    }
    status.textContent = `${results.length} result${results.length === 1 ? "" : "s"} — ↑ ↓ to move, ↵ to open.`;
    results.forEach((entry) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = root + entry.u;

      const title = document.createElement("b");
      markTerms(title, entry.t, terms);
      const kind = document.createElement("small");
      kind.textContent = entry.k;
      const text = document.createElement("span");
      markTerms(text, snippet(entry, terms), terms);

      link.append(title, kind, text);
      item.appendChild(link);
      list.appendChild(item);
    });
  }

  async function search(query) {
    const terms = normalize(query).split(/\s+/).filter(Boolean);
    const token = ++renderToken;
    if (!terms.length) {
      if (list) list.textContent = "";
      if (status) status.textContent = HINT;
      entries = [];
      return;
    }
    const data = await loadIndex();
    if (token !== renderToken) return;
    const scored = [];
    for (const entry of data) {
      const score = scoreEntry(entry, terms);
      if (score > 0) scored.push([score, entry]);
    }
    scored.sort((a, b) => b[0] - a[0] || a[1].t.localeCompare(b[1].t));
    renderResults(
      scored.slice(0, 9).map(([, entry]) => entry),
      terms
    );
  }

  function openSearch() {
    if (!dialog || dialog.open) return;
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    if (input) {
      input.focus();
      input.select();
    }
    if (input && input.value.trim()) search(input.value);
  }

  if (dialog && input) {
    document.querySelectorAll("[data-search-open]").forEach((button) => {
      button.addEventListener("click", openSearch);
    });

    let timer = 0;
    input.addEventListener("input", () => {
      window.clearTimeout(timer);
      const value = input.value;
      timer = window.setTimeout(() => search(value), 90);
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActive(active + 1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActive(active - 1);
      } else if (event.key === "Enter") {
        const link = list && list.querySelector("a.is-active");
        if (link) {
          event.preventDefault();
          window.location.href = link.href;
        }
      }
    });

    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });

    dialog.addEventListener("close", () => {
      if (list) list.textContent = "";
      entries = [];
      active = -1;
    });

    if (form) form.addEventListener("submit", (event) => event.preventDefault());
  }

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const typing =
      target &&
      (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
    if ((event.key === "k" || event.key === "K") && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      openSearch();
      return;
    }
    if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      openSearch();
    }
  });

  /* ----------------------------------------------------------------- toc -- */

  function initToc() {
    const links = Array.from(document.querySelectorAll(".rail-toc a[href^='#']"));
    if (!links.length) return;
    const targets = links
      .map((link) => {
        const id = decodeURIComponent(link.getAttribute("href").slice(1));
        const heading = document.getElementById(id);
        return heading ? { link, heading } : null;
      })
      .filter(Boolean);
    if (!targets.length) return;

    let frame = 0;
    const update = () => {
      frame = 0;
      const line = 110;
      let current = targets[0];
      for (const target of targets) {
        if (target.heading.getBoundingClientRect().top <= line) current = target;
      }
      links.forEach((link) => link.classList.toggle("is-active", link === current.link));
    };

    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  }

  /* ---------------------------------------------------------------- init -- */

  addCopyButtons();
  initToc();
  initMermaid();
})();
