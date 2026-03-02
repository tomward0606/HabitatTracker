window.HabitatCommon = (function () {
  async function postJSON(url, payload) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  function setLoading(on) {
    const el = document.getElementById("loading");
    if (!el) return;
    el.classList.toggle("is-on", on);
    el.setAttribute("aria-hidden", on ? "false" : "true");
  }

  function setText(id, msg) {
    const el = document.getElementById(id);
    if (el) el.textContent = msg;
  }

  function formatBBox(bbox, digits = 4) {
    return `${bbox.min_lon.toFixed(digits)}, ${bbox.min_lat.toFixed(digits)} → ${bbox.max_lon.toFixed(digits)}, ${bbox.max_lat.toFixed(digits)}`;
  }

  return { postJSON, setLoading, setText, formatBBox };
})();
