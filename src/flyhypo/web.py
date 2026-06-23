"""Minimal local web UI for flyhypo.

A single-file, stdlib-only server (no web framework dependency). It serves one
HTML page and a /api/run endpoint that drives the existing pipeline modules and
returns the structured JSON, which the page renders.

    uv run flyhypo-web            # then open http://127.0.0.1:8000

Note: this is a *local* tool — it runs the same neuPrint + Gemini pipeline as the
CLI, so it needs the same .env tokens. Web UI was an out-of-scope TODO in the
original spec; this is a thin convenience layer over the CLI, not a product.
"""

from __future__ import annotations

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import connectome, literature, synthesize
from .cli import _slug, render_markdown
from .schema import Hypothesis

OUTPUT_DIR = Path("outputs")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flyhypo</title>
<style>
  :root {
    --bg:#0f1117; --panel:#171a23; --line:#272b38; --ink:#e7e9ee;
    --muted:#9aa3b2; --accent:#7cc4ff; --accent2:#b69cff;
    --hi:#34d399; --med:#fbbf24; --lo:#fb923c; --spec:#94a3b8;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:22px 28px; border-bottom:1px solid var(--line);
    display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { margin:0; font-size:20px; letter-spacing:.5px; }
  h1 .fly { color:var(--accent); }
  header .tag { color:var(--muted); font-size:13px; }
  main { max-width:980px; margin:0 auto; padding:24px 20px 80px; }
  form { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end;
    background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
  label { display:flex; flex-direction:column; gap:5px; font-size:12px; color:var(--muted); }
  input, select { background:#0d0f15; color:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:9px 11px; font-size:14px; }
  input[type=text] { width:180px; }
  button { background:var(--accent); color:#06121f; border:0; border-radius:8px;
    padding:10px 18px; font-weight:600; font-size:14px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .banner { margin:18px 0; padding:11px 14px; border-radius:10px; font-size:13px;
    background:#1a1505; border:1px solid #4a3a08; color:#f4dca0; }
  .status { margin:18px 0; color:var(--muted); }
  .err { background:#2a0f12; border:1px solid #6b1d24; color:#ffb4ba;
    padding:12px 14px; border-radius:10px; white-space:pre-wrap; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:18px 20px; margin:16px 0; }
  section h2 { margin:0 0 12px; font-size:15px; color:var(--accent2);
    text-transform:uppercase; letter-spacing:.6px; }
  .kv { color:var(--muted); }
  .kv b { color:var(--ink); font-weight:600; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; color:var(--accent); }
  .chips span { display:inline-block; background:#0d0f15; border:1px solid var(--line);
    border-radius:20px; padding:3px 10px; margin:3px 4px 0 0; font-size:12px; }
  .hyp { border:1px solid var(--line); border-radius:10px; padding:14px; margin:12px 0;
    background:#0d0f15; }
  .badge { font-size:11px; font-weight:700; text-transform:uppercase; padding:2px 9px;
    border-radius:20px; letter-spacing:.5px; }
  .b-high{background:#042a1c;color:var(--hi);border:1px solid #0c5}
  .b-medium{background:#2a2305;color:var(--med);border:1px solid #a80}
  .b-low{background:#2a1505;color:var(--lo);border:1px solid #b60}
  .b-speculative{background:#1b2129;color:var(--spec);border:1px solid #455}
  .hyp h3 { margin:0 0 6px; font-size:14px; display:flex; gap:10px; align-items:center; }
  .hyp .rat { color:var(--muted); font-size:13px; margin:8px 0; }
  ul.grounded { margin:6px 0; padding-left:18px; font-size:12px; color:var(--muted); }
  ul.grounded code { color:var(--accent); }
  .lit a { color:var(--accent); text-decoration:none; }
  .lit li { margin-bottom:8px; font-size:13px; }
  .lit .why { color:var(--muted); font-size:12px; }
  .muted { color:var(--muted); }
  .suggest span { cursor:pointer; }
  .suggest span:hover { border-color:var(--accent); color:var(--accent); }
  .history { margin:14px 0 4px; font-size:13px; color:var(--muted); }
  .history .chip { display:inline-block; background:#0d0f15; border:1px solid var(--line);
    border-radius:20px; padding:4px 12px; margin:4px 6px 0 0; cursor:pointer; color:var(--ink); }
  .history .chip:hover { border-color:var(--accent); }
  .history .chip .lab { cursor:pointer; }
  .history .chip .lab:hover { color:var(--accent); }
  .history .chip .x { margin-left:9px; color:var(--muted); cursor:pointer; }
  .history .chip .x:hover { color:#ff8b91; }
  .toolbar { display:flex; gap:8px; justify-content:flex-end; margin-bottom:6px; }
  .toolbar button { background:#0d0f15; color:var(--accent); border:1px solid var(--line);
    font-weight:600; padding:7px 14px; }
  .toolbar button:hover { border-color:var(--accent); }
  footer { text-align:center; color:var(--muted); font-size:12px; padding:10px; }
</style>
</head>
<body>
<header>
  <h1><span class="fly">fly</span>hypo</h1>
  <span class="tag">Drosophila neuron functional-hypothesis generator · connectome + literature + LLM</span>
</header>
<main>
  <form id="f">
    <label>Cell type
      <input type="text" id="cell" value="EPG" autocomplete="off" required>
    </label>
    <label>Dataset
      <input type="text" id="dataset" value="hemibrain:v1.2.1">
    </label>
    <label>Top-K partners
      <input type="number" id="topk" value="15" min="1" max="50" style="width:80px">
    </label>
    <label>Mode
      <select id="mode">
        <option value="full">Full (structure + literature + LLM)</option>
        <option value="fingerprint">Fingerprint only (neuPrint)</option>
      </select>
    </label>
    <button type="submit" id="go">Generate</button>
  </form>

  <div id="history" class="history"></div>

  <div class="banner">A connectome gives connectivity — not synapse sign, effective
    strength, or neuromodulation. Output is a <b>hypothesis for experimentalists</b>,
    never a stated fact.</div>

  <div id="status"></div>
  <div id="out"></div>
</main>
<footer>local PoC · same .env tokens as the CLI</footer>

<script>
const $ = (s) => document.querySelector(s);
const el = (t, c, txt) => { const e=document.createElement(t); if(c)e.className=c; if(txt!=null)e.textContent=txt; return e; };

$("#f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const cell = $("#cell").value.trim();
  if(!cell) return;
  $("#go").disabled = true;
  $("#out").innerHTML = "";
  const mode = $("#mode").value;
  $("#status").innerHTML = '<div class="status">Running '+(mode==="full"?"full pipeline (neuPrint → PubMed → Gemini ×2)":"neuPrint fingerprint")+' for <b>'+cell+'</b>… this can take ~30–60s.</div>';
  try {
    const r = await fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({cell_type:cell, dataset:$("#dataset").value.trim(), top_k:+$("#topk").value, mode})});
    const data = await r.json();
    $("#status").innerHTML = "";
    if (!r.ok || data.error) { renderError(data.error || ("HTTP "+r.status)); return; }
    render(data);
    if ("hypotheses" in data) loadHistory();
  } catch (e) {
    $("#status").innerHTML = ""; renderError(String(e));
  } finally { $("#go").disabled = false; }
});

function renderError(msg){
  const d = el("div","err", msg);
  $("#out").appendChild(d);
}

function roiTable(title, rois){
  const s = el("section");
  s.appendChild(el("h2",null,title));
  if(!rois || !rois.length){ s.appendChild(el("div","muted","none")); return s; }
  const t = el("table");
  t.innerHTML = "<tr><th>ROI</th><th style='text-align:right'>synapses</th></tr>";
  for(const r of rois){
    const tr=el("tr"); tr.appendChild(el("td",null,r.roi));
    tr.appendChild(el("td","num", (r.weight).toLocaleString())); t.appendChild(tr);
  }
  s.appendChild(t); return s;
}

function partnerTable(title, ps){
  const s = el("section"); s.appendChild(el("h2",null,title));
  if(!ps || !ps.length){ s.appendChild(el("div","muted","none")); return s; }
  const t = el("table");
  t.innerHTML="<tr><th>type</th><th>cells</th><th style='text-align:right'>synapses</th><th>NT</th><th>class</th></tr>";
  for(const p of ps){
    const tr=el("tr");
    tr.appendChild(el("td",null,p.type||"?"));
    tr.appendChild(el("td",null,String(p.n_cells)));
    tr.appendChild(el("td","num",(p.total_weight).toLocaleString()));
    tr.appendChild(el("td","muted",p.predicted_nt||"—"));
    tr.appendChild(el("td","muted",p["class"]||"—"));
    t.appendChild(tr);
  }
  s.appendChild(t); return s;
}

function renderFingerprint(fp){
  const out = $("#out");
  const head = el("section");
  head.appendChild(el("h2",null,"Structural fingerprint"));
  const kv = el("div","kv");
  kv.innerHTML = "<b>"+(fp.cell_type_query)+"</b> · "+fp.dataset+" · "+
    (fp.resolved && fp.resolved.length ? ("resolved <b>"+fp.resolved.length+"</b> cell(s)") :
      "<b>type not found</b>") + " · predicted NT: <b>"+(fp.predicted_nt||"unknown")+"</b>";
  head.appendChild(kv);
  if(fp.notes) head.appendChild(el("div","muted",fp.notes));
  if(fp.suggestions && fp.suggestions.length){
    const sg = el("div","chips suggest"); sg.style.marginTop="10px";
    sg.appendChild(el("span","muted","Did you mean: "));
    for(const s of fp.suggestions){ const c=el("span",null,s);
      c.onclick=()=>{ $("#cell").value=s; $("#f").dispatchEvent(new Event("submit")); };
      sg.appendChild(c); }
    head.appendChild(sg);
  }
  out.appendChild(head);
  out.appendChild(roiTable("Top input ROIs (postsynaptic)", fp.input_rois));
  out.appendChild(roiTable("Top output ROIs (presynaptic)", fp.output_rois));
  out.appendChild(partnerTable("Top upstream partners", fp.upstream));
  out.appendChild(partnerTable("Top downstream partners", fp.downstream));
}

function render(data){
  // fingerprint-only payload: {fingerprint:{...}}; full payload: full Hypothesis
  const fp = data.fingerprint;
  $("#out").appendChild(toolbar(data));
  renderFingerprint(fp);
  renderGraph(fp);
  if(!("hypotheses" in data)) return;  // fingerprint-only

  // literature
  const lit = el("section"); lit.appendChild(el("h2",null,"Literature used"));
  if(data.literature && data.literature.length){
    const ul=el("ul","lit");
    for(const h of data.literature){
      const li=el("li");
      const id=h.id||"n/a";
      const link = id.includes("/") ? "https://doi.org/"+id :
                   (/^\\d+$/.test(id) ? "https://pubmed.ncbi.nlm.nih.gov/"+id : null);
      const a = link ? '<a href="'+link+'" target="_blank" rel="noopener">['+h.source+':'+id+']</a>'
                     : '['+h.source+':'+id+']';
      li.innerHTML = a + " " + (h.year?("("+h.year+") "):"") + escapeHtml(h.title) +
        '<div class="why">why: '+escapeHtml(h.relevance)+'</div>';
      ul.appendChild(li);
    }
    lit.appendChild(ul);
  } else lit.appendChild(el("div","muted","No literature retrieved."));
  $("#out").appendChild(lit);

  // hypotheses
  const hs = el("section"); hs.appendChild(el("h2",null,"Hypotheses (tiered)"));
  data.hypotheses.forEach((h,i)=>{
    const card=el("div","hyp");
    const head=el("h3"); head.appendChild(el("span",null,"H"+(i+1)));
    head.appendChild(el("span","badge b-"+h.confidence, h.confidence));
    card.appendChild(head);
    card.appendChild(el("div",null,h.statement));
    card.appendChild(el("div","rat","Rationale: "+h.rationale));
    if(h.supporting_structure && h.supporting_structure.length){
      const ul=el("ul","grounded");
      for(const s of h.supporting_structure){ const li=el("li"); li.innerHTML="<code>"+escapeHtml(s)+"</code>"; ul.appendChild(li); }
      card.appendChild(ul);
    }
    if(h.supporting_literature && h.supporting_literature.length)
      card.appendChild(el("div","why","Supporting literature: "+h.supporting_literature.join(", ")));
    hs.appendChild(card);
  });
  $("#out").appendChild(hs);

  $("#out").appendChild(bullets("Not supported by connectivity", data.not_supported_by_connectivity));

  const exp = el("section"); exp.appendChild(el("h2",null,"Proposed experiments"));
  const eul=el("ul","lit");
  for(const e of (data.proposed_experiments||[])){
    const li=el("li"); li.innerHTML="<b>"+escapeHtml(e.hypothesis_ref)+"</b> — <i>"+
      escapeHtml(e.method)+"</i>: "+escapeHtml(e.expected_result); eul.appendChild(li);
  }
  exp.appendChild(eul); $("#out").appendChild(exp);

  if(data.caveats && data.caveats.length)
    $("#out").appendChild(bullets("Caveats", data.caveats));

  const v=el("section"); v.appendChild(el("h2",null,"Verification notes"));
  v.appendChild(el("div","muted", data.verification_notes||"(none)")); $("#out").appendChild(v);
}

function bullets(title, items){
  const s=el("section"); s.appendChild(el("h2",null,title));
  const ul=el("ul"); ul.style.margin="0"; ul.style.paddingLeft="18px";
  for(const it of (items||[])){ const li=el("li"); li.style.marginBottom="5px"; li.textContent=it; ul.appendChild(li); }
  s.appendChild(ul); return s;
}
function escapeHtml(s){ const d=el("div"); d.textContent=s==null?"":String(s); return d.innerHTML; }

// --- connectivity graph (SVG, no external libs) ------------------------- //
const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs){ const e=document.createElementNS(SVGNS,tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]); return e; }
function svgTitle(str){ const t=svg("title",{}); t.textContent=str; return t; }
function trunc(s,n){ s=String(s||"?"); return s.length>n ? s.slice(0,n-1)+"…" : s; }
function edgeLabel(x,y,str,color){
  const t=svg("text",{x:x,y:y,"text-anchor":"middle",fill:color,"font-size":"10","font-weight":"600"});
  t.textContent=str; return t;
}
function drawNode(root,x,y,w,h,title,sub,fill,stroke,center,tip){
  const r=svg("rect",{x:x,y:y,rx:8,width:w,height:h,fill:fill,stroke:stroke,"stroke-width":center?2:1});
  if(tip) r.appendChild(svgTitle(tip));
  root.appendChild(r);
  const t1=svg("text",{x:x+w/2,y:y+(sub?16:h/2+4),"text-anchor":"middle",fill:"#e7e9ee",
    "font-size":center?"14":"12","font-weight":center?"700":"600"});
  t1.textContent=trunc(title,16); root.appendChild(t1);
  if(sub){ const t2=svg("text",{x:x+w/2,y:y+h-8,"text-anchor":"middle",fill:"#9aa3b2","font-size":"10"});
    t2.textContent=sub; root.appendChild(t2); }
}
function renderGraph(fp){
  const N=8;
  const up=(fp.upstream||[]).slice(0,N), down=(fp.downstream||[]).slice(0,N);
  if(!up.length && !down.length) return;
  const s=el("section");
  s.appendChild(el("h2",null,"Connectivity graph (evidence for the hypotheses)"));
  s.appendChild(el("div","muted",
    "Arrows = synapse direction into / out of "+fp.cell_type_query+
    "; line thickness ∝ synapse weight; numbers = synapses; n = #cells. Showing top "+
    up.length+" upstream / "+down.length+" downstream of "+
    ((fp.upstream||[]).length)+" / "+((fp.downstream||[]).length)+"."));
  const rows=Math.max(up.length,down.length,1);
  const rowH=48, padTop=24, padBot=16, W=920, H=padTop+rows*rowH+padBot;
  const root=svg("svg",{viewBox:"0 0 "+W+" "+H, width:"100%",
    style:"max-width:100%;height:auto;margin-top:10px"});
  const defs=svg("defs",{});
  for(const a of [["aUp","#b69cff"],["aDown","#7cc4ff"]]){
    const m=svg("marker",{id:a[0],viewBox:"0 0 10 10",refX:"9",refY:"5",
      markerWidth:"7",markerHeight:"7",orient:"auto-start-reverse"});
    m.appendChild(svg("path",{d:"M0,0 L10,5 L0,10 z",fill:a[1]})); defs.appendChild(m);
  }
  root.appendChild(defs);
  const maxW=Math.max(1,...up.map(p=>p.total_weight),...down.map(p=>p.total_weight));
  const sw=w=>1.2+5.5*(w/maxW);
  const cy=H/2, cxL=380, cxR=540;
  up.forEach((p,i)=>{
    const y=padTop+i*rowH, x=10, w=180, h=34, yc=y+h/2;
    const ln=svg("line",{x1:x+w,y1:yc,x2:cxL,y2:cy,stroke:"#b69cff",
      "stroke-width":sw(p.total_weight),"stroke-opacity":"0.7","marker-end":"url(#aUp)"});
    ln.appendChild(svgTitle((p.type||"?")+" → "+fp.cell_type_query+": "+
      p.total_weight+" synapses, "+p.n_cells+" cells")); root.appendChild(ln);
    root.appendChild(edgeLabel(x+w+0.35*(cxL-(x+w)), yc+0.35*(cy-yc)-3,
      p.total_weight.toLocaleString(), "#b69cff"));
    drawNode(root,x,y,w,h,p.type,"n="+p.n_cells,"#0d0f15","#272b38",false,
      (p.type||"?")+" — n="+p.n_cells+" — w="+p.total_weight+(p.predicted_nt?(" — "+p.predicted_nt):""));
  });
  down.forEach((p,i)=>{
    const y=padTop+i*rowH, x=W-190, w=180, h=34, yc=y+h/2;
    const ln=svg("line",{x1:cxR,y1:cy,x2:x,y2:yc,stroke:"#7cc4ff",
      "stroke-width":sw(p.total_weight),"stroke-opacity":"0.7","marker-end":"url(#aDown)"});
    ln.appendChild(svgTitle(fp.cell_type_query+" → "+(p.type||"?")+": "+
      p.total_weight+" synapses, "+p.n_cells+" cells")); root.appendChild(ln);
    root.appendChild(edgeLabel(cxR+0.65*(x-cxR), cy+0.65*(yc-cy)-3,
      p.total_weight.toLocaleString(), "#7cc4ff"));
    drawNode(root,x,y,w,h,p.type,"n="+p.n_cells,"#0d0f15","#272b38",false,
      (p.type||"?")+" — n="+p.n_cells+" — w="+p.total_weight+(p.predicted_nt?(" — "+p.predicted_nt):""));
  });
  drawNode(root,cxL,cy-20,160,40,fp.cell_type_query,"n="+((fp.resolved||[]).length||"?"),
    "#1a2740","#7cc4ff",true,fp.cell_type_query);
  root.appendChild(svg("text",{x:10,y:14,fill:"#9aa3b2","font-size":"10"})).textContent="upstream →";
  const dlab=svg("text",{x:W-10,y:14,"text-anchor":"end",fill:"#9aa3b2","font-size":"10"});
  dlab.textContent="→ downstream"; root.appendChild(dlab);
  s.appendChild(root); $("#out").appendChild(s);
}

function slugify(s){ return (s||"report").replace(/[^A-Za-z0-9_.-]+/g,"_").replace(/^_+|_+$/g,"")||"report"; }
function downloadBlob(name, text, type){
  const b=new Blob([text],{type}); const a=el("a");
  a.href=URL.createObjectURL(b); a.download=name; document.body.appendChild(a);
  a.click(); a.remove(); URL.revokeObjectURL(a.href);
}
function toolbar(data){
  const slug = slugify(data.cell_type || (data.fingerprint && data.fingerprint.cell_type_query));
  const bar = el("div","toolbar");
  const jb = el("button",null,"⬇ JSON");
  jb.onclick = () => { const c={...data}; delete c._markdown;
    downloadBlob(slug+".json", JSON.stringify(c,null,2), "application/json"); };
  bar.appendChild(jb);
  if (data._markdown){
    const mb = el("button",null,"⬇ Markdown");
    mb.onclick = () => downloadBlob(slug+".md", data._markdown, "text/markdown");
    bar.appendChild(mb);
  }
  return bar;
}

async function loadHistory(){
  try {
    const r = await fetch("/api/history"); const items = await r.json();
    const box = $("#history"); box.innerHTML = "";
    if (!items.length) return;
    box.appendChild(el("span",null,"Saved reports: "));
    for (const it of items){
      const c = el("span","chip");
      const lab = el("span","lab", it.slug + " · " + relTime(it.mtime));
      lab.title = new Date(it.mtime*1000).toLocaleString();
      lab.onclick = () => openReport(it.slug);
      const x = el("span","x","✕"); x.title = "delete";
      x.onclick = (ev) => { ev.stopPropagation(); delReport(it.slug); };
      c.appendChild(lab); c.appendChild(x); box.appendChild(c);
    }
  } catch (e) { /* ignore */ }
}

function relTime(sec){
  const s = Math.max(0, (Date.now()/1000) - sec);
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s/60) + "m ago";
  if (s < 86400) return Math.floor(s/3600) + "h ago";
  return Math.floor(s/86400) + "d ago";
}

async function delReport(slug){
  if (!confirm("Delete saved report '"+slug+"'?")) return;
  try { await fetch("/api/report/"+encodeURIComponent(slug), {method:"DELETE"}); }
  catch (e) { /* ignore */ }
  loadHistory();
}

async function openReport(slug){
  $("#go").disabled = true; $("#out").innerHTML = "";
  $("#status").innerHTML = '<div class="status">Loading saved report <b>'+slug+'</b>…</div>';
  try {
    const r = await fetch("/api/report/"+encodeURIComponent(slug));
    const data = await r.json(); $("#status").innerHTML = "";
    if (!r.ok || data.error) { renderError(data.error || ("HTTP "+r.status)); return; }
    render(data);
  } catch (e) { $("#status").innerHTML=""; renderError(String(e)); }
  finally { $("#go").disabled = false; }
}

loadHistory();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/history":
            self._send(200, json.dumps(self._history()).encode("utf-8"),
                       "application/json")
        elif path.startswith("/api/report/"):
            slug = path[len("/api/report/"):]
            fmt = (parse_qs(parsed.query).get("fmt") or ["json"])[0]
            self._serve_report(slug, fmt)
        else:
            self._send(404, b"not found", "text/plain")

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/report/"):
            slug = _slug(path[len("/api/report/"):])  # sanitise — no traversal
            deleted = 0
            for ext in ("json", "md"):
                p = OUTPUT_DIR / f"{slug}.{ext}"
                if p.exists():
                    p.unlink()
                    deleted += 1
            self._send(200, json.dumps({"ok": True, "deleted": deleted}).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            payload = self._run(req)
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
        except Exception as e:  # surface a readable error to the UI
            msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            self._send(400, json.dumps({"error": msg}).encode("utf-8"), "application/json")

    @staticmethod
    def _run(req: dict) -> dict:
        cell = (req.get("cell_type") or "").strip()
        dataset = (req.get("dataset") or connectome.DEFAULT_DATASET).strip()
        top_k = int(req.get("top_k") or 15)
        mode = req.get("mode", "full")
        if not cell:
            return {"error": "cell_type is required"}

        fp = connectome.build_fingerprint(cell, dataset, top_k)
        if mode == "fingerprint" or not fp.found:
            # Fingerprint-only, OR not-found: skip the LLM for the pure-structure
            # view, but still synthesise for not-found in full mode (degraded).
            if mode == "fingerprint":
                return {"fingerprint": fp.model_dump(by_alias=True)}

        lit = literature.fetch_literature(fp)
        result = synthesize.synthesize(fp, lit)

        # Persist like the CLI does, so it shows up in history.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        slug = _slug(cell)
        (OUTPUT_DIR / f"{slug}.json").write_text(
            result.model_dump_json(indent=2, by_alias=True)
        )
        (OUTPUT_DIR / f"{slug}.md").write_text(render_markdown(result))

        payload = result.model_dump(by_alias=True)
        payload["_markdown"] = render_markdown(result)  # for the MD download button
        return payload

    @staticmethod
    def _history() -> list[dict]:
        if not OUTPUT_DIR.is_dir():
            return []
        items = []
        for p in OUTPUT_DIR.glob("*.json"):
            items.append({"slug": p.stem, "mtime": p.stat().st_mtime})
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return items

    def _serve_report(self, slug: str, fmt: str) -> None:
        slug = _slug(slug)  # sanitise — no path traversal
        jpath = OUTPUT_DIR / f"{slug}.json"
        if not jpath.exists():
            self._send(404, json.dumps({"error": "report not found"}).encode(),
                       "application/json")
            return
        if fmt == "md":
            md = (OUTPUT_DIR / f"{slug}.md")
            body = md.read_text() if md.exists() else "# (markdown missing)"
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{slug}.md"')
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return
        # Re-render markdown so the loaded report also gets a MD download button.
        result = Hypothesis.model_validate_json(jpath.read_text())
        payload = result.model_dump(by_alias=True)
        payload["_markdown"] = render_markdown(result)
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flyhypo-web", description="Local web UI for flyhypo.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"flyhypo web UI → http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
