"""The dashboard shell (v4.22): one app frame around every page of the
website -- a collapsible sidebar with the sections in groups and the recent
runs, a sticky top bar, a command palette (Cmd/Ctrl-K, or "/") that finds any
ticker on the page, click-to-sort tables and a system / light / dark switch.

Two modes share one stylesheet and one script:

  pages  the local website (web.py): every section is its own URL and the
         sidebar links are real links; the browser's cross-document view
         transition (where supported) makes the switch feel instant.
  spa    the published site (static_site.py): every section is a pane of ONE
         encrypted page, chosen by the URL hash (#stocks). Switching needs no
         download and no second password; old per-section links redirect.

Everything is plain HTML, CSS and vanilla JavaScript: no library, no font
download, nothing fetched from outside (the published site is private).
"""

from __future__ import annotations

import html
from dataclasses import dataclass

# (key, label, group) -- the order is the sidebar's order
SECTIONS: list[tuple[str, str, str]] = [
    ("overview", "Overview", "Trade"), ("stocks", "Stocks", "Trade"),
    ("etfs", "ETFs", "Trade"), ("sell", "When to sell", "Trade"),
    ("tracker", "Tracker", "Performance"), ("graphs", "Graphs", "Performance"),
    ("montecarlo", "Monte Carlo", "Performance"), ("risk", "Risk (VaR)", "Performance"),
    ("macro", "Macro", "Context"), ("checks", "Checks", "Context"),
    ("runs", "History", "Runs")]
LABELS = {k: label for k, label, _ in SECTIONS}

_STROKE = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
           'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>')
ICONS = {
    "overview": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" '
                'width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" '
                'rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "stocks": '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    "etfs": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "sell": '<path d="M17 3l4 4-4 4"/><path d="M3 7h18"/><path d="M7 21l-4-4 4-4"/>'
            '<path d="M21 17H3"/>',
    "tracker": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "graphs": '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/>'
              '<path d="M22 20H2"/>',
    "montecarlo": '<path d="M16 3h5v5"/><path d="M4 20L21 3"/><path d="M21 16v5h-5"/>'
                  '<path d="M15 15l6 6"/><path d="M4 4l5 5"/>',
    "risk": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"/>',
    "macro": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/>'
             '<path d="M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18z"/>',
    "checks": '<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/>',
    "runs": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "menu": '<path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h16"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
    "logo": '<path d="M3 11a9 6.5 0 0 1 18 0c0 .6-.4 1-1 1H4c-.6 0-1-.4-1-1z"/>'
            '<path d="M9 12l-.5 6a1.5 1.5 0 0 0 1.5 1.6h4a1.5 1.5 0 0 0 1.5-1.6L15 12"/>',
}


def icon(name: str) -> str:
    return _STROKE.format(ICONS[name])


@dataclass(frozen=True)
class Shell:
    """What the frame around a page needs to know."""
    active: str                                  # section key, "" for none
    links: dict[str, str]                        # section key -> href
    mode: str = "pages"                          # "pages" | "spa"
    as_of: str = ""                              # the close the signals use
    status: str = ""                             # sidebar foot: generated / version
    updated: str = ""                            # "Updated 02 Nov 07:41"
    action: str = ""                             # HTML: Run now (button or link)
    runs: tuple[tuple[str, str, str], ...] = ()  # (label, href, "ok" | "bad")
    notice: str = ""                             # HTML banner under the top bar
    extra_js: str = ""                           # e.g. the local Run now poller
    extra_css: str = ""


_GROUP_ORDER = ["Trade", "Performance", "Context", "Runs"]


def sidebar(shell: Shell) -> str:
    groups: dict[str, list[str]] = {g: [] for g in _GROUP_ORDER}
    for key, label, group in SECTIONS:
        href = shell.links.get(key)
        if href is None:
            continue
        cur = ' aria-current="page"' if key == shell.active else ""
        groups.setdefault(group, []).append(
            f'<a data-key="{key}" href="{html.escape(href)}"{cur}>{icon(key)}'
            f"{html.escape(label)}</a>")
    nav = []
    for g in _GROUP_ORDER:
        if groups.get(g):
            nav.append(f'<div class="grp">{html.escape(g)}</div>' + "".join(groups[g]))
    runs = ""
    if shell.runs:
        items = []
        for label, href, state in shell.runs:
            dot = f'<span class="{"ok" if state == "ok" else "bad"}">●</span>'
            items.append(f'<a href="{html.escape(href)}"><span>{html.escape(label)}</span>'
                         f"{dot}</a>" if href else
                         f'<span class="run"><span>{html.escape(label)}</span>{dot}</span>')
        runs = ('<div class="grp">Recent runs</div><div class="runs-list">'
                + "".join(items) + "</div>")
    as_of = (f"<small>Signals of {html.escape(shell.as_of)}</small>" if shell.as_of
             else "<small>No run yet</small>")
    seg = "".join(
        f'<button type="button" data-theme-pick="{t}" aria-pressed="{str(t == "system").lower()}">'
        f"{t.capitalize()}</button>" for t in ("system", "light", "dark"))
    foot = [shell.action, f'<div class="seg" role="group" aria-label="Theme">{seg}</div>']
    if shell.status:
        foot.append(f"<small>{html.escape(shell.status)}</small>")
    if shell.updated:
        foot.append(f"<small>{html.escape(shell.updated)}</small>")
    return (f'<aside class="side" id="umw-side"><div class="brand"><span class="logo">'
            f'{icon("logo")}</span><div><b>UMushroom</b>{as_of}</div></div>'
            f'<nav class="nav" aria-label="Sections">{"".join(nav)}{runs}</nav>'
            f'<div class="side-foot">{"".join(foot)}</div></aside>')


def topbar(shell: Shell, title: str) -> str:
    chip = (f'<span class="chip">as of {html.escape(shell.as_of)}</span>' if shell.as_of
            else "")
    return (f'<header class="top"><button class="icon-btn" id="umw-menu" type="button" '
            f'aria-label="Sections" aria-expanded="false" aria-controls="umw-side">'
            f'{icon("menu")}</button><h1 id="umw-title">{html.escape(title)}</h1>{chip}'
            f'<span class="sp"></span><button class="search" id="umw-search" type="button" '
            f'aria-label="Search this page (Cmd or Ctrl K)">{icon("search")}'
            f"<span>Find a ticker</span><kbd>⌘K</kbd></button></header>")


def palette() -> str:
    return ('<div class="pal" id="umw-pal" hidden role="dialog" aria-modal="true" '
            'aria-label="Find on this page"><div class="pal-box">'
            '<input id="umw-q" type="search" autocomplete="off" spellcheck="false" '
            'placeholder="Ticker, name or section…" aria-label="Search">'
            '<div class="pal-res" id="umw-res" role="listbox"></div></div></div>')


def pane(key: str, body: str, active: bool) -> str:
    """One section of a single-page (spa) site."""
    return (f'<section class="pane" data-key="{key}" data-label="{html.escape(LABELS[key])}"'
            f'{"" if active else " hidden"}>{body}</section>')


def report_bar(back_href: str, label: str) -> str:
    """The slim bar at the top of a stored full report: back to the dashboard."""
    return (f'<style>{BAR_CSS}</style><nav class="umw-bar" aria-label="Dashboard">'
            f'<a class="umw-back" href="{html.escape(back_href)}">&larr; Back to the dashboard'
            f'</a><span class="umw-lbl">{html.escape(label)}</span></nav>')


THEME_BOOT = ("(function(){try{var t=localStorage.getItem('umw-theme');"
              "if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-theme',t);}}"
              "catch(e){}})();")

BAR_CSS = """
.umw-bar{position:sticky;top:0;z-index:50;display:flex;gap:12px;align-items:center;
padding:8px max(16px,env(safe-area-inset-right,0px)) 8px max(16px,env(safe-area-inset-left,0px));
background:var(--surface-1,#fcfcfb);border-bottom:1px solid var(--border,#e2e0da);
font:14px/1.4 system-ui,-apple-system,sans-serif}
.umw-back{color:var(--series-1,#2a78d6);text-decoration:none;font-weight:600}
.umw-lbl{color:var(--muted,#888);font-size:13px}
"""

SHELL_CSS = """
@view-transition{navigation:auto}
.app{display:grid;grid-template-columns:264px minmax(0,1fr);min-height:100vh}
.side{position:sticky;top:0;height:100vh;height:100dvh;display:flex;
 flex-direction:column;gap:6px;padding:14px 12px;background:var(--surface-2);
 border-right:1px solid var(--border);font-size:14px}
.brand{display:flex;align-items:center;gap:10px;padding:4px 8px 10px}
.brand .logo{width:34px;height:34px;border-radius:10px;background:var(--series-1);color:#fff;
 display:grid;place-items:center;flex:none}
.brand .logo svg{width:20px;height:20px}
.brand b{display:block;font-size:15px}.brand small{display:block;color:var(--text-secondary);
 font-size:12px}
.nav{display:flex;flex-direction:column;gap:2px;flex:1;min-height:0;overflow-y:auto;
 margin:0 -4px;padding:0 4px}
.grp{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
 padding:12px 10px 4px}
.nav a{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:10px;
 color:var(--text-secondary);text-decoration:none;line-height:1.3}
.nav a svg{width:18px;height:18px;flex:none;opacity:.85}
.nav a:hover{background:var(--hover);color:var(--text-primary)}
.nav a[aria-current="page"]{background:var(--accent-soft);color:var(--text-primary);
 font-weight:600}
.nav a[aria-current="page"] svg{color:var(--series-1);opacity:1}
.runs-list{display:flex;flex-direction:column;gap:1px}
.runs-list a,.runs-list .run{display:flex;justify-content:space-between;gap:8px;
 padding:6px 10px;border-radius:8px;color:var(--text-secondary);text-decoration:none;
 font-size:13px;font-variant-numeric:tabular-nums}
.runs-list a:hover{background:var(--hover);color:var(--text-primary)}
.runs-list .ok{color:var(--good)}.runs-list .bad{color:var(--bad)}
.side-foot{display:flex;flex-direction:column;gap:10px;padding:12px 4px 0;
 border-top:1px solid var(--border);font-size:12px;color:var(--muted)}
.app .tiles{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.side-foot small{display:block;line-height:1.4}
.seg{display:grid;grid-template-columns:repeat(3,1fr);background:var(--hover);
 border-radius:10px;padding:3px;gap:2px}
.seg button{font:inherit;font-size:12px;padding:5px 0;border:0;border-radius:8px;
 background:transparent;color:var(--text-secondary);cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--surface-1);color:var(--text-primary);
 box-shadow:var(--shadow-sm)}
.btn{font:inherit;font-size:13px;font-weight:600;padding:8px 12px;border-radius:10px;border:0;
 cursor:pointer;background:var(--series-1);color:#fff;text-decoration:none;text-align:center;
 display:inline-flex;align-items:center;justify-content:center;gap:6px}
.btn:disabled{opacity:.6;cursor:default}
.st{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;
 white-space:nowrap}
.main{min-width:0;display:flex;flex-direction:column}
.top{position:sticky;top:env(safe-area-inset-top,0px);z-index:40;display:flex;align-items:center;
 gap:10px;padding:10px max(16px,env(safe-area-inset-right,0px)) 10px
 max(16px,env(safe-area-inset-left,0px));background:var(--glass);backdrop-filter:blur(12px);
 -webkit-backdrop-filter:blur(12px);border-bottom:1px solid var(--border)}
.top h1{font-size:16px;margin:0;font-weight:600;white-space:nowrap;overflow:hidden;
 text-overflow:ellipsis}
.chip{font-size:12px;color:var(--text-secondary);background:var(--surface-1);
 border:1px solid var(--border);border-radius:999px;padding:3px 10px;white-space:nowrap}
.sp{flex:1}
.icon-btn{display:none;width:36px;height:36px;border-radius:10px;border:1px solid var(--border);
 background:var(--surface-1);color:var(--text-primary);cursor:pointer;place-items:center;
 flex:none}
.icon-btn svg{width:18px;height:18px}
.search{display:inline-flex;align-items:center;gap:8px;font:inherit;font-size:13px;
 color:var(--text-secondary);background:var(--surface-1);border:1px solid var(--border);
 border-radius:999px;padding:6px 10px 6px 12px;cursor:pointer;min-width:0}
.search:hover{color:var(--text-primary)}
.search svg{width:16px;height:16px;flex:none}
.search kbd{font:11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--border);
 border-radius:6px;padding:3px 5px;color:var(--muted)}
.app main{width:100%;max-width:1100px;margin:0 auto;padding:20px max(16px,env(safe-area-inset-right,0px))
 72px max(16px,env(safe-area-inset-left,0px))}
.app main>h2:first-child,.pane>h2:first-child{margin-top:8px}
.pane[hidden]{display:none}
.notice{background:var(--surface-1);border:1px solid var(--border);border-left:4px solid var(--warn);
 border-radius:var(--radius);padding:10px 14px;margin:0 0 16px;font-size:14px}
.notice a{color:inherit}
.tile,.card,table.data,.due{border-radius:var(--radius)}
.tile{box-shadow:var(--shadow-sm)}
th.sortable{cursor:pointer;user-select:none}th.sortable:hover{color:var(--text-primary)}
th[aria-sort="ascending"]::after{content:" \\25B4"}th[aria-sort="descending"]::after{content:" \\25BE"}
tbody tr:hover{background:var(--hover)}
tr.hit{animation:hit 1.6s ease-out}
@keyframes hit{0%{background:var(--accent-soft)}100%{background:transparent}}
.pal{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.35);display:grid;
 place-items:start center;padding:12vh 16px 16px}
.pal[hidden]{display:none}
.pal-box{width:min(640px,100%);background:var(--surface-1);border:1px solid var(--border);
 border-radius:16px;box-shadow:var(--shadow);overflow:hidden}
.pal-box input{width:100%;box-sizing:border-box;font:inherit;font-size:16px;padding:14px 16px;
 border:0;border-bottom:1px solid var(--border);background:transparent;
 color:var(--text-primary);outline:none}
.pal-res{max-height:50vh;overflow:auto;padding:6px}
.pal-res button{display:flex;width:100%;gap:10px;align-items:baseline;text-align:left;
 font:inherit;font-size:14px;padding:8px 10px;border:0;border-radius:8px;background:transparent;
 color:var(--text-primary);cursor:pointer}
.pal-res button[aria-selected="true"]{background:var(--accent-soft)}
.pal-res .where{margin-left:auto;font-size:12px;color:var(--muted);white-space:nowrap}
.pal-empty{padding:14px 12px;color:var(--muted);font-size:13px}
.scrim{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:45}
.up{color:var(--good)}.down{color:var(--bad)}
.runs-tbl .succeeded{color:var(--good)}
.runs-tbl .failed,.runs-tbl .interrupted{color:var(--bad)}
form.pick{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}
form.pick a{color:var(--series-1)}
form.pick select,form.pick button{font:inherit;padding:5px 8px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary)}
.act{display:inline-block;font-weight:600;font-size:12px;padding:2px 8px;border-radius:6px;
 white-space:nowrap;border:1px solid currentColor}
.act-sell{color:var(--bad)}.act-buy{color:var(--good)}.act-hold{color:var(--text-secondary)}
.due{background:var(--surface-1);border:1px solid var(--border);
 border-left:4px solid var(--series-1);padding:12px 16px;margin:0 0 16px}
.due.now{border-left-color:var(--bad)}.due b{display:block;font-size:17px;margin-bottom:2px}
.sell-tbl td:last-child{white-space:normal;min-width:260px}
a:focus-visible,button:focus-visible,th:focus-visible{outline:2px solid var(--series-1);
 outline-offset:2px}
@media (max-width:900px){
 .app{grid-template-columns:1fr}
 .side{position:fixed;left:0;top:0;bottom:0;width:min(300px,86vw);z-index:50;
  transform:translateX(-102%);transition:transform .2s ease;box-shadow:var(--shadow)}
 .app.nav-open .side{transform:none}.app.nav-open .scrim{display:block}
 .icon-btn{display:grid}
 .search span,.search kbd{display:none}.search{padding:8px}
 .chip{display:none}
}
@media (prefers-reduced-motion:reduce){.side{transition:none}tr.hit{animation:none}}
"""

SHELL_JS = r"""
(function(){
var root=document.documentElement,app=document.querySelector('.app');if(!app){return;}
var KEY='umw-theme';
function applyTheme(t){
 if(t==='light'||t==='dark'){root.setAttribute('data-theme',t);}else{root.removeAttribute('data-theme');t='system';}
 app.querySelectorAll('[data-theme-pick]').forEach(function(b){
  b.setAttribute('aria-pressed',String(b.getAttribute('data-theme-pick')===t));});}
var saved=null;try{saved=localStorage.getItem(KEY);}catch(e){}
applyTheme(saved||'system');
app.querySelectorAll('[data-theme-pick]').forEach(function(b){b.addEventListener('click',function(){
 var t=b.getAttribute('data-theme-pick');
 try{if(t==='system'){localStorage.removeItem(KEY);}else{localStorage.setItem(KEY,t);}}catch(e){}
 applyTheme(t);});});
var menu=document.getElementById('umw-menu'),scrim=app.querySelector('.scrim');
function closeNav(){app.classList.remove('nav-open');if(menu){menu.setAttribute('aria-expanded','false');}}
if(menu){menu.addEventListener('click',function(){var on=!app.classList.contains('nav-open');
 app.classList.toggle('nav-open',on);menu.setAttribute('aria-expanded',String(on));});}
if(scrim){scrim.addEventListener('click',closeNav);}
var panes=app.querySelectorAll('.pane'),links=app.querySelectorAll('.nav a[data-key]');
var title=document.getElementById('umw-title');
function show(key){var hit=null;
 panes.forEach(function(p){var on=p.getAttribute('data-key')===key;p.hidden=!on;if(on){hit=p;}});
 if(!hit){return key==='overview'?null:show('overview');}
 links.forEach(function(a){if(a.getAttribute('data-key')===key){a.setAttribute('aria-current','page');}
  else{a.removeAttribute('aria-current');}});
 if(title){var l=hit.getAttribute('data-label');title.textContent=l;
  document.title=key==='overview'?'UMushroom':'UMushroom: '+l;}
 closeNav();window.scrollTo(0,0);return hit;}
if(panes.length){
 var fromHash=function(){show((location.hash||'#overview').slice(1));};
 window.addEventListener('hashchange',fromHash);fromHash();}
function num(s){var n=parseFloat(s.replace(/[%,+x£$€\s]/g,'').replace(/−/g,'-'));
 return isNaN(n)?null:n;}
app.querySelectorAll('main table').forEach(function(tbl){
 var body=tbl.tBodies[0],ths=tbl.querySelectorAll('thead th');
 if(!body||!ths.length||body.rows.length<2){return;}
 ths.forEach(function(th,i){th.classList.add('sortable');th.tabIndex=0;
  th.setAttribute('title','Sort by '+th.textContent.trim());
  function go(){var dir=th.getAttribute('aria-sort')==='ascending'?'descending':'ascending';
   ths.forEach(function(x){x.removeAttribute('aria-sort');});th.setAttribute('aria-sort',dir);
   var rows=Array.prototype.slice.call(body.rows);
   function val(r){var c=r.cells[i];if(!c){return '';}var s=c.textContent.trim();var n=num(s);
    return n===null?s.toLowerCase():n;}
   rows.sort(function(a,b){var va=val(a),vb=val(b),na=typeof va==='number',nb=typeof vb==='number';
    if(na&&nb){return dir==='ascending'?va-vb:vb-va;}if(na){return -1;}if(nb){return 1;}
    var c=va<vb?-1:va>vb?1:0;return dir==='ascending'?c:-c;});
   rows.forEach(function(r){body.appendChild(r);});}
  th.addEventListener('click',go);
  th.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}});});});
var pal=document.getElementById('umw-pal'),q=document.getElementById('umw-q'),
 res=document.getElementById('umw-res'),items=[],list=[],sel=0;
function heading(el){var n=el.closest('table');
 while(n&&n!==app){if(n.tagName==='H2'||n.tagName==='H3'){return n.textContent.trim();}
  n=n.previousElementSibling||n.parentElement;}return '';}
function build(){items=[];
 links.forEach(function(a){items.push({t:a.textContent.trim(),w:'Section',go:function(){
  if(panes.length){location.hash='#'+a.getAttribute('data-key');}else{location.href=a.getAttribute('href');}}});});
 app.querySelectorAll('main table tbody tr').forEach(function(tr){var c=tr.cells;
  if(!c||c.length<2||tr.closest('details')){return;}var t=[];
  for(var i=0;i<Math.min(3,c.length);i++){var s=c[i].textContent.trim();if(s){t.push(s);}}
  t=t.join('  ');if(!/[A-Za-z]/.test(t)){return;}
  var p=tr.closest('.pane'),w=p?p.getAttribute('data-label'):(title?title.textContent:'');
  var h=heading(tr);if(h){w+=' · '+h;}
  items.push({t:t.slice(0,80),w:w,go:function(){
   if(p){location.hash='#'+p.getAttribute('data-key');}
   tr.scrollIntoView({block:'center'});tr.classList.remove('hit');void tr.offsetWidth;tr.classList.add('hit');}});});}
function mark(){var bs=res.querySelectorAll('button');bs.forEach(function(b,i){b.setAttribute('aria-selected',String(i===sel));});
 var cur=bs[sel];if(cur&&cur.scrollIntoView){cur.scrollIntoView({block:'nearest'});}}
function render(){var v=(q.value||'').trim().toLowerCase();list=items;
 if(v){list=items.filter(function(it){return it.t.toLowerCase().indexOf(v)>=0||it.w.toLowerCase().indexOf(v)>=0;});
  list=list.slice().sort(function(a,b){var sa=a.t.toLowerCase().indexOf(v)===0?0:1,sb=b.t.toLowerCase().indexOf(v)===0?0:1;return sa-sb;});}
 list=list.slice(0,12);sel=Math.min(sel,Math.max(0,list.length-1));res.textContent='';
 if(!list.length){var e=document.createElement('div');e.className='pal-empty';
  e.textContent='Nothing on this page matches "'+q.value+'".';res.appendChild(e);return;}
 list.forEach(function(it,i){var b=document.createElement('button');b.type='button';
  b.setAttribute('role','option');var s=document.createElement('span');s.textContent=it.t;
  var w=document.createElement('span');w.className='where';w.textContent=it.w;
  b.appendChild(s);b.appendChild(w);
  b.addEventListener('click',function(){close();it.go();});
  b.addEventListener('mouseenter',function(){sel=i;mark();});res.appendChild(b);});mark();}
function open(){if(!pal){return;}if(!items.length){build();}pal.hidden=false;q.value='';sel=0;render();q.focus();}
function close(){if(pal){pal.hidden=true;}}
var sb=document.getElementById('umw-search');if(sb){sb.addEventListener('click',open);}
if(pal){pal.addEventListener('click',function(e){if(e.target===pal){close();}});
 q.addEventListener('input',function(){sel=0;render();});
 q.addEventListener('keydown',function(e){
  if(e.key==='ArrowDown'){e.preventDefault();sel=Math.min(sel+1,list.length-1);mark();}
  else if(e.key==='ArrowUp'){e.preventDefault();sel=Math.max(sel-1,0);mark();}
  else if(e.key==='Enter'){e.preventDefault();if(list[sel]){close();list[sel].go();}}});}
document.addEventListener('keydown',function(e){
 if(e.key==='Escape'){if(pal&&!pal.hidden){close();}else{closeNav();}return;}
 var typing=/^(INPUT|TEXTAREA|SELECT)$/.test((e.target&&e.target.tagName)||'');
 if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();if(pal&&!pal.hidden){close();}else{open();}}
 else if(e.key==='/'&&!typing){e.preventDefault();open();}});
})();
"""

APP_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{title}</title><script>{boot}</script><style>{css}{shell_css}{extra_css}</style></head>
<body><div class="app">{sidebar}<div class="scrim"></div><div class="main">{topbar}<main>{notice}{body}</main></div>{palette}</div>
<script id="report-data" type="application/json">{data}</script>
<script>{js}</script><script>{shell_js}</script>{extra}</body></html>
"""


def render_app(shell: Shell, title: str, body: str, css: str, js: str, data: str,
               sub: str = "") -> str:
    """The whole document: shell around `body` (report.render_page builds it)."""
    if sub and not shell.status:
        shell = Shell(**{**shell.__dict__, "status": sub})
    extra = f"<script>{shell.extra_js}</script>" if shell.extra_js else ""
    notice = f'<div class="notice">{shell.notice}</div>' if shell.notice else ""
    return APP_PAGE.format(
        title=html.escape(title), boot=THEME_BOOT, css=css, shell_css=SHELL_CSS,
        extra_css=shell.extra_css, sidebar=sidebar(shell), topbar=topbar(shell, title),
        notice=notice, body=body, palette=palette(), data=data, js=js,
        shell_js=SHELL_JS, extra=extra)


def redirect_stub(target: str) -> str:
    """An unencrypted, contentless page that forwards an old per-section URL
    to the single-page dashboard (bookmarks keep working)."""
    t = html.escape(target)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"robots\" content=\"noindex,nofollow\">"
            f"<meta http-equiv=\"refresh\" content=\"0;url={t}\"><title>UMushroom</title>"
            f"</head><body><p><a href=\"{t}\">Continue to the dashboard</a></p></body></html>\n")

