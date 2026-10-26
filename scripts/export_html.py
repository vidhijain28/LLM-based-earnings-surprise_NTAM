"""Export executed notebooks to docs/*.html: code hidden, table of contents, cleaner styling.

Usage:
    uv run --with nbconvert python scripts/export_html.py notebooks/ciq_transcripts_eda.ipynb [...]
"""
import html
import re
import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

DOCS = Path(__file__).resolve().parent.parent / 'docs'

CSS = """
<style>
  :root { --ink: #0b0b0b; --ink2: #52514e; --rule: #e8e7e3; --accent: #2a78d6; --surface: #fcfcfb; }
  body { background: var(--surface); color: var(--ink);
         font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  main { max-width: 1120px; margin: 0 auto !important; padding: 24px 32px 80px !important; }
  .jp-Cell { padding: 2px 0 !important; }
  .jp-RenderedHTMLCommon h1 { font-size: 2em; margin: 0.4em 0 0.6em; }
  .jp-RenderedHTMLCommon h2 { font-size: 1.45em; margin-top: 1.8em; padding-top: 0.6em; border-top: 1px solid var(--rule); }
  .jp-RenderedHTMLCommon h3 { font-size: 1.15em; margin-top: 1.4em; }
  .jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2, .jp-RenderedHTMLCommon h3 { scroll-margin-top: 16px; }
  .jp-RenderedHTMLCommon .anchor-link { visibility: hidden; text-decoration: none; margin-left: 6px; color: var(--ink2); }
  .jp-RenderedHTMLCommon h2:hover .anchor-link, .jp-RenderedHTMLCommon h3:hover .anchor-link { visibility: visible; }
  .jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li { max-width: 860px; }
  .jp-RenderedHTMLCommon code { background: #f1f0ec; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
  /* tables */
  .jp-RenderedHTMLCommon table { border-collapse: collapse; font-size: 13px; margin: 8px 0 16px; border: none; }
  .jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td { padding: 4px 10px; border: none; border-bottom: 1px solid var(--rule); }
  .jp-RenderedHTMLCommon thead th { border-bottom: 1.5px solid #c3c2b7; color: var(--ink2); font-weight: 600; }
  .jp-RenderedHTMLCommon tbody tr:nth-child(odd) { background: rgba(0, 0, 0, 0.018); }
  .jp-RenderedHTMLCommon tbody tr:hover { background: rgba(42, 120, 214, 0.06); }
  .jp-RenderedHTMLCommon caption { caption-side: top; text-align: left; color: var(--ink2); font-weight: 600; padding: 4px 0; }
  .jp-OutputArea-output pre { font-size: 12.5px; line-height: 1.45; color: var(--ink2); }
  .jp-OutputArea-output img { max-width: 100%; height: auto; }
  .jp-OutputArea-output { overflow-x: auto; }
  /* table of contents */
  nav.toc { font-size: 13.5px; line-height: 1.45; }
  nav.toc .toc-title { font-weight: 700; text-transform: uppercase; letter-spacing: .06em; font-size: 11.5px; color: var(--ink2); margin-bottom: 6px; }
  nav.toc ol { list-style: none; margin: 0 !important; padding: 0 !important; }
  nav.toc li { margin: 2px 0; }
  nav.toc li.h3 { padding-left: 14px; font-size: 12.5px; }
  nav.toc a { color: var(--ink); text-decoration: none; border-left: 2px solid transparent; padding-left: 8px; display: block; }
  nav.toc a:hover { color: var(--accent); border-left-color: var(--accent); }
  @media (min-width: 1500px) {
    nav.toc { position: fixed; top: 24px; left: 24px; width: 260px; max-height: calc(100vh - 48px); overflow-y: auto; }
    main { margin-left: 300px !important; margin-right: auto !important; }
  }
  @media (max-width: 1499px) {
    nav.toc { border: 1px solid var(--rule); border-radius: 6px; padding: 12px 16px; margin: 8px 0 24px; background: #fff; }
  }
</style>
"""

HEADING = re.compile(r'<h([23])\s+id="([^"]+)"[^>]*>(.*?)</h\1>', re.S)


def build_toc(body):
    items = []
    for level, anchor, inner in HEADING.findall(body):
        text = re.sub(r'<a class="anchor-link".*?</a>', '', inner, flags=re.S)
        text = html.unescape(re.sub(r'<[^>]+>', '', text)).strip()
        items.append(f'<li class="h{level}"><a href="#{anchor}">{html.escape(text)}</a></li>')
    return f'<nav class="toc"><div class="toc-title">Contents</div><ol>{"".join(items)}</ol></nav>'


def export(nb_path):
    nb_path = Path(nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    # drop the WRDS connection chatter ("Loading library list...", "Done"), which may be split across outputs
    chatter = {'Loading library list...', 'Done', 'Loading library list...\nDone'}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = [o for o in cell.get('outputs', [])
                            if not (o.get('output_type') == 'stream' and o.get('text', '').strip() in chatter)]
    exporter = HTMLExporter(template_name='lab')
    exporter.exclude_input = True           # hide code cells
    exporter.exclude_input_prompt = True    # hide In [ ] prompts
    exporter.exclude_output_prompt = True   # hide Out[ ] prompts
    body, _ = exporter.from_notebook_node(nb)

    body = body.replace('</head>', CSS + '</head>', 1)
    toc = build_toc(body)
    # place the TOC right after the notebook title (first h1)
    m = re.search(r'</h1>', body)
    body = body[:m.end()] + toc + body[m.end():] if m else body.replace('<body', toc + '<body', 1)

    out = DOCS / nb_path.with_suffix('.html').name
    out.write_text(body, encoding='utf-8')
    print(f'wrote {out} ({len(body) / 1e6:.1f} MB, {toc.count("<li")} TOC entries)')


if __name__ == '__main__':
    DOCS.mkdir(exist_ok=True)
    for p in sys.argv[1:]:
        export(p)
