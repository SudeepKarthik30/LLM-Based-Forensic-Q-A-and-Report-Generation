"""pdf_debug.py — step-by-step PDF renderer trace to find the failing section."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpdf import FPDF

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'output', 'pdf_debug.pdf')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def _safe(text):
    return str(text).encode("latin-1", errors="replace").decode("latin-1")

mock_history = [
    {"question": "Any failed logon?", "answer": "Yes [Source 1].",
     "sources": [
         {"num": 1, "source_file": "test.evtx", "event_id": "4625",
          "event_type": "Failed Logon", "timestamp": "2019-04-30T02:14:00Z",
          "hostname": "DC01"},
     ]},
]

from report_generator import _get_inventory_rows
rows = _get_inventory_rows(mock_history)

try:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=20, top=20, right=20)
    print("  [OK] FPDF init")

    W = 170
    # Page 1 — Title
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(W, 12, "FORENSIC INVESTIGATION REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    print("  [OK] Title cell")

    col_w = [75, 95]
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(col_w[0], 7, "Generated (UTC)", border=1, fill=False)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(col_w[1], 7, "2024-01-15T10:00:00Z", border=1, new_x="LMARGIN", new_y="NEXT")
    print("  [OK] Meta table")

    # Page 2 — Narrative
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(W, 10, "Narrative", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(W, 5, _safe("EXECUTIVE SUMMARY"))
    pdf.multi_cell(W, 5, _safe("Attacker conducted password spray then ran mimikatz."))
    print("  [OK] Narrative page")

    # Page 3 — Evidence Inventory
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(W, 10, "Evidence Inventory", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    cw = [8, 42, 14, 36, 36, 34]
    headers = ["#", "Source File", "EID", "Event Type", "Timestamp (UTC)", "Host"]
    print(f"  [OK] cw sum={sum(cw)}")

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(50, 50, 50)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(cw[i], 7, _safe(h), border=1, fill=True)
    pdf.ln()
    print("  [OK] Header row")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    max_chars = [4, 28, 8, 22, 22, 20]
    for row_idx, row in enumerate(rows):
        fill = row_idx % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)
        vals = [
            str(row["num"]),
            row["source_file"],
            row["event_id"],
            row["event_type"],
            row["timestamp"],
            row["hostname"],
        ]
        for i, v in enumerate(vals):
            pdf.cell(cw[i], 6, _safe(str(v))[:max_chars[i]], border=1, fill=fill)
        pdf.ln()
    print("  [OK] Inventory rows")

    # Page 4 — Q&A
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(W, 10, "Q&A Transcript", new_x="LMARGIN", new_y="NEXT")
    for idx, turn in enumerate(mock_history, 1):
        q = turn.get("question", "")
        a = turn.get("answer", "")
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(W, 6, _safe(f"Q{idx}: {q}"))
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(W, 5, _safe(a[:1200]))
        pdf.ln(3)
    print("  [OK] Q&A transcript")

    pdf.output(OUT)
    size = os.path.getsize(OUT)
    print(f"\n  SUCCESS  {size:,} bytes  ->  {OUT}")

except Exception as exc:
    traceback.print_exc()
    print(f"\n  FAILED: {exc}")
