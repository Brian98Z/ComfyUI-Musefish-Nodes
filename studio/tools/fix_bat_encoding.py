# -*- coding: utf-8 -*-
"""Rewrite start_server.bat as GBK (ANSI) so cmd.exe parses Chinese rem/echo
correctly on the default codepage (936). UTF-8 .bat files break cmd's parser:
the CP850/GBK decoder mangles multi-byte sequences and swallows line breaks,
which made the script exit instantly ('flash crash')."""
from pathlib import Path
import sys

# relative to repo root so the tool works on any machine, not just this one
root = Path(__file__).resolve().parent.parent
p = root / "start_server.bat"
if len(sys.argv) > 1:
    p = Path(sys.argv[1])
text = p.read_text("utf-8")
p.write_bytes(text.encode("gbk"))
print("rewritten as GBK:", p, len(text), "chars")
