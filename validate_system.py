"""
validate_system.py — ARIA System Validation Script
Run this to verify all fixes are correctly applied before starting the dashboard.

Usage:
    python validate_system.py

Expected output: All 10 checks should show [OK]
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 65)
print("  ARIA System Validation")
print("=" * 65)

errors = []
warnings = []


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg, e):
    print(f"  [FAIL] {msg}: {e}")
    errors.append(msg)


def warn(msg):
    print(f"  [WARN] {msg}")
    warnings.append(msg)


# ─── Check 1: Tools package — all 8 tools ─────────────────────────────────────
try:
    from tools import ALL_TOOLS
    tool_names = [t.name for t in ALL_TOOLS]
    assert len(ALL_TOOLS) == 8, f"Expected 8 tools, got {len(ALL_TOOLS)}"
    assert "search_official_protocols" in tool_names, "search_official_protocols missing"
    assert "search_social_reports" in tool_names, "search_social_reports missing"
    ok(f"tools/__init__.py — {len(ALL_TOOLS)} tools: {', '.join(tool_names)}")
except Exception as e:
    fail("tools/__init__.py", e)

# ─── Check 2: Agent — correct LangGraph API kwarg ────────────────────────────
try:
    import inspect
    import langgraph
    try:
        import importlib.metadata
        lg_version = importlib.metadata.version("langgraph")
    except Exception:
        lg_version = getattr(langgraph, '__version__', 'unknown')
    from agents.disaster_agent import get_agent, SYSTEM_PROMPT
    src = inspect.getsource(get_agent)
    # Accept either kwarg depending on installed LangGraph version:
    #   state_modifier=  (LangGraph >=0.2.x)
    #   messages_modifier= (LangGraph 0.1.x)
    #   prompt=          (very old LangGraph)
    has_valid_kwarg = any(kw in src for kw in ("state_modifier", "messages_modifier", "prompt="))
    assert has_valid_kwarg, "No recognized system prompt kwarg in create_react_agent call"
    assert "lru_cache" in src, "lru_cache not found"
    assert "Temporal Context Bridge" in SYSTEM_PROMPT, "Temporal Bridge missing from prompt"
    # Check which kwarg is actually used
    used_kw = next(kw for kw in ("state_modifier", "messages_modifier", "prompt=") if kw in src)
    ok(f"agents/disaster_agent.py — LangGraph {lg_version}, using '{used_kw}', lru_cache, Temporal Bridge all present")
except Exception as e:
    fail("agents/disaster_agent.py", e)

# ─── Check 3: agents/__init__.py ─────────────────────────────────────────────
try:
    from agents import run_agent
    ok("agents/__init__.py — run_agent exported correctly")
except Exception as e:
    fail("agents/__init__.py", e)

# ─── Check 4: RAG chain — MMR fix ────────────────────────────────────────────
try:
    import inspect
    from core.rag_chain import get_retriever
    src = inspect.getsource(get_retriever)
    assert "fetch_k" in src, "fetch_k missing from MMR search_kwargs"
    ok("core/rag_chain.py — MMR fetch_k parameter present")
except Exception as e:
    fail("core/rag_chain.py", e)

# ─── Check 5: Knowledge loader — regex fix ───────────────────────────────────
try:
    import inspect
    from core.knowledge_loader import load_and_chunk_knowledge_base
    src = inspect.getsource(load_and_chunk_knowledge_base)
    # Check the ACTUAL separators list in code — not docstring comments.
    # The docstring MENTIONS the old regex for documentation purposes, that's fine.
    # We confirm the separators= line uses the fixed list, not the lookbehind.
    assert 'separators=["\\n\\n"' in src or "separators=[\"\\n\\n\"" in src or '"\\n\\n"' in src, \
        "separators list not found — file may not have been updated"
    # Verify the actual RecursiveCharacterTextSplitter call doesn't use lookbehind in the list literal
    import ast, textwrap
    # Simple check: (?<= should NOT appear on a line containing 'separators=['
    for line in src.splitlines():
        if 'separators=[' in line:
            assert '(?<=' not in line, f"Lookbehind regex found in separators list: {line.strip()}"
    ok("core/knowledge_loader.py — lookbehind regex removed from separators list")
except Exception as e:
    fail("core/knowledge_loader.py", e)

# ─── Check 6: Sensor tool — datetime fix ─────────────────────────────────────
try:
    import re, inspect
    import tools.sensor_tool as st_module
    full_src = inspect.getsource(st_module)
    # Strip ALL triple-quoted strings (docstrings/multi-line strings) before checking.
    # This prevents fix-notes like '- FIXED: datetime.utcfromtimestamp()' from matching.
    code_only = re.sub(r'""".*?"""', '', full_src, flags=re.DOTALL)
    code_only = re.sub(r"'''.*?'''", '', code_only, flags=re.DOTALL)
    code_only = re.sub(r'#.*$', '', code_only, flags=re.MULTILINE)
    assert "utcfromtimestamp" not in code_only, \
        "datetime.utcfromtimestamp() still in live code (not just docstring)"
    assert "CRITICAL: IMMEDIATE EVACUATION" in full_src, "CRITICAL alarm string missing"
    ok("tools/sensor_tool.py — deprecated datetime fixed, CRITICAL alarm string present")
except Exception as e:
    fail("tools/sensor_tool.py", e)

# ─── Check 7: Vision tool — structured output ────────────────────────────────
try:
    import inspect
    import tools.vision_tool as vt_module
    src = inspect.getsource(vt_module)
    assert "SUMMARY" in src, "Structured output SUMMARY not found"
    assert "DAMAGE" in src, "Structured output DAMAGE not found"
    assert "ACCESS" in src, "Structured output ACCESS not found"
    assert "_check_ollama_available" in src, "Ollama pre-check missing"
    ok("tools/vision_tool.py — structured output format and Ollama pre-check present")
except Exception as e:
    fail("tools/vision_tool.py", e)

# ─── Check 8: Drone loader — no mock data ────────────────────────────────────
try:
    import re, inspect
    import ingestion.drone_loader as dl_module
    src = inspect.getsource(dl_module)
    # Strip ALL triple-quoted docstrings before checking.
    # The module docstring says '- REMOVED: ... import random' as a fix note — that's fine.
    code_only = re.sub(r'""".*?"""', '', src, flags=re.DOTALL)
    code_only = re.sub(r"'''.*?'''", '', code_only, flags=re.DOTALL)
    # Also strip single-line comments
    code_only = re.sub(r'#.*$', '', code_only, flags=re.MULTILINE)
    assert "def generate_mock" not in code_only, "Mock generator function still defined in code"
    assert "import random" not in code_only, "import random still in live code (not just docstring)"
    assert "observations" in src or "vlm_description" in src, \
        "Real VLM field names not found — may be old file"
    ok("ingestion/drone_loader.py — mock data generator removed, real VLM path present")
except Exception as e:
    fail("ingestion/drone_loader.py", e)

# ─── Check 9: IoT loader — makedirs fix ──────────────────────────────────────
try:
    import inspect
    import ingestion.iot_loader as il_module
    src = inspect.getsource(il_module)
    assert "if parent_dir" in src or "parent := " in src, "makedirs guard missing"
    ok("ingestion/iot_loader.py — makedirs empty-string guard present")
except Exception as e:
    fail("ingestion/iot_loader.py", e)

# ─── Check 10: Evaluation — lambda capture fix ───────────────────────────────
try:
    import inspect
    import evaluation.run_evals as eval_module
    src = inspect.getsource(eval_module)
    assert "cur_chain" in src or "c=chain" in src, "Lambda capture fix not found"
    assert '"text"' in src and '"knowledge_base"' in src, "Modality case fix not found"
    assert '"Text"' not in src.split("CONFIGS")[1].split("LLM Only")[0], "Old uppercase 'Text' still in CONFIGS"
    ok("evaluation/run_evals.py — lambda capture bug fixed, modality strings normalized")
except Exception as e:
    fail("evaluation/run_evals.py", e)

# ─── Live API tests (optional, requires network) ─────────────────────────────
print()
print("  Live API Tests (requires internet):")

try:
    import requests
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 40.015, "longitude": -105.2705,
                "current": "temperature_2m", "timezone": "auto"},
        timeout=8
    )
    data = r.json()["current"]
    ok(f"Open-Meteo API — reachable, temp: {data['temperature_2m']}°C")
except Exception as e:
    warn(f"Open-Meteo API — {e}")

try:
    import requests
    r = requests.get(
        "https://waterservices.usgs.gov/nwis/iv/",
        params={"sites": "06730200", "parameterCd": "00065", "format": "json"},
        timeout=8
    )
    ts = r.json().get("value", {}).get("timeSeries", [])
    if ts:
        val = ts[0]["values"][0]["value"][-1]["value"]
        ok(f"USGS NWIS API — reachable, Boulder Creek: {val} ft")
    else:
        warn("USGS NWIS API — returned empty timeSeries")
except Exception as e:
    warn(f"USGS NWIS API — {e}")

try:
    import requests
    r = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params={"format": "geojson", "starttime": "2025-01-01", "minmagnitude": 5.0, "limit": 1},
        timeout=8
    )
    count = len(r.json().get("features", []))
    ok(f"USGS Earthquake API — reachable, {count} M≥5.0 event(s) found")
except Exception as e:
    warn(f"USGS Earthquake API — {e}")

try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=3)
    if r.status_code == 200:
        models = [m["name"] for m in r.json().get("models", [])]
        has_llava = any("llava" in m for m in models)
        ok(f"Ollama — running, models: {models[:3]}, LLaVA installed: {has_llava}")
    else:
        warn("Ollama — running but unexpected status code")
except Exception:
    warn("Ollama — not running (optional, needed only for image analysis)")

# ─── Final summary ────────────────────────────────────────────────────────────
print()
print("=" * 65)
if errors:
    print(f"  RESULT: {len(errors)} CRITICAL error(s), {len(warnings)} warning(s)")
    for e in errors:
        print(f"    ERROR: {e}")
    sys.exit(1)
else:
    print(f"  RESULT: All checks PASSED! {len(warnings)} optional warning(s).")
    print()
    print("  Next steps:")
    print("    1. python core/populate_db.py --reset   (populate vector DB)")
    print("    2. streamlit run app.py                 (start dashboard)")
print("=" * 65)
