"""G4, kernel names against a reference capture, on synthetic events."""

from __future__ import annotations

import json
import random

import pytest

from gitm.optimizer import witness
from gitm.optimizer.witness import Header, account
from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, Trace

MS = 1_000_000
CUPTI = Header(source="cupti", device_count=1, vendor="nvidia")
NOT_ID = "not_identifiable_from_this_data"


def K(start, dur, name, *, dev=0, pid=None):
    return KernelEvent(start_ns=start, end_ns=start + dur, stream_id=0, device_id=dev, name=name,
                       pid=pid)


def capture(names, *, rounds=400, dur=MS, seed=0, dev=0, jitter=0.1):
    """``rounds`` passes over ``names``, one kernel each with a 1 ms gap; ~0.8 s per 400 kernels."""
    rng = random.Random(seed)
    ev, t = [], 0
    for _ in range(rounds):
        for name in names:
            d = int(dur * rng.uniform(1 - jitter, 1 + jitter))
            ev.append(K(t, d, name, dev=dev))
            t += d + MS
    return ev


def one(report, row_id):
    (row,) = [r for r in report.results if r["id"] == row_id]
    return row


# ── G4 ───────────────────────────────────────────────────────────────────────

BASE = ["gemm_fp16", "attn_fused", "rms_norm", "silu"]


def ref(events, header=CUPTI):
    return (header, events)


def test_g4_without_a_reference_is_channel_missing():
    r = one(account(capture(BASE), CUPTI), "G4")
    assert (r["state"], r["reason"]) == ("cannot_tell", "channel_missing")
    assert "--reference" in r["next"] and r["assumptions"] == []
    assert r["scope"] == "device" and r["partition"] == {"pid": None, "device_id": 0}


def test_g4_fires_on_a_kernel_name_that_changed_and_is_unpriced():
    changed = ["gemm_fp32" if n == "gemm_fp16" else n for n in BASE]
    r = one(account(capture(changed), CUPTI, reference=ref(capture(BASE, seed=1))), "G4")
    assert r["state"] == "fired"
    assert r["only_in_capture"] == ["gemm_fp32"] and r["only_in_reference"] == ["gemm_fp16"]
    assert r["observation"] == {"kind": "names", "value": 2, "unit": "kernel names",
                                "names": ["gemm_fp32", "gemm_fp16"]}
    assert r["seconds"] is None and r["price_reason"].startswith("unpriced: needs ")
    assert (r["status"], r["status_reason"]) == (NOT_ID, "unpriced")
    assert r["assumptions"] == ["reference_asserted_by_caller"]
    # Each member keeps its own levers; the group cannot say which member applies.
    assert {m["member"] for m in r["candidate_levers"]} == {"13", "14", "15"}
    for member in ("13", "14", "15"):
        assert one(account(capture(changed), CUPTI, reference=ref(capture(BASE))),
                   member)["reason"] == "undivided_group"


def test_g4_same_kernels_at_another_load_do_not_fire():
    # Twice the rounds, and a batch-dependent kernel under the 0.5% launch floor.
    rare = [K(10_000 * MS + i * 3 * MS, MS, "pad_to_batch") for i in range(5)]
    r = one(account(capture(BASE) + rare, CUPTI, reference=ref(capture(BASE, rounds=800))),
            "G4")
    assert r["state"] == "not_fired" and r["observation"]["value"] == 0
    assert (r["status"], r["status_reason"]) == (NOT_ID, "structural")
    assert r["assumptions"] == ["reference_asserted_by_caller"]


def with_new(n_base, n_new):
    base = capture(["a"], rounds=n_base)
    t = max(e.end_ns for e in base) + MS
    return base + [K(t + i * 3 * MS, MS, "new") for i in range(n_new)]


def test_g4_launch_share_floor_is_inclusive_at_half_a_percent():
    reference = capture(["a"], rounds=4000, seed=1)
    at_floor = one(account(with_new(995, 5), CUPTI, reference=ref(reference)), "G4")  # 5 of 1000
    assert at_floor["only_in_capture"] == ["new"] and at_floor["state"] == "fired"
    below = one(account(with_new(995, 4), CUPTI, reference=ref(reference)), "G4")     # 4 of 999
    assert (below["state"], below["names_too_rare_to_judge"]) == ("not_fired", 0)


def test_g4_lists_a_missing_name_only_when_its_absence_is_improbable():
    # "new" has 10 of the capture's 1000 launches and none of the reference's. Were it
    # equally common on both sides, that split has probability (1000 / (1000 + n)) ** 10:
    # 1.7% against 500 reference launches, 0.1% against 1000. Three names were examined
    # ("a" on each side, "new"), so each must clear 1% / 3.
    mine = with_new(990, 10)
    small = one(account(mine, CUPTI, reference=ref(capture(["a"], rounds=500, dur=3 * MS))), "G4")
    assert (small["state"], small["only_in_capture"]) == ("not_fired", [])
    assert (small["names_too_rare_to_judge"], small["too_rare_examples"]) == (1, ["new"])
    assert small["launches"] == {"capture": 1000, "reference": 500}
    assert small["names_examined"] == 3
    enough = one(account(mine, CUPTI, reference=ref(capture(["a"], rounds=1000, dur=3 * MS))),
                 "G4")
    assert (enough["state"], enough["only_in_capture"]) == ("fired", ["new"])
    assert enough["names_too_rare_to_judge"] == 0
    # The same from the reference's side.
    swapped = one(account(capture(["a"], rounds=500, dur=3 * MS), CUPTI, reference=ref(mine)),
                  "G4")
    assert (swapped["only_in_reference"], swapped["names_too_rare_to_judge"],
            swapped["too_rare_examples"]) == ([], 1, ["new"])


def test_g4_shares_its_level_across_every_name_examined():
    # Against 800 reference launches the split has probability 0.28%: under 1% / 3 when
    # three names were examined, over 1% / 43 when forty-three were.
    few = one(account(with_new(990, 10), CUPTI,
                      reference=ref(capture(["a"], rounds=800, dur=3 * MS))), "G4")
    assert few["only_in_capture"] == ["new"] and few["names_examined"] == 3
    common = [f"c{i}" for i in range(21)]
    mine = capture(common, rounds=47, dur=3 * MS)      # 987 launches of 21 common names
    t = max(e.end_ns for e in mine) + MS
    mine += [K(t + i * 3 * MS, MS, "new") for i in range(10)]
    many = one(account(mine, CUPTI, reference=ref(capture(common, rounds=38, dur=3 * MS))), "G4")
    assert many["names_examined"] == 43
    assert (many["state"], many["names_too_rare_to_judge"]) == ("not_fired", 1)


def test_g4_does_not_fire_between_two_small_samples_of_one_kernel_mix():
    # 20 common names and 8 rare ones at 0.5% to 1%, drawn independently for each side.
    # A fixed 200-launch minimum fired on 95% of such pairs.
    names = [f"common{i}" for i in range(20)] + [f"rare{i}" for i in range(8)]
    weights = [1.0] * 20 + [0.1 + 0.02 * i for i in range(8)]

    def sample(n, seed):
        rng = random.Random(seed)
        return [K(i * 6 * MS, 5 * MS, rng.choices(names, weights)[0]) for i in range(n)]

    for seed in range(20):
        for n_mine, n_theirs in ((200, 200), (200, 3000), (500, 3000)):
            r = one(account(sample(n_mine, seed), CUPTI,
                            reference=ref(sample(n_theirs, 1000 + seed))), "G4")
            assert r["state"] == "not_fired", (seed, n_mine, n_theirs, r["observation"])


@pytest.mark.parametrize("other, field", [
    (Header(source="cupti", device_count=1, vendor="amd"), "vendor"),
    (Header(source="cupti", device_count=2, vendor="nvidia"), "device_count"),
    (Header(source="nsys-import", device_count=1, vendor="nvidia"), "source"),
])
def test_g4_header_mismatch_is_an_invalid_capture(other, field):
    r = one(account(capture(BASE), CUPTI, reference=ref(capture(BASE), other)), "G4")
    assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")
    assert field in r["what"] and r["assumptions"] == ["reference_asserted_by_caller"]


def test_g4_unusable_short_or_devicless_references():
    mine = capture(BASE)
    short = one(account(mine, CUPTI, reference=ref(capture(BASE, rounds=20))), "G4")
    assert short["reason"] == "too_short" and "reference" in short["what"]
    invalid = one(account(mine, CUPTI, reference=ref([])), "G4")
    assert invalid["reason"] == "invalid_capture" and "no events" in invalid["why"]
    # A device only one side has is reported on both sides, not dropped.
    rows = {r["partition"]["device_id"]: r for r in account(
        mine, CUPTI, reference=ref(capture(BASE, dev=3))).results if r["id"] == "G4"}
    assert set(rows) == {0, 3}
    assert "reference has no kernels" in rows[0]["what"]
    assert "capture has no kernels" in rows[3]["what"]
    assert {r["reason"] for r in rows.values()} == {"channel_missing"}
    for r in (short, invalid, *rows.values()):
        assert r["assumptions"] == ["reference_asserted_by_caller"]
    unreadable = one(account(mine, CUPTI, _reference={"problem": "cannot read REF/trace.jsonl"}), "G4")
    assert unreadable["reason"] == "channel_missing" and "REF/trace.jsonl" in unreadable["why"]
    # The capture's own short window is named as the capture's.
    assert "capture" in one(account(capture(BASE, rounds=20), CUPTI, reference=ref(mine)),
                            "G4")["what"]


def test_g4_compares_per_device_with_pids_pooled():
    mine = capture(BASE, dev=0) + capture(["gemm_fp32", "rms_norm"], dev=1)
    theirs = capture(BASE, dev=0, seed=1) + capture(["gemm_fp16", "rms_norm"], dev=1, seed=1)
    rep = account(mine, Header(source="cupti", device_count=2, vendor="nvidia"),
                  reference=ref(theirs, Header(source="cupti", device_count=2, vendor="nvidia")))
    by_dev = {r["partition"]["device_id"]: r for r in rep.results if r["id"] == "G4"}
    assert by_dev[0]["state"] == "not_fired" and by_dev[1]["state"] == "fired"
    assert rep.run_presence["G4"] == "fired"


def write_run(path, events, manifest=None):
    path.mkdir(parents=True, exist_ok=True)
    write_trace_jsonl(path / "trace.jsonl", Trace(
        workload_id="w", fingerprint="f", run_id="r", device_count=1, vendor="nvidia",
        captured_at_ns=0, duration_ns=max(e.end_ns for e in events), events=events))
    if manifest is not None:
        (path / "run_manifest.json").write_text(json.dumps(manifest))
    return path


def test_reference_flag_end_to_end_with_manifest_warnings(tmp_path, capsys):
    changed = ["gemm_fp32" if n == "gemm_fp16" else n for n in BASE]
    run_dir = write_run(tmp_path / "run", capture(changed),
                        {"model": "m", "engine_version": "0.9", "only_here": 1,
                         "target": {"pid": 1}})
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1),
                        {"model": "m", "engine_version": "0.8", "target": {"pid": 2}})
    assert witness.main([str(run_dir), "--reference", str(ref_dir)]) == 0
    out = capsys.readouterr().out
    assert "G4        fired" in out and "gemm_fp32" in out
    assert "kernel names differ from the reference" in out
    data = json.loads((run_dir / "witness.json").read_text())
    g4 = next(r for r in data["results"] if r["id"] == "G4")
    assert g4["state"] == "fired"
    # A manifest difference is a warning, not a state change; keys in one file only are ignored.
    # A manifest difference is named, and only a few known scalars are ever quoted.
    assert g4["reference_warnings"] == ["run_manifest.json engine_version differs"]
    assert "note: reference differs: run_manifest.json engine_version differs" in out
    assert data["notes"] == ["reference differs: " + g4["reference_warnings"][0]]
    assert f"reference {ref_dir / 'trace.jsonl'}" in data["inputs"]["found"]


def test_a_missing_or_truncated_reference_is_a_state_not_an_exit_code(tmp_path, capsys):
    run_dir = write_run(tmp_path / "run", capture(BASE))
    assert witness.main([str(run_dir), "--reference", str(tmp_path / "nowhere")]) == 0
    g4 = one(witness.run(run_dir, tmp_path / "nowhere"), "G4")
    assert (g4["state"], g4["reason"]) == ("cannot_tell", "channel_missing")
    assert "nowhere" in g4["why"]
    ref_dir = write_run(tmp_path / "ref", capture(BASE))
    trace = ref_dir / "trace.jsonl"
    trace.write_text(trace.read_text()[:-40])
    g4 = one(witness.run(run_dir, ref_dir), "G4")
    assert (g4["state"], g4["reason"]) == ("cannot_tell", "channel_missing")
    assert "line" in g4["why"]
    capsys.readouterr()


# ── G4 abstains where it has no basis ────────────────────────────────────────


def test_g4_needs_a_one_second_window_on_each_side():
    quick = capture(BASE, rounds=100, dur=MS)  # 400 launches in 0.8 s
    long = capture(BASE, rounds=300)
    for mine, theirs, label in ((quick, long, "capture"), (long, quick, "reference")):
        r = one(account(mine, CUPTI, reference=ref(theirs)), "G4")
        assert (r["state"], r["reason"]) == ("cannot_tell", "too_short")
        assert f"the {label}'s device window is" in r["what"]


def test_g4_two_pids_on_a_device_are_pooled():
    mine = [e.model_copy(update={"pid": 1}) for e in capture(BASE[:2])]
    mine += [e.model_copy(update={"pid": 2, "stream_id": 5}) for e in capture(BASE[2:], seed=2)]
    r = one(account(mine, CUPTI, reference=ref(capture(BASE, seed=1))), "G4")
    assert r["state"] == "not_fired"
    # A stream overlap in one of the two pids makes the device's comparison unusable.
    broken = mine + [K(500, 900, BASE[0], pid=1)]
    r = one(account(broken, CUPTI, reference=ref(capture(BASE, seed=1))), "G4")
    assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")


def test_g4_abstains_on_an_inconsistent_reference_or_an_unusable_capture():
    bad_ref = capture(BASE) + [K(500, 900, BASE[0])]
    r = one(account(capture(BASE), CUPTI, reference=ref(bad_ref)), "G4")
    assert (r["reason"], r["assumptions"]) == ("invalid_capture", ["reference_asserted_by_caller"])
    r = one(account([], CUPTI, reference=ref(capture(BASE))), "G4")
    assert (r["reason"], r["assumptions"]) == ("invalid_capture", ["reference_asserted_by_caller"])
    assert r["partition"]["all"] is True


def test_the_capture_directory_is_not_a_reference_for_itself(tmp_path):
    run_dir = write_run(tmp_path / "run", capture(BASE))
    g4 = one(witness.run(run_dir, run_dir), "G4")
    assert (g4["state"], g4["reason"]) == ("cannot_tell", "channel_missing")
    assert "capture directory itself" in g4["why"]
    assert f"reference {run_dir / 'trace.jsonl'}" in witness.run(run_dir, run_dir).inputs["missing"]
    (tmp_path / "link").symlink_to(run_dir)
    assert one(witness.run(run_dir, tmp_path / "link"), "G4")["reason"] == "channel_missing"


HEADER = {"source": "cupti", "device_count": 1, "vendor": "nvidia"}


def set_header(run_dir, header):
    lines = (run_dir / "trace.jsonl").read_text().splitlines()
    lines[0] = json.dumps({"_header": header})
    (run_dir / "trace.jsonl").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("side", ["run", "ref"])
@pytest.mark.parametrize("field, value", [("device_count", "absent"), ("vendor", "absent"),
                                          ("vendor", None), ("vendor", "")])
def test_a_header_field_one_file_does_not_state_is_not_agreement(tmp_path, side, field, value):
    dirs = {"run": write_run(tmp_path / "run", capture(BASE)),
            "ref": write_run(tmp_path / "ref", capture(BASE, seed=1))}
    for name, d in dirs.items():
        header = dict(HEADER)
        if name == side:
            if value == "absent":
                del header[field]
            else:
                header[field] = value
        set_header(d, header)
    g4 = one(witness.run(dirs["run"], dirs["ref"]), "G4")
    assert (g4["state"], g4["reason"]) == ("cannot_tell", "invalid_capture")
    assert f"does not state {field}" in g4["what"]


def test_a_trace_that_does_not_state_its_source_is_refused(tmp_path):
    run_dir = write_run(tmp_path / "run", capture(BASE))
    set_header(run_dir, {"device_count": 1, "vendor": "nvidia"})
    with pytest.raises(witness.WitnessInputError, match="does not state its source"):
        witness.run(run_dir)
    # As a reference, that is a state on the row.
    good = write_run(tmp_path / "good", capture(BASE))
    g4 = one(witness.run(good, run_dir), "G4")
    assert g4["reason"] == "channel_missing" and "source" in g4["why"]


@pytest.mark.parametrize("manifest", ["{not json", "[1]", '"text"', "[" * 60_000 + "]" * 60_000])
def test_a_malformed_manifest_is_said_and_is_no_exception(tmp_path, manifest):
    run_dir = write_run(tmp_path / "run", capture(BASE), {"model": "m"})
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1))
    (ref_dir / "run_manifest.json").write_text(manifest)
    g4 = one(witness.run(run_dir, ref_dir), "G4")
    assert g4["state"] == "not_fired" and len(g4["reference_warnings"]) == 1
    assert "not compared" in g4["reference_warnings"][0]


def test_a_command_line_secret_never_reaches_the_witness_output(tmp_path, capsys):
    # The key is the first flag, so no truncation can hide it.
    argv = ["vllm", "serve", "m", "--api-key", "sk-LIVE", "--port", "8000"]
    run_dir = write_run(tmp_path / "run", capture(BASE), {
        "serve_argv": argv, "target": {"pid": 1, "cmdline": argv}, "served_model": "a",
        "load": {"token": "sk-LIVE"}})
    other = [*argv[:4], "sk-OTHER", "--port", "9"]
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1), {
        "serve_argv": other, "target": {"pid": 2, "cmdline": other}, "served_model": "b",
        "load": {"token": "sk-OTHER"}})
    assert witness.main([str(run_dir), "--reference", str(ref_dir)]) == 0
    out = capsys.readouterr().out
    written = (run_dir / "witness.json").read_text()
    for secret in ("sk-LIVE", "sk-OTHER"):
        assert secret not in out and secret not in written
    assert "run_manifest.json serve_argv differs" in out and "load differs" in out
    # A known scalar is still quoted.
    assert 'run_manifest.json served_model: "a" against "b"' in out


def test_manifest_warnings_skip_per_run_blocks_and_nan_equals_nan(tmp_path):
    def m(events):
        return {"model": "m", "trace": {"events": events}, "preflight": [events]}

    run_dir = write_run(tmp_path / "run", capture(BASE), m(1))
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1), m(2))
    for d in (run_dir, ref_dir):
        text = (d / "run_manifest.json").read_text()
        (d / "run_manifest.json").write_text(text.replace('"m"', "NaN"))
    assert one(witness.run(run_dir, ref_dir), "G4")["reference_warnings"] == []


def test_names_from_a_trace_cannot_forge_table_rows(tmp_path, capsys):
    evil = "\x1b[2Kevil\nrun              3         fired       FORGED \ud800"
    run_dir = write_run(tmp_path / "run", capture([*BASE, "plain"]))
    text = (run_dir / "trace.jsonl").read_text().replace('"plain"', json.dumps(evil))
    (run_dir / "trace.jsonl").write_text(text)
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1))
    assert witness.main([str(run_dir), "--reference", str(ref_dir)]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\\x1b[2Kevil\\nrun" in out
    assert not [line for line in out.splitlines() if line.startswith("run") and "FORGED" in line]
    assert json.loads((run_dir / "witness.json").read_text())["run_presence"]["G4"] == "fired"


def test_malformed_in_memory_events_are_an_invalid_capture_not_an_exception():
    class NoName:
        kind, start_ns, end_ns, stream_id, device_id = "kernel", 0, 5, 0, 0

    good = capture(BASE)
    t = max(e.end_ns for e in good) + MS  # after the capture, so nothing overlaps
    E = witness._Event
    bads = [NoName(), E("kernel", t, t + 5, 0, None, None, "k"), E("kernel", t, t + 5, 0, 0, None, 7),
            E("kernel", t, t + 5, "a", 0, None, "k"), E("kernel", t, t + 5, None, 0, None, "k"),
            E("kernel", t, t + 5, 0, 0, "x", "k"), E("kernel", t, t + 5, 0, 0, None, "k", "c"),
            E("kernel", t, None, 0, 0, None, "k"), E("kernel", t, t + 2.5, 0, 0, None, "k"),
            E("kernel", -2**70, 5, 0, 0, None, "k"), E("kernel", 2**63, 2**63 + 5, 0, 0, None, "k"),
            E("kernel", t, t + 5, True, 0, None, "k"), E("sync", t, t + 5, 0, 0, None, 7)]
    for bad in bads:
        r = one(account([*good, bad], CUPTI), "10")
        assert r["reason"] == "invalid_capture" and "wrong type" in r["why"], bad
        g4 = one(account(good, CUPTI, reference=(CUPTI, [*good, bad])), "G4")
        assert g4["reason"] == "invalid_capture", bad
    for odd in ((None, good), (CUPTI,), (CUPTI, 5), 7,
                (Header(source="cupti", device_count=None), good)):
        g4 = one(account(good, CUPTI, reference=odd), "G4")
        assert g4["reason"] == "channel_missing", odd
    r = one(account(good, Header(source="cupti", device_count=None)), "10")
    assert r["reason"] == "invalid_capture" and "device_count" in r["why"]
    json.dumps(account(good, CUPTI, reference=(CUPTI, good),
                       reference_warnings=[b"x", object()]).to_dict())


def test_a_partition_mixing_kernels_with_and_without_a_context_is_invalid():
    good = [e.model_copy(update={"context_id": 1}) for e in capture(BASE)]
    mixed = [*good[:-1], good[-1].model_copy(update={"context_id": None})]
    r = one(account(mixed, CUPTI), "10")
    assert r["reason"] == "invalid_capture" and "context id" in r["why"]
    assert one(account(good, CUPTI), "10")["state"] == "not_fired"


def test_context_survives_the_trace_file(tmp_path):
    a = [K(i * 2 * MS, MS, "k").model_copy(update={"context_id": 1}) for i in range(1000)]
    b = [K(i * 2 * MS + 500_000, MS, "k").model_copy(update={"context_id": 2})
         for i in range(1000)]
    assert one(witness.run(write_run(tmp_path / "two", a + b)), "10")["state"] == "not_fired"
    same = [e.model_copy(update={"context_id": 1}) for e in b]
    assert one(witness.run(write_run(tmp_path / "one", a + same)), "10")[
        "reason"] == "invalid_capture"


def test_manifest_bounds(tmp_path):
    run_dir = write_run(tmp_path / "run", capture(BASE), {f"k{i:02d}": "a" for i in range(30)})
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1),
                        {f"k{i:02d}": "b" * 5000 for i in range(30)})
    rep = witness.run(run_dir, ref_dir)
    warnings = one(rep, "G4")["reference_warnings"]
    assert len(warnings) == 21 and warnings[-1] == "run_manifest.json has 10 more differing keys"
    assert warnings[0] == "run_manifest.json k00 differs"
    assert max(len(line) for line in witness._render(rep).splitlines()) < 500
    (ref_dir / "run_manifest.json").write_text(json.dumps({"k00": "c", "pad": "x" * 1_100_000}))
    assert one(witness.run(run_dir, ref_dir), "G4")["reference_warnings"] == [
        "run_manifest.json is too large to compare"]


def test_an_unreadable_reference_is_listed_as_missing(tmp_path):
    run_dir = write_run(tmp_path / "run", capture(BASE), {"model": "a"})
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1), {"model": "b"})
    trace = ref_dir / "trace.jsonl"
    trace.write_text(trace.read_text()[:-40])
    rep = witness.run(run_dir, ref_dir)
    assert f"reference {trace}" in rep.inputs["missing"] and rep.notes == []


@pytest.mark.parametrize("line", [
    json.dumps({"kind": "kernel", "name": "k", "start_ns": 2**70, "end_ns": 5}),
    json.dumps({"kind": "kernel", "name": 7, "start_ns": 0, "end_ns": 5}),
    json.dumps({"kind": "kernel", "name": "x" * 1_100_000, "start_ns": 0, "end_ns": 5}),
])
def test_a_bad_event_line_exits_2(tmp_path, capsys, line):
    run_dir = write_run(tmp_path, capture(BASE))
    with (run_dir / "trace.jsonl").open("a") as fh:
        fh.write(line + "\n")
    assert witness.main([str(run_dir)]) == 2
    assert "did not run" in capsys.readouterr().err


def test_a_reference_that_is_not_a_regular_file_is_a_state(tmp_path):
    import os

    run_dir = write_run(tmp_path / "run", capture(BASE))
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    os.mkfifo(ref_dir / "trace.jsonl")
    g4 = one(witness.run(run_dir, ref_dir), "G4")
    assert g4["reason"] == "channel_missing" and "not a regular file" in g4["why"]


def test_a_different_served_model_is_not_compared(tmp_path, capsys):
    changed = ["gemm_fp32" if n == "gemm_fp16" else n for n in BASE]
    run_dir = write_run(tmp_path / "run", capture(changed), {"served_model": "a"})
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1), {"served_model": "b"})
    assert witness.main([str(run_dir), "--reference", str(ref_dir)]) == 0
    capsys.readouterr()
    rep = witness.run(run_dir, ref_dir)
    g4 = one(rep, "G4")
    assert (g4["state"], g4["reason"]) == ("cannot_tell", "invalid_capture")
    assert g4["what"] == "capture and reference differ in served_model"
    assert rep.run_presence["G4"] == "cannot_tell" and "candidate_levers" not in g4


@pytest.mark.parametrize("key", ["served_model", "window", "workload_id"])
def test_a_manifest_value_that_is_not_a_scalar_is_never_quoted(tmp_path, key):
    run_dir = write_run(tmp_path / "run", capture(BASE), {key: {"cmd": "--api-key sk-LIVE"}})
    ref_dir = write_run(tmp_path / "ref", capture(BASE, seed=1), {key: ["--api-key", "sk-OTHER"]})
    warnings, _ = witness._manifest_differences(run_dir, ref_dir)
    assert warnings == [f"run_manifest.json {key} differs"]


def test_a_zero_length_kernel_far_away_does_not_stretch_the_g4_window():
    quick, long = capture(BASE, rounds=100), capture(BASE, rounds=300)
    far = K(50_000 * MS, 0, BASE[0])
    for mine, theirs, label in ((quick + [far], long, "capture"),
                                (long, quick + [far], "reference")):
        r = one(account(mine, CUPTI, reference=ref(theirs)), "G4")
        assert (r["state"], r["reason"]) == ("cannot_tell", "too_short"), label
        assert f"the {label}'s device window is 0." in r["what"]
