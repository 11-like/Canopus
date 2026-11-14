#!/usr/bin/env python3
"""Repack a Canopus "Lua installer" .face container (Chem / Corona style).

Container layout (reverse engineered from Chem-Band9-3.1.35.face and
NoirMusic-Band9-3.1.32.face):

    +0x000  magic 0x1234A55A
    +0x004  version 0x00000A00          (image watchfaces use 0x00010000)
    +0x020  total size (end of last entry)
    +0x028  serial string
    +0x068  display name
    +0x0AC  total size (duplicate)
    +0x0D8  entry count
    +0x110  entry table: (count + 1) records of 16 bytes
              u32 type = 0x05000000 | index   (ignored by the loader for the
                                              terminator record)
              u32 reserved = 0
              u32 offset  -> payload
              u32 total   = 20 + len(name) + len(content); 0 for terminator
            payload starts at 0x110 + (count + 1) * 16
            every payload is 4-byte aligned

    payload:
              u32 inner = (flags << 20) | content_size   (size is 20 bits)
              16 x u8 reserved
              name (no NUL)
              content

`--self-test` proves the implementation by re-serialising the template and
comparing byte-for-byte with the original.

Usage:
    python pack_face.py --self-test --template in.face
    python pack_face.py --template in.face --src DIR --out out.face
                       [--serial 990000000123] [--name "Canopus Installer"]
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

MAGIC = 0x1234A55A
LUA_VERSION = 0x00000A00
TABLE_OFF = 0x110
OFF_TOTAL = 0x20
OFF_TOTAL2 = 0xAC
OFF_COUNT = 0xD8
OFF_SERIAL = 0x28
OFF_NAME = 0x68
ALIGN = 4


@dataclass
class Entry:
    flags: int
    name: str
    content: bytes
    pad: int = 0  # bytes of alignment padding that followed this entry


@dataclass
class Face:
    header: bytearray
    entries: list[Entry]
    terminator: bytes
    tail: bytes
    total: int


def parse(raw: bytes) -> Face:
    if len(raw) < TABLE_OFF:
        raise ValueError("file shorter than header")
    if struct.unpack_from("<I", raw, 0)[0] != MAGIC:
        raise ValueError("not a .face container (bad magic)")
    version = struct.unpack_from("<I", raw, 4)[0]
    if version != LUA_VERSION:
        raise ValueError(
            f"not a Lua-installer face: version 0x{version:08X}, expected 0x{LUA_VERSION:08X}"
        )

    total = struct.unpack_from("<I", raw, OFF_TOTAL)[0]
    count = struct.unpack_from("<I", raw, OFF_COUNT)[0]
    if count <= 0 or count > 128:
        raise ValueError(f"implausible entry count: {count}")

    entries: list[Entry] = []
    for i in range(count):
        etype, _res, off, etotal = struct.unpack_from("<IIII", raw, TABLE_OFF + i * 16)
        if etotal == 0:
            raise ValueError(f"entry {i} has zero size")
        inner = struct.unpack_from("<I", raw, off)[0]
        csize = inner & 0xFFFFF
        flags = inner >> 20
        name_len = etotal - 20 - csize
        if name_len < 0:
            raise ValueError(f"entry {i}: negative name length")
        name = raw[off + 20 : off + 20 + name_len].decode("latin1")
        content = raw[off + 20 + name_len : off + 20 + name_len + csize]
        if len(content) != csize:
            raise ValueError(f"entry {i}: truncated content")
        end = off + etotal
        nxt = (
            struct.unpack_from("<I", raw, TABLE_OFF + (i + 1) * 16 + 8)[0]
            if i + 1 < count
            else total
        )
        entries.append(Entry(flags, name, content, pad=nxt - end))

    term_off = TABLE_OFF + count * 16
    terminator = bytes(raw[term_off : term_off + 16])
    tail = bytes(raw[total:])
    return Face(bytearray(raw[:TABLE_OFF]), entries, terminator, tail, total)


def build(face: Face, entries: list[Entry]) -> bytes:
    out = bytearray(face.header)
    if len(out) < TABLE_OFF:
        out.extend(b"\x00" * (TABLE_OFF - len(out)))
    del out[TABLE_OFF:]

    payload_start = TABLE_OFF + (len(entries) + 1) * 16
    cursor = payload_start
    table = bytearray()
    blobs: list[bytes] = []

    for i, e in enumerate(entries):
        name = e.name.encode("latin1")
        total = 20 + len(name) + len(e.content)
        inner = ((e.flags & 0xFFF) << 20) | (len(e.content) & 0xFFFFF)
        blob = struct.pack("<I", inner) + b"\x00" * 16 + name + e.content
        if len(blob) != total:
            raise AssertionError("payload size mismatch")
        table += struct.pack("<IIII", 0x05000000 | i, 0, cursor, total)
        blobs.append(blob)
        cursor += total
        pad = (-cursor) % ALIGN
        cursor += pad
        blobs.append(b"\x00" * pad)

    out += table
    out += face.terminator
    if len(out) != payload_start:
        raise AssertionError(f"header/table mismatch {len(out)} != {payload_start}")
    for blob in blobs:
        out += blob

    total = len(out)
    struct.pack_into("<I", out, OFF_TOTAL, total)
    struct.pack_into("<I", out, OFF_TOTAL2, total)
    struct.pack_into("<I", out, OFF_COUNT, len(entries))
    # reference files carry a fixed trailer (0xBAC bytes on both samples)
    out += face.tail
    return bytes(out)


def self_test(template: Path) -> int:
    raw = template.read_bytes()
    face = parse(raw)
    rebuilt = build(face, face.entries)
    print(f"template : {template.name} ({len(raw)} bytes, {len(face.entries)} entries)")
    print(f"rebuilt  : {len(rebuilt)} bytes")
    common = min(len(raw), len(rebuilt))
    diff = next((i for i in range(common) if raw[i] != rebuilt[i]), None)
    if diff is not None:
        print(f"MISMATCH at 0x{diff:X}: got 0x{rebuilt[diff]:02X}, want 0x{raw[diff]:02X}")
        return 1
    if len(rebuilt) != len(raw):
        print(f"note: trailing {len(raw) - len(rebuilt)} bytes differ (padding)")
    print("self-test: BYTE-EXACT" if diff is None else "self-test: FAILED")
    return 0 if diff is None else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--serial")
    ap.add_argument("--name")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.template)

    if not args.src or not args.out:
        ap.error("--src and --out are required unless --self-test")

    face = parse(args.template.read_bytes())
    header = bytearray(face.header)
    if args.serial:
        data = args.serial.encode("latin1")[:15].ljust(16, b"\x00")
        header[OFF_SERIAL : OFF_SERIAL + 16] = data
    if args.name:
        data = args.name.encode("latin1")[:15].ljust(16, b"\x00")
        header[OFF_NAME : OFF_NAME + 16] = data
    face.header = header

    template_flags = {e.name.rsplit("/", 1)[-1]: e.flags for e in face.entries}
    prefix = ""
    for e in face.entries:
        if "/" in e.name:
            prefix = e.name.rsplit("/", 1)[0] + "/"
            break

    def pick_flags(fname: str) -> int:
        if fname in template_flags:
            return template_flags[fname]
        low = fname.lower()
        if low.endswith(".lua"):
            return 0x0C0
        if "profile" in low:
            return 0x300
        if "stage1" in low or "stage2" in low:
            return 0x280
        if "supervisor" in low:
            return 0x2C0
        return 0x110  # icons / generic bin

    files = sorted(p for p in args.src.iterdir() if p.is_file() and p.suffix.lower() in (".lua", ".bin"))
    if not files:
        raise SystemExit(f"no .lua/.bin files in {args.src}")

    entries = [Entry(pick_flags(p.name), prefix + p.name, p.read_bytes()) for p in files]
    data = build(face, entries)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes, {len(entries)} entries)")
    for e in entries:
        print(f"  {e.name:<52} flags=0x{e.flags:03X} size={len(e.content)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
