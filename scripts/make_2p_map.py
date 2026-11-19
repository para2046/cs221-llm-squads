"""Make <map>_2p.SC2Map: a copy of a SMAC map that two API clients can join (model vs model).

Two edits, each a same-length byte change:
  * MapInfo: player 2's control type 4 -> 1 (a user slot, as in Blizzard's 2-player melee
    maps). Without it a second client gets "There is no room for another user in this game".
  * MapScript.galaxy: SMAC's "Attack" trigger orders every idle player-2 unit to attack the
    nearest player-1 unit every 0.5 s (this, not the built-in AI, is what makes SMAC enemies
    attack). We point it at player 9, who has no units, so team B only does what its model says.

SMAC maps are MPQ v4 archives with several integrity checks, so no table may change:
  * each edited file is recompressed to exactly its old compressed size (searching zlib settings),
  * the MD5 of each raw 16 KiB chunk stored after a file's data is updated,
  * the CRC32/MD5 of each edited file in (attributes) is updated, plus that file's chunk MD5.
Block/hash/HET/BET tables and the header (with their MD5s) stay byte-identical.

usage: .venv-smac/bin/python scripts/make_2p_map.py external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map
"""
import hashlib
import itertools
import shutil
import struct
import sys
import zlib

import mpyq

EDITS = {  # file -> (old bytes, new bytes); each must occur exactly once
    "MapInfo": (b"\x02\x04\x00\x00\x00\x02", b"\x02\x01\x00\x00\x00\x02"),
    "MapScript.galaxy": (b"UnitGroup(null, 2, RegionEntireMap()", b"UnitGroup(null, 9, RegionEntireMap()"),
}

src = sys.argv[1]
dst = src.replace(".SC2Map", "_2p.SC2Map")
a = mpyq.MPQArchive(src)
raw = open(src, "rb").read()
chunk = struct.unpack_from("<I", raw, 108)[0]  # v4 header: dwRawChunkSize
assert a.header["format_version"] == 3 and chunk > 0, "expected an MPQ v4 archive with chunk MD5s"


def entry(name):
    i = a.get_hash_table_entry(name).block_table_index
    off, arch, size, _flags = a.block_table[i]
    return i, off, arch, size


def chunk_md5s(data):
    return b"".join(hashlib.md5(data[k:k + chunk]).digest() for k in range(0, len(data), chunk))


def same_size_zlib(data, target):
    """A zlib stream of `data` that is exactly `target` bytes long, or None."""
    for lvl, mem, strat, wbits in itertools.product(range(1, 10), range(1, 10), range(5), range(15, 8, -1)):
        c = zlib.compressobj(lvl, zlib.DEFLATED, wbits, mem, strat)
        z = c.compress(data) + c.flush()
        if len(z) == target:
            return z
    return None


ai, a_off, a_arch, a_size = entry("(attributes)")
attrs = bytearray(a.read_file("(attributes)"))
n = len(a.block_table)
assert a_arch == a_size and struct.unpack("<II", attrs[:8])[1] == 0x5 and len(attrs) == 8 + 20 * n

writes = []  # (offset, bytes)
patched = {}
for name, (old, new) in EDITS.items():
    d = a.read_file(name)
    assert d.count(old) == 1 and len(old) == len(new), name
    p = d.replace(old, new)
    i, off, arch, size = entry(name)
    assert raw[off] == 2, f"{name} is expected to be zlib-compressed"
    assert raw[off + arch:off + arch + 16] == chunk_md5s(raw[off:off + arch])[:16], "unexpected layout"
    z = same_size_zlib(p, arch - 1)
    assert z, f"no zlib setting reproduces the compressed size of {name}; try another edit"
    stored = b"\x02" + z
    writes.append((off, stored + chunk_md5s(stored)))
    assert struct.unpack_from("<I", attrs, 8 + 4 * i)[0] == zlib.crc32(d)
    struct.pack_into("<I", attrs, 8 + 4 * i, zlib.crc32(p))
    attrs[8 + 4 * n + 16 * i:8 + 4 * n + 16 * (i + 1)] = hashlib.md5(p).digest()
    patched[name] = p
writes.append((a_off, bytes(attrs) + chunk_md5s(bytes(attrs))))

shutil.copy(src, dst)
with open(dst, "r+b") as f:
    for off, data in writes:
        f.seek(off)
        f.write(data)

b = mpyq.MPQArchive(dst)
for x in a.files:
    name = x.decode()
    want = patched.get(name, bytes(attrs) if name == "(attributes)" else a.read_file(name))
    assert b.read_file(name) == want, name
print("wrote", dst)
