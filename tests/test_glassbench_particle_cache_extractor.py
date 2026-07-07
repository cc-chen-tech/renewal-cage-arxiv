import hashlib
import http.client
import io
import lzma
import tarfile
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from glassbench_particle_cache import (  # noqa: E402
    build_real_multilag_particle_caches,
    extract_first_npz_positions_cache_from_xz_bytes,
    fetch_range_bytes,
)


class GlassBenchParticleCacheExtractorTests(unittest.TestCase):
    def _fixture_xz_bytes(self) -> tuple[bytes, str, str]:
        positions = np.arange(24, dtype=float).reshape(3, 4, 2)
        initial_positions = positions[0] - 0.25
        box = np.array(10.0)
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, positions=positions, initial_positions=initial_positions, box=box)
        npz_bytes = npz_buffer.getvalue()
        member = "T0.99/test/example.npz"

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
            info = tarfile.TarInfo(member)
            info.size = len(npz_bytes)
            archive.addfile(info, io.BytesIO(npz_bytes))
        return lzma.compress(tar_buffer.getvalue()), member, hashlib.md5(npz_bytes).hexdigest()

    def _multi_member_fixture_xz_bytes(self) -> tuple[bytes, dict[str, str]]:
        md5s = {}
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
            for code, offset in [("tc05", 0.0), ("tc10", 100.0)]:
                positions = np.arange(24, dtype=float).reshape(3, 4, 2) + offset
                npz_buffer = io.BytesIO()
                np.savez_compressed(npz_buffer, positions=positions, box=np.array(10.0))
                npz_bytes = npz_buffer.getvalue()
                member = f"T0.99/test/N1290T0.99_151_{code}.npz"
                md5s[member] = hashlib.md5(npz_bytes).hexdigest()
                info = tarfile.TarInfo(member)
                info.size = len(npz_bytes)
                archive.addfile(info, io.BytesIO(npz_bytes))
        return lzma.compress(tar_buffer.getvalue()), md5s

    def test_extract_first_npz_positions_cache_from_bounded_xz_bytes(self):
        xz_bytes, member, expected_md5 = self._fixture_xz_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "cache" / "positions.npz"
            manifest = extract_first_npz_positions_cache_from_xz_bytes(
                xz_bytes=xz_bytes,
                first_npz_member=member,
                expected_npz_md5=expected_md5,
                target_path=target,
                system_id="KA2D",
                temperature="0.99",
                source_path="GlassBench/fixture.tar.xz",
                compressed_probe_range_start=10,
                compressed_probe_range_end=20,
            )

            self.assertTrue(target.exists())
            self.assertEqual(manifest["cache_stage"], "particle_coordinate_cache_written")
            self.assertEqual(manifest["positions_shape"], "3x4x2")
            self.assertEqual(float(manifest["frame_count"]), 3.0)
            self.assertEqual(float(manifest["particle_count"]), 4.0)
            self.assertEqual(float(manifest["spatial_dimension"]), 2.0)
            self.assertEqual(manifest["npz_member_md5"], expected_md5)
            self.assertEqual(float(manifest["particle_resolved_positions_cached"]), 1.0)

            cached = np.load(target)
            np.testing.assert_allclose(cached["positions"], np.arange(24, dtype=float).reshape(3, 4, 2))
            np.testing.assert_allclose(cached["initial_positions"], np.arange(8, dtype=float).reshape(4, 2) - 0.25)
            self.assertEqual(str(cached["source_path"]), "GlassBench/fixture.tar.xz")
            self.assertEqual(manifest["initial_positions_shape"], "4x2")
            self.assertEqual(float(manifest["initial_reference_positions_cached"]), 1.0)

    def test_extract_first_npz_positions_cache_from_zip_deflated_xz_payload(self):
        xz_bytes, member, expected_md5 = self._fixture_xz_bytes()
        deflated_xz_bytes = zlib.compressobj(wbits=-15)
        payload = deflated_xz_bytes.compress(xz_bytes) + deflated_xz_bytes.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "cache" / "positions.npz"
            manifest = extract_first_npz_positions_cache_from_xz_bytes(
                xz_bytes=payload,
                first_npz_member=member,
                expected_npz_md5=expected_md5,
                target_path=target,
                system_id="KA2D",
                temperature="0.99",
                source_path="GlassBench/fixture.tar.xz",
                compressed_probe_range_start=10,
                compressed_probe_range_end=20,
                probe_encoding="zip_deflate_xz",
            )

            self.assertTrue(target.exists())
            self.assertEqual(manifest["cache_stage"], "particle_coordinate_cache_written")
            self.assertEqual(manifest["probe_encoding"], "zip_deflate_xz")
            self.assertEqual(float(manifest["particle_resolved_positions_cached"]), 1.0)

    def test_extract_first_npz_positions_cache_rejects_md5_mismatch(self):
        xz_bytes, member, _expected_md5 = self._fixture_xz_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "positions.npz"
            with self.assertRaisesRegex(ValueError, "NPZ md5 mismatch"):
                extract_first_npz_positions_cache_from_xz_bytes(
                    xz_bytes=xz_bytes,
                    first_npz_member=member,
                    expected_npz_md5="0" * 32,
                    target_path=target,
                    system_id="KA2D",
                    temperature="0.99",
                    source_path="GlassBench/fixture.tar.xz",
                    compressed_probe_range_start=10,
                    compressed_probe_range_end=20,
                )
            self.assertFalse(target.exists())

    def test_fetch_range_bytes_retries_incomplete_remote_reads(self):
        import glassbench_particle_cache

        class FakeResponse:
            def __init__(self, payload: bytes | None, error: Exception | None = None):
                self.payload = payload
                self.error = error

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                if self.error is not None:
                    raise self.error
                return self.payload

        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"abc", 3)
            return FakeResponse(b"def")

        original_urlopen = glassbench_particle_cache.urlopen
        try:
            glassbench_particle_cache.urlopen = fake_urlopen
            self.assertEqual(fetch_range_bytes("https://example.invalid/data", 10, 15), b"abcdef")
        finally:
            glassbench_particle_cache.urlopen = original_urlopen

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0].get_header("Range"), "bytes=10-15")
        self.assertEqual(calls[1][0].get_header("Range"), "bytes=13-15")

    def test_build_real_multilag_particle_caches_records_extracted_and_prefix_missing_targets(self):
        xz_bytes, md5s = self._multi_member_fixture_xz_bytes()
        deflater = zlib.compressobj(wbits=-15)
        payload = deflater.compress(xz_bytes) + deflater.flush()
        visible_member = "T0.99/test/N1290T0.99_151_tc05.npz"
        missing_member = "T0.99/test/N1290T0.99_151_tc10.npz"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_csv = root / "targets.csv"
            target_csv.write_text(
                "target_id,system_id,temperature,source_path,selected_structure_id,"
                "selected_time_codes,target_members,target_member_md5s,target_lag_times\n"
                "fixture,KA2D,0.99,GlassBench/fixture.tar.xz,151,"
                f"tc05;tc10,{visible_member};{missing_member},"
                f"{md5s[visible_member]};{md5s[missing_member]},0.1;1.1\n"
            )
            member_index = root / "member_index.json"
            member_index.write_text(
                """{
  "archive_url": "https://example.invalid/archive.zip",
  "entries": [
    {
      "path": "GlassBench/fixture.tar.xz",
      "compressed_probe_range_start": 10,
      "compressed_probe_range_end": 20,
      "compressed_probe_bytes": 11,
      "npz_members": [{"name": "T0.99/test/N1290T0.99_151_tc05.npz", "size_bytes": 1}]
    }
  ]
}"""
            )
            output_manifest = root / "manifest.csv"

            rows = build_real_multilag_particle_caches(
                target_csv=target_csv,
                member_index_manifest_path=member_index,
                output_manifest_path=output_manifest,
                output_root=root,
                range_fetcher=lambda url, start, end: payload,
            )

            by_code = {row["time_code"]: row for row in rows}
            self.assertEqual(by_code["tc05"]["cache_stage"], "multi_lag_particle_coordinate_cache_written")
            self.assertEqual(float(by_code["tc05"]["particle_resolved_positions_cached"]), 1.0)
            self.assertTrue((root / by_code["tc05"]["particle_cache_path"]).exists())
            self.assertEqual(by_code["tc10"]["cache_stage"], "multi_lag_target_outside_bounded_prefix")
            self.assertEqual(float(by_code["tc10"]["particle_resolved_positions_cached"]), 0.0)
            self.assertEqual(by_code["tc10"]["primary_blocker"], "member_not_in_bounded_prefix_index")
            self.assertTrue(output_manifest.exists())

    def test_build_real_multilag_particle_caches_can_extract_required_members_beyond_index(self):
        xz_bytes, md5s = self._multi_member_fixture_xz_bytes()
        deflater = zlib.compressobj(wbits=-15)
        payload = deflater.compress(xz_bytes) + deflater.flush()
        visible_member = "T0.99/test/N1290T0.99_151_tc05.npz"
        deeper_member = "T0.99/test/N1290T0.99_151_tc10.npz"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_csv = root / "targets.csv"
            target_csv.write_text(
                "target_id,system_id,temperature,source_path,selected_structure_id,"
                "selected_time_codes,target_members,target_member_md5s,target_lag_times\n"
                "fixture,KA2D,0.99,GlassBench/fixture.tar.xz,151,"
                f"tc05;tc10,{visible_member};{deeper_member},"
                f"{md5s[visible_member]};{md5s[deeper_member]},0.1;1.1\n"
            )
            member_index = root / "member_index.json"
            member_index.write_text(
                """{
  "archive_url": "https://example.invalid/archive.zip",
  "entries": [
    {
      "path": "GlassBench/fixture.tar.xz",
      "compressed_probe_range_start": 10,
      "compressed_probe_range_end": 20,
      "compressed_probe_bytes": 11,
      "required_members": [
        "T0.99/test/N1290T0.99_151_tc05.npz",
        "T0.99/test/N1290T0.99_151_tc10.npz"
      ],
      "npz_members": [{"name": "T0.99/test/N1290T0.99_151_tc05.npz", "size_bytes": 1}]
    }
  ]
}"""
            )

            rows = build_real_multilag_particle_caches(
                target_csv=target_csv,
                member_index_manifest_path=member_index,
                output_manifest_path=root / "manifest.csv",
                output_root=root,
                range_fetcher=lambda url, start, end: payload,
            )

            by_code = {row["time_code"]: row for row in rows}
            self.assertEqual(by_code["tc10"]["cache_stage"], "multi_lag_particle_coordinate_cache_written")
            self.assertEqual(float(by_code["tc10"]["member_in_bounded_prefix_index"]), 1.0)
            self.assertEqual(float(by_code["tc10"]["particle_resolved_positions_cached"]), 1.0)


if __name__ == "__main__":
    unittest.main()
