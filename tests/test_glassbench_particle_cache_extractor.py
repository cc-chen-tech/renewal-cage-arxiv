import hashlib
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

from glassbench_particle_cache import extract_first_npz_positions_cache_from_xz_bytes  # noqa: E402


class GlassBenchParticleCacheExtractorTests(unittest.TestCase):
    def _fixture_xz_bytes(self) -> tuple[bytes, str, str]:
        positions = np.arange(24, dtype=float).reshape(3, 4, 2)
        box = np.array(10.0)
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, positions=positions, box=box)
        npz_bytes = npz_buffer.getvalue()
        member = "T0.99/test/example.npz"

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
            info = tarfile.TarInfo(member)
            info.size = len(npz_bytes)
            archive.addfile(info, io.BytesIO(npz_bytes))
        return lzma.compress(tar_buffer.getvalue()), member, hashlib.md5(npz_bytes).hexdigest()

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
            self.assertEqual(str(cached["source_path"]), "GlassBench/fixture.tar.xz")

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


if __name__ == "__main__":
    unittest.main()
