import tempfile
import unittest
from pathlib import Path

from pvreglab.implementation import (
    implementation_hash,
    restore_implementation,
    snapshot_implementation,
)


class ImplementationVersionTests(unittest.TestCase):
    def test_snapshot_hash_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            editable = root / "editable"
            state = root / "state"
            editable.mkdir()
            source = editable / "algorithm.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            original_hash = implementation_hash(editable)
            snapshot_implementation(editable, state)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(implementation_hash(editable), original_hash)
            restored = restore_implementation(editable, state, original_hash)
            self.assertEqual(restored, original_hash)
            self.assertEqual(source.read_text(encoding="utf-8"), "VALUE = 1\n")


if __name__ == "__main__":
    unittest.main()
