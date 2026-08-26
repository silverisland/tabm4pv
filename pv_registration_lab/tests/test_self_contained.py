import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SelfContainedLabTests(unittest.TestCase):
    def test_model_pipeline_sources_are_bundled(self):
        required = [
            ROOT / "pipelines" / "tabm4pv.py",
            ROOT / "pipelines" / "registered_tabm4pv.py",
            ROOT / "pipelines" / "pv_curve_registration.py",
            ROOT / "requirements.txt",
        ]
        self.assertTrue(all(path.is_file() for path in required))

    def test_example_project_root_is_the_lab(self):
        config = json.loads(
            (ROOT / "config.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["project_root"], ".")

    def test_lab_python_does_not_import_parent_project_modules(self):
        for directory in (
            ROOT / "adapters",
            ROOT / "editable",
            ROOT / "protected",
            ROOT / "pvreglab",
        ):
            for path in directory.glob("*.py"):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("from registered_tabm4pv", source)
                self.assertNotIn("from pv_curve_registration", source)


if __name__ == "__main__":
    unittest.main()
