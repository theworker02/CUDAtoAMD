import json
import tempfile
import unittest
from pathlib import Path

from compat.analyzer import analyze
from compat.portplan import build_port_plan, render_port_plan


class PortPlanTests(unittest.TestCase):
    def _sample_report(self) -> dict:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "kernel.cu").write_text(
                '#include <cuda_runtime.h>\n'
                "void f(){ cudaMalloc(0, 0); cudaIpcOpenMemHandle(0,0,0); "
                'k<<<1, 1>>>(); asm("bar"); }\n',
                encoding="utf-8",
            )
            (root / "blob.cubin").write_bytes(b"\x7fELF\x00\x00")
            return analyze(root)

    def test_groups_prioritize_blockers_first(self):
        plan = build_port_plan(self._sample_report())
        statuses = [group["status"] for group in plan["groups"]]
        self.assertIn("UNSUPPORTED", statuses)
        self.assertIn("NVIDIA_SPECIFIC", statuses)
        self.assertIn("DIRECT", statuses)
        self.assertLess(statuses.index("UNSUPPORTED"), statuses.index("DIRECT"))
        self.assertLess(statuses.index("NVIDIA_SPECIFIC"), statuses.index("DIRECT"))
        self.assertEqual(plan["kind"], "port_plan")
        self.assertTrue(plan["caveats"])

    def test_per_file_actions_and_effort(self):
        plan = build_port_plan(self._sample_report())
        files = {item["file"]: item for item in plan["per_file"]}
        self.assertIn("kernel.cu", files)
        kernel = files["kernel.cu"]
        self.assertEqual(kernel["effort_hint"], "high")
        self.assertTrue(kernel["recommended_actions"])
        self.assertIn("UNSUPPORTED", kernel["statuses"])
        self.assertIn("blob.cubin", files)
        self.assertEqual(files["blob.cubin"]["effort_hint"], "high")

    def test_render_contains_priority_and_caveats(self):
        plan = build_port_plan(self._sample_report())
        text = render_port_plan(plan)
        self.assertIn("Prioritized groups", text)
        self.assertIn("UNSUPPORTED", text)
        self.assertIn("Caveats:", text)
        self.assertIn("not automatic migration", text.lower().replace("—", "-"))

    def test_json_serializable(self):
        plan = build_port_plan(self._sample_report())
        payload = json.dumps(plan)
        restored = json.loads(payload)
        self.assertEqual(restored["schema_version"], "1.0")
        self.assertEqual(len(restored["per_file"]), len(plan["per_file"]))


if __name__ == "__main__":
    unittest.main()
