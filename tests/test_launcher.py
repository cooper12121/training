"""Exercise the actual shell payload without allocating GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "script" / "train.sh"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="merlin check ")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.capture = root / "argv.json"
        fake = root / "fake python"
        fake.write_text(f"#!{sys.executable}\n" +
                        "import json, os, sys\n" +
                        "with open(os.environ['CAPTURE'], 'w') as f:\n" +
                        " json.dump({'args': sys.argv[1:], 'devices': os.getenv('CUDA_VISIBLE_DEVICES')}, f)\n" +
                        "sys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n")
        fake.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'LANG')}
        self.env.update(PYTHON_BIN=str(fake), CAPTURE=str(self.capture),
                        MODEL_PATH="/models/model with spaces", DATA_PATH="/data/train.json",
                        OUTPUT_DIR=str(root / "output"), CUDA_VISIBLE_DEVICES="2,5")

    def run_script(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], env=self.env,
                              cwd=self.tmp.name, text=True, capture_output=True)

    def captured(self):
        return json.loads(self.capture.read_text())

    def test_default_uses_allocated_gpus_and_preserves_paths(self):
        self.assertEqual(self.run_script().returncode, 0)
        actual = self.captured()
        self.assertEqual(actual['devices'], '2,5')
        self.assertIn('--standalone', actual['args'])
        self.assertIn('/models/model with spaces', actual['args'])
        self.assertNotIn('--max_steps', actual['args'])
        self.assertEqual(actual['args'][actual['args'].index('--nproc_per_node') + 1], 'gpu')

    def test_debug_overrides_steps_and_disables_save(self):
        self.assertEqual(self.run_script('--debug', '--max_steps', '999').returncode, 0)
        self.assertEqual(self.captured()['args'][-6:],
                         ['--max_steps', '5', '--save_strategy', 'no', '--skip_final_save', 'True'])
        self.env['DEBUG_STEPS'] = '-1'
        self.assertNotEqual(self.run_script('--debug').returncode, 0)

    def test_multi_node_requires_explicit_topology(self):
        self.env.update(NNODES='2')
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse(self.capture.exists())
        self.env.update(NODE_RANK='1', MASTER_ADDR='chief', MASTER_PORT='23456')
        self.assertEqual(self.run_script().returncode, 0)
        args = self.captured()['args']
        self.assertNotIn('--standalone', args)
        self.assertEqual(args[args.index('--node_rank') + 1], '1')
        self.assertEqual(args[args.index('--master_addr') + 1], 'chief')

    def test_process_mode_and_failure_propagation(self):
        self.env.update(LAUNCH_MODE='process', FAKE_EXIT='7')
        self.assertEqual(self.run_script().returncode, 7)
        self.assertEqual(self.captured()['args'][0], str(SCRIPT.parents[1] / 'train' / 'train.py'))
        self.assertNotIn('torch.distributed.run', self.captured()['args'])

    def test_configuration_in_script_needs_no_exported_paths(self):
        configured = Path(self.tmp.name) / 'configured.sh'
        script = SCRIPT.read_text()
        for key, value in [('MODEL_PATH', '/models/from script'),
                           ('DATA_PATH', '/data/from script.json'),
                           ('OUTPUT_DIR', '/output/from script')]:
            script = script.replace('${' + key + ':-}', '${' + key + ':-' + value + '}')
            self.env.pop(key)
        script = script.replace('${TRAIN_MODE:-train}', '${TRAIN_MODE:-debug}')
        script = script.replace('${DEBUG_STEPS:-5}', '${DEBUG_STEPS:-3}')
        script = script.replace('${LEARNING_RATE:-2e-5}', '${LEARNING_RATE:-1e-4}')
        configured.write_text(script)
        result = subprocess.run(['bash', str(configured)], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.captured()['args']
        self.assertEqual(args[args.index('--model_name_or_path') + 1], '/models/from script')
        self.assertEqual(args[args.index('--learning_rate') + 1], '1e-4')
        self.assertEqual(args[args.index('--max_steps') + 1], '3')

    def test_missing_paths_fail_before_launch(self):
        del self.env['MODEL_PATH']
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse(self.capture.exists())


if __name__ == '__main__':
    unittest.main()
