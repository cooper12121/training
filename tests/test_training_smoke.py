"""Offline CPU smoke test; run in the training image with its dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEPS = ('torch', 'transformers', 'accelerate', 'datasets', 'peft', 'pandas')


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in DEPS),
                     'requires the training Python dependencies')
class TrainingSmokeTests(unittest.TestCase):
    def test_debug_steps_and_formal_export(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory(prefix='training-smoke-') as folder:
            root = Path(folder)
            model_path = root / 'model'
            vocab = {word: i for i, word in enumerate(
                ['[UNK]', '[PAD]', '[EOS]', '[BOS]', 'user', 'assistant', 'hello', 'world'])}
            backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]', bos_token='[BOS]')
            tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}"
            tokenizer.save_pretrained(model_path)
            model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2, max_position_embeddings=128))
            model.save_pretrained(model_path)
            data = root / 'data.json'
            data.write_text(json.dumps([{'instruction': 'hello', 'output': 'world'}] * 4))
            for debug in (True, False):
                output = root / ('debug' if debug else 'formal')
                env = dict(os.environ, PYTHON_BIN=sys.executable, MODEL_PATH=str(model_path),
                    DATA_PATH=str(data), OUTPUT_DIR=str(output), LAUNCH_MODE='process',
                    TRAIN_MODE='train', BF16='False', MAX_LENGTH='64', DEBUG_STEPS='2',
                    MICRO_BATCH_SIZE='1', GRADIENT_ACCUMULATION_STEPS='1',
                    DEEPSPEED_CONFIG='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                    OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
                args = ['bash', str(ROOT / 'script/train.sh')]
                if debug:
                    # A stale checkpoint must not change the requested smoke steps.
                    (output / 'checkpoint-100').mkdir(parents=True)
                    args.append('--debug')
                else:
                    args += ['--do_eval', 'True', '--eval_path', str(data),
                             '--eval_strategy', 'steps', '--eval_steps', '1']
                args += ['--use_cpu', 'True', '--gradient_checkpointing', 'False',
                         '--dataloader_pin_memory', 'False', '--disable_tqdm', 'True',
                         '--max_steps', '1', '--save_strategy', 'no']
                result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('train_loss', result.stdout)
                if debug:
                    self.assertIn("'epoch': 0.5", result.stdout)
                    self.assertEqual([p.name for p in output.glob('checkpoint-*')], ['checkpoint-100'])
                    self.assertFalse((output / 'model.safetensors').exists())
                else:
                    self.assertIn('eval_loss', result.stdout)
                    self.assertTrue((output / 'model.safetensors').is_file())
                    saved = PreTrainedTokenizerFast.from_pretrained(output)
                    self.assertIn('<Retrieval>', saved.get_vocab())
                    state = json.loads((output / 'trainer_state.json').read_text())
                    self.assertEqual(state['global_step'], 1)


if __name__ == '__main__':
    unittest.main()
