"""PhysGM-compatible supervised training entry point.

The implementation lives in train_pretrain.py; this alias preserves the
upstream `main.py` convention for launch scripts and torchrun.
"""

from train_pretrain import main


if __name__ == "__main__":
    main()
