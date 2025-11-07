from pathlib import Path
p=Path(r'C:/Users/projl/anaconda3/envs/wukong/Lib/site-packages/torchao/quantization/quant_primitives.py')
bak=p.with_suffix('.py.bak')
print('Backing up',p,'->',bak)
if not p.exists():
    print('Target file not found:',p)
else:
    bak.write_bytes(p.read_bytes())
    src=p.read_text(encoding='utf-8')
    needle='    INT7 = auto()\n'
    idx=src.find(needle)
    if idx==-1:
        print('Could not find insertion point; aborting')
    else:
        insert_point=idx+len(needle)
        compat='''

# --- BEGIN PATCH: compatibility for PyTorch versions without sub-byte dtypes ---
class _TorchAODPlaceholder:
    """Simple placeholder object used when PyTorch doesn't expose sub-byte dtypes yet."""
    def __init__(self, name):
        self._name = name
    def __repr__(self):
        return f"<torch.dtype placeholder {self._name}>"

# Create placeholder attributes on torch for int1..int7 and uint1..uint7 when missing.
for _prefix in ("int", "uint"):
    for _i in range(1, 8):
        _nm = f"{_prefix}{_i}"
        if not hasattr(torch, _nm):
            setattr(torch, _nm, _TorchAODPlaceholder(_nm))
# --- END PATCH ---
'''
        new=src[:insert_point]+compat+src[insert_point:]
        p.write_text(new,encoding='utf-8')
        print('Patched file written')
