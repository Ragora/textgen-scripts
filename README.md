# Test

```bash
conda create --name aitextgentest python==3.10 --file requirements.txt
```

```bash
conda activate aitextgentest
git clone git@github.com:Ragora/aitextgen.git
cd aitextgen
pip install .
```

## Notes

Deepspeed does not work (looks like a Linux only package, so might work there).