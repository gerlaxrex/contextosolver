import pathlib as pl

ROOT_PATH = pl.Path.cwd().parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

print(DATA_PATH)
