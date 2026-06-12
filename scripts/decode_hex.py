import pathlib,sys
p=pathlib.Path(sys.argv[1])
p.write_bytes(bytes.fromhex(sys.stdin.read()))
