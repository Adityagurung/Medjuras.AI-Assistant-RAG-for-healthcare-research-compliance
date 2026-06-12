import base64, pathlib, sys
pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode(pathlib.Path(sys.argv[2]).read_text()))
