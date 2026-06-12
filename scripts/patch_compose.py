from pathlib import Path
p=Path(__file__).resolve().parents[1]/"docker-compose.yaml"
t=p.read_text()
if "ollama_medi_rag" in t:
    print("skip")
    raise SystemExit(0)
img=chr(111)+chr(108)+chr(108)+chr(97)+chr(109)+chr(97)+chr(47)+chr(111)+chr(108)+chr(108)+chr(97)+chr(109)+chr(97)+chr(58)+chr(108)+chr(97)+chr(116)+chr(101)+chr(115)+chr(116)
import pathlib
base=pathlib.Path(__file__).resolve().parent.parent
fp=base / ("docker"+"-compose.yaml")
text=fp.read_text()
if "ollama_medi_rag" in text:
    raise SystemExit(0)
vol="./data/"+"ollama:"+"/root/.ollama"
img="ollama/ollama:latest"
lines=["","  ollama:","    image: "+img,",    container_name: ollama_medi_rag","    ports:","      - \"11434:11434\"","    volumes:","      - "+vol,",    restart: unless-stopped","","  qdrant:"]
block=chr(10).join(lines)+chr(10)
text=text.replace("  qdrant:", block, 1)
