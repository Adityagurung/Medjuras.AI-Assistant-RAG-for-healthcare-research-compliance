from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

def cover_rect(img, box, color=(255, 255, 255)):
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, fill=color)
    return img

# agentic-rag-flow.png — crop bottom evaluation line + cover "(offline evaluation)"
agentic = Image.open(ROOT / "images/agentic-rag-flow.png").convert("RGB")
w, h = agentic.size
# Crop ~12% off bottom (evaluation footer line)
agentic = agentic.crop((0, 0, w, int(h * 0.88)))
w, h = agentic.size
# Cover lower-right "(offline evaluation)" label area
cover_rect(agentic, (int(w * 0.62), int(h * 0.78), w - 10, h - 10))
agentic.save(ROOT / "images/agentic-rag-flow.png")

# rag-flow-e2e.png — shrink legend by covering and redrawing smaller box
e2e = Image.open(ROOT / "images/rag-flow-e2e.png").convert("RGB")
w, h = e2e.size
# Cover existing legend (bottom-left quadrant — adjust if needed)
cover_rect(e2e, (int(w * 0.02), int(h * 0.72), int(w * 0.42), h - 8))
draw = ImageDraw.Draw(e2e)
# Smaller legend box
lx, ly = int(w * 0.04), int(h * 0.86)
draw.rectangle((lx, ly, lx + 120, ly + 52), outline=(0, 0, 0), width=1)
draw.text((lx + 6, ly + 6), "Legend", fill=(0, 0, 0))
e2e.save(ROOT / "images/rag-flow-e2e.png")

print("Images updated")