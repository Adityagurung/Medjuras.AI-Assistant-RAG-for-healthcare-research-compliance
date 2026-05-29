import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(document, max_tokens=1000, overlap=200):
    """
    Simple chunker: split by headings/blank lines then merge small parts.
    # target chunk size in characters (or tokens)
    # overlap to maintain context between chunks
    # The splitter will try to split by double newline (paragraph), then newline, then space, then as last resort character.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )
    chunks = text_splitter.split_text(document)
    return chunks
