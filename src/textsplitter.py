import re
from typing import List

class SimpleTextSplitter:
    """A very simple text splitter: splits by paragraph and length limit.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # Maximum number of characters per chunk
        self.chunk_size = chunk_size
        # Number of overlapping characters between consecutive chunks
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # Split text into paragraphs by empty lines
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks = []  # Initialize list to store final chunks
        for p in paras:
            if len(p) <= self.chunk_size:
                # Paragraph is short enough; add as a single chunk
                chunks.append(p)
            else:
                # Paragraph too long; split by sentences or line breaks
                tokens = re.split(r"(?<=。|\.|!|\?|\\n)", p)
                cur = ""  # Temporary storage for the current chunk
                for t in tokens:
                    if len(cur) + len(t) <= self.chunk_size:
                        # Append sentence to current chunk if size permits
                        cur += t
                    else:
                        if cur:
                            # Save current chunk to list
                            chunks.append(cur.strip())
                        # Start new chunk, keep overlap from previous chunk
                        cur = (cur[-self.chunk_overlap:] if self.chunk_overlap < len(cur) else cur) + t
                if cur:
                    # Add remaining text as final chunk
                    chunks.append(cur.strip())
        # Remove empty strings before returning
        return [c for c in chunks if c]
