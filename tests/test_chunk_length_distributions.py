import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from figures.rq1_chunk_length_distributions import (
    read_confirmatory_chunk_lengths,
    write_rq1_chunk_length_distributions,
)


class ChunkLengthDistributionTests(unittest.TestCase):
    def test_reads_list_and_wrapped_chunk_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixed = write_chunks(root / "fixed.json", [64, 128], wrapped=False)
            semantic = write_chunks(root / "semantic.json", [32, 96], wrapped=True)
            hierarchical = write_chunks(
                root / "hierarchical.json", [16, 48], wrapped=False
            )

            values = read_confirmatory_chunk_lengths(
                fixed_size_path=fixed,
                semantic_path=semantic,
                hierarchical_path=hierarchical,
            )

            self.assertEqual(values["fixed_size"], [64, 128])
            self.assertEqual(values["semantic"], [32, 96])
            self.assertEqual(values["hierarchical"], [16, 48])

    def test_writes_pdf_and_png_with_family_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixed = write_chunks(root / "fixed.json", [256, 256, 128])
            semantic = write_chunks(root / "semantic.json", [32, 64, 128])
            hierarchical = write_chunks(root / "hierarchical.json", [8, 64, 512])

            summary = write_rq1_chunk_length_distributions(
                output_path=root / "chunk_lengths",
                fixed_size_path=fixed,
                semantic_path=semantic,
                hierarchical_path=hierarchical,
            )

            self.assertEqual(
                summary["configurations"]["fixed_size"]["median_tokens"],
                256.0,
            )
            self.assertEqual(
                summary["configurations"]["hierarchical"]["maximum_tokens"],
                512,
            )
            self.assertEqual(len(summary["written"]), 2)
            self.assertTrue(all(Path(path).is_file() for path in summary["written"]))


def write_chunks(
    path: Path,
    token_counts: list[int],
    *,
    wrapped: bool = False,
) -> Path:
    chunks = [{"token_count": value} for value in token_counts]
    payload = {"chunks": chunks} if wrapped else chunks
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
