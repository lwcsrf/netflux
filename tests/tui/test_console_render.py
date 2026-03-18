import unittest

from ...core import TokenBill
from ...providers import Provider
from ...tui import ConsoleRender


class TestConsoleRender(unittest.TestCase):
    def test_total_token_bill_uses_compact_k_suffixes(self) -> None:
        rendered = ConsoleRender._format_total_token_bill(
            {
                Provider.Gemini: TokenBill(
                    input_tokens_cache_read=91234,
                    input_tokens_regular=131499,
                    output_tokens_total=6499,
                )
            }
        )

        self.assertEqual(rendered, "g31pp[CR:91k Reg:131k Out:6.5k]")

    def test_cache_write_uses_thousand_rounding(self) -> None:
        rendered = ConsoleRender._format_token_bill_fields(
            TokenBill(input_tokens_cache_write=1501)
        )

        self.assertEqual(rendered, "CW:2k")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
