import importlib
import os
import unittest

import vidx


class MplBackendTest(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.get("MPLBACKEND")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = self.saved

    def test_notebook_inline_backend_replaced(self):
        os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
        importlib.reload(vidx)
        self.assertEqual(os.environ["MPLBACKEND"], "Agg")

    def test_valid_backend_kept(self):
        os.environ["MPLBACKEND"] = "pdf"
        importlib.reload(vidx)
        self.assertEqual(os.environ["MPLBACKEND"], "pdf")


if __name__ == "__main__":
    unittest.main()
