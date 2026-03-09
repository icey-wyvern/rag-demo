import unittest

from api.config import Settings, validate_generation_settings, validate_ingest_settings


class ConfigTests(unittest.TestCase):
    def test_ingest_requires_openai_key(self):
        with self.assertRaises(ValueError):
            validate_ingest_settings(Settings(OPENAI_API_KEY=""))

    def test_generation_requires_openai_key(self):
        with self.assertRaises(ValueError):
            validate_generation_settings(Settings(OPENAI_API_KEY=""))


if __name__ == "__main__":
    unittest.main()
