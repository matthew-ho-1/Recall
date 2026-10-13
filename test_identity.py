import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from identity import set_identity_from_indexed_photo


class TestIdentitySetup(unittest.TestCase):

    def test_identity_setup_preserves_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "recall.db"
            identity_path = root / "identities" / "me.npy"
            face_path = root / "face.npy"

            np.save(face_path, np.array([3.0, 4.0]))

            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    "CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT)"
                )
                connection.execute(
                    "CREATE TABLE faces (id INTEGER PRIMARY KEY, "
                    "image_id INTEGER, embedding_path TEXT)"
                )
                connection.execute(
                    "INSERT INTO images VALUES (1, 'test.jpg')"
                )
                connection.execute(
                    "INSERT INTO faces VALUES (1, 1, ?)",
                    (str(face_path),),
                )
                connection.commit()
            finally:
                connection.close()

            for _ in range(3):
                set_identity_from_indexed_photo(
                    1,
                    database_path=database_path,
                    identity_path=identity_path,
                )

            np.testing.assert_allclose(
                np.load(identity_path),
                [0.6, 0.8],
            )

            self.assertTrue(
                (identity_path.parent / "me.backup-1.npy").exists()
            )
            self.assertTrue(
                (identity_path.parent / "me.backup-2.npy").exists()
            )


if __name__ == "__main__":
    unittest.main()