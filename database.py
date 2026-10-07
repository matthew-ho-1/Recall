import sqlite3
from pathlib import Path
import numpy as np

# Open recall.db
def connect(
    database_path: Path,
) -> sqlite3.Connection:

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        database_path
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection

#Make sure image table exists
def initialize_database(
    connection: sqlite3.Connection,
) -> None:

    # --------------------------------------------------
    # Images
    # --------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            extension TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_at REAL NOT NULL,
            indexed_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP,

            width INTEGER,
            height INTEGER,
            format TEXT,
            thumbnail_path TEXT,
            processed_at TEXT,
            processing_error TEXT,

            embedding_path TEXT,
            embedded_at TEXT,

            faces_processed_at TEXT
        )
        """
    )

    # --------------------------------------------------
    # Temporary v0.5 migration
    #
    # Existing databases created before v0.5 will not
    # have faces_processed_at.
    # --------------------------------------------------

    try:
        connection.execute(
            """
            ALTER TABLE images
            ADD COLUMN faces_processed_at TEXT
            """
        )

    except sqlite3.OperationalError as error:
        if (
            "duplicate column name"
            not in str(error)
        ):
            raise

    # --------------------------------------------------
    # Faces
    # --------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,

            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,

            embedding_path TEXT NOT NULL,

            detected_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (image_id)
                REFERENCES images(id)
                ON DELETE CASCADE
        )
        """
    )

    # --------------------------------------------------
    # Evaluation relevance judgments
    # --------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS relevance_judgments (
            query TEXT NOT NULL,
            image_id INTEGER NOT NULL,

            relevant INTEGER NOT NULL
                CHECK (relevant IN (0, 1)),

            judged_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP,

            PRIMARY KEY (
                query,
                image_id
            ),

            FOREIGN KEY (image_id)
                REFERENCES images(id)
                ON DELETE CASCADE
        )
        """
    )

    connection.commit()

#Have I seen D:\Photos\foo.jpg before?
def get_image(
    connection: sqlite3.Connection,
    path: Path,
):
    cursor = connection.execute(
        """
        SELECT file_size, modified_at
        FROM images
        WHERE path = ?
        """,
        (str(path.resolve()),),
    )

    return cursor.fetchone()


# new / modified / unchanged
def get_image_status(
    connection: sqlite3.Connection,
    path: Path,
) -> str:
    existing = get_image(connection, path)

    if existing is None:
        return "new"

    existing_size, existing_modified_at = existing
    stat = path.stat()

    if (
        stat.st_size != existing_size
        or stat.st_mtime != existing_modified_at
    ):
        return "modified"

    return "unchanged"

#INSERT new photo or UPDATE existing photo
def index_image(connection, path):
    stat = path.stat()

    connection.execute(
        """
        INSERT INTO images (
            path,
            extension,
            file_size,
            modified_at
        )
        VALUES (?, ?, ?, ?)

        ON CONFLICT(path) DO UPDATE SET
            extension = excluded.extension,
            file_size = excluded.file_size,
            modified_at = excluded.modified_at,

            processed_at = NULL,
            processing_error = NULL,

            embedding_path = NULL,
            embedded_at = NULL,

            faces_processed_at = NULL,

            indexed_at = CURRENT_TIMESTAMP
        """,
        (
            str(path.resolve()),
            path.suffix.lower(),
            stat.st_size,
            stat.st_mtime,
        ),
    )

#Run that process across all discovered photos
def index_images(
    connection: sqlite3.Connection,
    images: list[Path],
) -> dict[str, int]:
    stats = {
        "new": 0,
        "modified": 0,
        "unchanged": 0,
    }

    for image in images:
        status = get_image_status(
            connection,
            image,
        )

        stats[status] += 1

        if status in ("new", "modified"):
            index_image(
                connection,
                image,
            )

    connection.commit()

    return stats

def remove_deleted_images(
    connection: sqlite3.Connection,
    current_paths: set[Path],
    scan_root: Path,
) -> int:

    scan_root = scan_root.resolve()

    current_resolved_paths = {
        path.resolve()
        for path in current_paths
    }

    cursor = connection.execute(
        """
        SELECT
            id,
            path,
            thumbnail_path,
            embedding_path
        FROM images
        """
    )

    rows = cursor.fetchall()

    deleted_count = 0

    for (
        image_id,
        indexed_path,
        thumbnail_path,
        embedding_path,
    ) in rows:

        indexed_file = Path(
            indexed_path
        ).resolve()

        # ----------------------------------------------
        # Only consider images belonging to the folder
        # currently being scanned.
        # ----------------------------------------------

        try:
            indexed_file.relative_to(
                scan_root
            )

        except ValueError:
            continue

        # ----------------------------------------------
        # If it was discovered during this scan,
        # it still exists.
        # ----------------------------------------------

        if (
            indexed_file
            in current_resolved_paths
        ):
            continue

        # ----------------------------------------------
        # Safety check:
        #
        # Don't delete DB state merely because the file
        # wasn't discovered for some unexpected reason.
        # Confirm that the original really is gone.
        # ----------------------------------------------

        if indexed_file.exists():
            continue

        # ----------------------------------------------
        # Delete Recall thumbnail
        # ----------------------------------------------

        if thumbnail_path:
            thumbnail_file = Path(
                thumbnail_path
            )

            if thumbnail_file.exists():
                thumbnail_file.unlink()

        # ----------------------------------------------
        # Delete Recall CLIP embedding
        # ----------------------------------------------

        if embedding_path:
            embedding_file = Path(
                embedding_path
            )

            if embedding_file.exists():
                embedding_file.unlink()

        # ----------------------------------------------
        # Delete face embedding files + face DB rows
        #
        # This MUST happen before deleting the image row.
        # ----------------------------------------------

        delete_faces_for_image(
            connection,
            image_id,
        )

        # ----------------------------------------------
        # Delete image DB row
        #
        # relevance_judgments rows will be removed by
        # ON DELETE CASCADE when foreign keys are enabled.
        # ----------------------------------------------

        connection.execute(
            """
            DELETE FROM images
            WHERE id = ?
            """,
            (image_id,),
        )

        deleted_count += 1

    connection.commit()

    return deleted_count

def update_image_metadata(
    connection: sqlite3.Connection,
    path: Path,
    metadata: dict,
    thumbnail_path: Path,
) -> None:
    connection.execute(
        """
        UPDATE images
        SET
            width = ?,
            height = ?,
            format = ?,
            thumbnail_path = ?,
            processed_at = CURRENT_TIMESTAMP,
            processing_error = NULL
        WHERE path = ?
        """,
        (
            metadata["width"],
            metadata["height"],
            metadata["format"],
            str(thumbnail_path),
            str(path.resolve()),
        ),
    )

def record_processing_error(
    connection: sqlite3.Connection,
    path: Path,
    error: str,
) -> None:
    connection.execute(
        """
        UPDATE images
        SET
            processing_error = ?,
            processed_at = CURRENT_TIMESTAMP
        WHERE path = ?
        """,
        (
            error,
            str(path.resolve()),
        ),
    )

def needs_processing(
    connection: sqlite3.Connection,
    path: Path,
) -> bool:
    cursor = connection.execute(
        """
        SELECT processed_at
        FROM images
        WHERE path = ?
        """,
        (str(path.resolve()),),
    )

    row = cursor.fetchone()

    if row is None:
        return True

    return row[0] is None

def get_image_id(
    connection: sqlite3.Connection,
    path: Path,
) -> int | None:
    cursor = connection.execute(
        """
        SELECT id
        FROM images
        WHERE path = ?
        """,
        (str(path.resolve()),),
    )

    row = cursor.fetchone()

    if row is None:
        return None

    return row[0]

def update_image_embedding(
    connection: sqlite3.Connection,
    path: Path,
    embedding_path: Path,
) -> None:
    connection.execute(
        """
        UPDATE images
        SET
            embedding_path = ?,
            embedded_at = CURRENT_TIMESTAMP
        WHERE path = ?
        """,
        (
            str(embedding_path),
            str(path.resolve()),
        ),
    )

def needs_embedding(
    connection: sqlite3.Connection,
    path: Path,
) -> bool:
    cursor = connection.execute(
        """
        SELECT embedded_at, embedding_path
        FROM images
        WHERE path = ?
        """,
        (str(path.resolve()),),
    )

    row = cursor.fetchone()

    if row is None:
        return True

    embedded_at, embedding_path = row

    if embedded_at is None:
        return True

    if embedding_path is None:
        return True

    if not Path(embedding_path).exists():
        return True

    return False


def get_searchable_images(
    connection: sqlite3.Connection,
) -> list[tuple]:

    cursor = connection.execute(
        """
        SELECT
            id,
            path,
            thumbnail_path,
            embedding_path
        FROM images
        WHERE embedding_path IS NOT NULL
          AND embedded_at IS NOT NULL
        """
    )

    return cursor.fetchall()

def insert_face(
    connection: sqlite3.Connection,
    image_id: int,
    bbox,
) -> int:

    x1, y1, x2, y2 = bbox

    cursor = connection.execute(
        """
        INSERT INTO faces (
            image_id,
            x1,
            y1,
            x2,
            y2,
            embedding_path
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            "",
        ),
    )

    return cursor.lastrowid

def update_face_embedding(
    connection: sqlite3.Connection,
    face_id: int,
    embedding_path: Path,
) -> None:

    connection.execute(
        """
        UPDATE faces
        SET embedding_path = ?
        WHERE id = ?
        """,
        (
            str(embedding_path),
            face_id,
        ),
    )

def needs_face_processing(
    connection: sqlite3.Connection,
    path: Path,
) -> bool:

    cursor = connection.execute(
        """
        SELECT faces_processed_at
        FROM images
        WHERE path = ?
        """,
        (str(path.resolve()),),
    )

    row = cursor.fetchone()

    if row is None:
        return True

    faces_processed_at = row[0]

    return faces_processed_at is None

def mark_faces_processed(
    connection: sqlite3.Connection,
    path: Path,
) -> None:

    connection.execute(
        """
        UPDATE images
        SET faces_processed_at = CURRENT_TIMESTAMP
        WHERE path = ?
        """,
        (str(path.resolve()),),
    )

def delete_faces_for_image(
    connection: sqlite3.Connection,
    image_id: int,
) -> None:

    cursor = connection.execute(
        """
        SELECT embedding_path
        FROM faces
        WHERE image_id = ?
        """,
        (image_id,),
    )

    rows = cursor.fetchall()

    for (embedding_path,) in rows:
        if embedding_path:
            path = Path(embedding_path)

            if path.exists():
                path.unlink()

    connection.execute(
        """
        DELETE FROM faces
        WHERE image_id = ?
        """,
        (image_id,),
    )

if __name__ == "__main__":
    connection = connect(
        Path(".recall/recall.db")
    )

    image_id = 26  # Replace with your actual ID

    fake_embedding = Path(
        ".recall/faces/test-face.npy"
    ).resolve()

    cursor = connection.execute(
        """
        INSERT INTO faces (
            image_id,
            x1,
            y1,
            x2,
            y2,
            embedding_path
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            10.0,
            20.0,
            100.0,
            120.0,
            str(fake_embedding),
        ),
    )

    face_id = cursor.lastrowid

    connection.commit()

    print(
        "Created fake face:",
        face_id,
    )

    print(
        "File exists:",
        fake_embedding.exists(),
    )

    delete_faces_for_image(
        connection,
        image_id,
    )

    connection.commit()

    print(
        "File exists after cleanup:",
        fake_embedding.exists(),
    )

    cursor = connection.execute(
        """
        SELECT COUNT(*)
        FROM faces
        WHERE image_id = ?
        """,
        (image_id,),
    )

    print(
        "Face rows after cleanup:",
        cursor.fetchone()[0],
    )

    connection.close()

def get_searchable_faces(
    connection: sqlite3.Connection,
) -> list[tuple]:
    cursor = connection.execute(
        """
        SELECT
            faces.id,
            faces.image_id,
            images.path,
            faces.x1,
            faces.y1,
            faces.x2,
            faces.y2,
            faces.embedding_path
        FROM faces
        JOIN images
            ON faces.image_id = images.id
        WHERE faces.embedding_path IS NOT NULL
          AND faces.embedding_path != ''
        """
    )

    return cursor.fetchall()

def get_relevance_judgment(
    connection: sqlite3.Connection,
    query: str,
    image_id: int,
) -> bool | None:

    cursor = connection.execute(
        """
        SELECT relevant
        FROM relevance_judgments
        WHERE query = ?
          AND image_id = ?
        """,
        (
            query,
            image_id,
        ),
    )

    row = cursor.fetchone()

    if row is None:
        return None

    return bool(row[0])

def save_relevance_judgment(
    connection: sqlite3.Connection,
    query: str,
    image_id: int,
    relevant: bool,
) -> None:

    connection.execute(
        """
        INSERT INTO relevance_judgments (
            query,
            image_id,
            relevant
        )
        VALUES (?, ?, ?)

        ON CONFLICT(query, image_id)
        DO UPDATE SET
            relevant = excluded.relevant,
            judged_at = CURRENT_TIMESTAMP
        """,
        (
            query,
            image_id,
            int(relevant),
        ),
    )

    connection.commit()