import sqlite3
from pathlib import Path

# Open recall.db
def connect(database_path: Path) -> sqlite3.Connection: 
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sqlite3.connect(database_path)

#Make sure image table exists
def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            extension TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_at REAL NOT NULL,
            indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            width INTEGER,
            height INTEGER,
            format TEXT,
            thumbnail_path TEXT,
            processed_at TEXT,
            processing_error TEXT
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
def index_image(
    connection: sqlite3.Connection,
    path: Path,
) -> None:
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
            indexed_at = CURRENT_TIMESTAMP,
            processed_at = NULL,
            processing_error = NULL
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
    images: list[Path],
    root: Path,
) -> int:
    root = root.resolve()

    current_paths = {
        str(image.resolve())
        for image in images
    }

    cursor = connection.execute(
        """
        SELECT path, thumbnail_path
        FROM images
        """
    )

    deleted_images = []

    for row in cursor.fetchall():
        indexed_path = Path(row[0])
        thumbnail_path = row[1]

        # Only check images belonging to the directory
        # we're currently scanning.
        if indexed_path.is_relative_to(root):
            if str(indexed_path) not in current_paths:
                deleted_images.append(
                    (str(indexed_path), thumbnail_path)
                )

    for path, thumbnail_path in deleted_images:

        # Delete cached thumbnail if one exists.
        if thumbnail_path is not None:
            thumbnail = Path(thumbnail_path)

            if thumbnail.exists():
                thumbnail.unlink()

        # Delete database record.
        connection.execute(
            """
            DELETE FROM images
            WHERE path = ?
            """,
            (path,),
        )

    connection.commit()

    return len(deleted_images)

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