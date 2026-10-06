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
            indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
        SELECT path
        FROM images
        """
    )

    deleted_paths = []

    for row in cursor.fetchall():
        indexed_path = Path(row[0])

        # Only consider files belonging to this scan root.
        if indexed_path.is_relative_to(root):
            if str(indexed_path) not in current_paths:
                deleted_paths.append(str(indexed_path))

    for path in deleted_paths:
        connection.execute(
            """
            DELETE FROM images
            WHERE path = ?
            """,
            (path,),
        )

    connection.commit()

    return len(deleted_paths)