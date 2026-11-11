import secrets


def new_id(nbytes: int = 8) -> str:
    """Random lowercase hex id (16 chars by default)."""
    return secrets.token_hex(nbytes)
