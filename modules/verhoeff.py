# ─────────────────────────────────────────────────────────
# Verhoeff Algorithm Implementation
# Used for validating numeric IDs such as Aadhaar numbers.
# It detects single-digit errors and transposition errors.
# If checksum result = 0 → ID is valid
# ─────────────────────────────────────────────────────────


# Multiplication table used in Verhoeff algorithm
D_TABLE = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0],
]


# Permutation table used during digit processing
P_TABLE = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8],
]


# Inverse table used to generate checksum digit
INV_TABLE = [0,4,3,2,1,5,6,7,8,9]


def generate_checksum(number: str) -> str:
    """
    Generate the Verhoeff checksum digit for a number.

    Args:
        number (str): Numeric string without checksum.

    Returns:
        str: The calculated checksum digit.
    """

    c = 0
    digits = list(reversed(number))

    # Process each digit through multiplication and permutation tables
    for i, d in enumerate(digits):
        c = D_TABLE[c][P_TABLE[(i + 1) % 8][int(d)]]

    # Return the final checksum digit
    return str(INV_TABLE[c])


def validate_checksum(number: str) -> bool:
    """
    Validate a number that already includes a Verhoeff checksum.

    Args:
        number (str): Full numeric string including checksum digit.

    Returns:
        bool: True if checksum is valid, False otherwise.
    """

    c = 0
    digits = list(reversed(number))

    # Apply Verhoeff validation process
    for i, d in enumerate(digits):
        c = D_TABLE[c][P_TABLE[i % 8][int(d)]]

    # If final result is 0 → valid number
    return c == 0


def check_id_integrity(document_id: str):
    """
    Validate a document ID using the Verhoeff checksum algorithm.

    Args:
        document_id (str): ID extracted from OCR text.

    Returns:
        dict: Validation result containing ID and status.
    """

    # Remove all non-digit characters
    digits_only = ''.join(filter(str.isdigit, document_id))

    # If ID length is too short, validation cannot be done
    if len(digits_only) < 2:
        return {
            "document_id": document_id,
            "is_valid": False,
            "verdict": "ID too short to validate"
        }

    # Validate checksum
    is_valid = validate_checksum(digits_only)

    # Generate human-readable result
    verdict = "Valid ID (checksum passed)" if is_valid else "Invalid ID (checksum failed)"

    return {
        "document_id": document_id,
        "is_valid": is_valid,
        "verdict": verdict
    }