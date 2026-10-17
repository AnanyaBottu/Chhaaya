"""Escalation cases (design section 5.4)."""

from sqlalchemy import Connection, text


def open_case(conn: Connection, cause: str, patient_id: int, asha_id: int) -> int:
    """Open a numbered case for the patient's ASHA and return its number."""
    return conn.execute(
        text(
            "INSERT INTO cases (cause, patient_id, asha_id) "
            "VALUES (:cause, :patient_id, :asha_id) RETURNING number"
        ),
        {"cause": cause, "patient_id": patient_id, "asha_id": asha_id},
    ).scalar_one()
