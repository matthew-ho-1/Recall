
from dataclasses import dataclass


# Empirically calibrated using the current
# v0.6 labeled evaluation dataset.
#
# These are conservative technical-defect
# boundaries, not universal photography standards.

BLUR_EVIDENCE_THRESHOLD = 14.88
UNDEREXPOSURE_EVIDENCE_THRESHOLD = 89.44
OVEREXPOSURE_EVIDENCE_THRESHOLD = 156.20


@dataclass
class QualityEvidence:
    face_sharpness: float | None

    whole_mean_brightness: float
    face_mean_brightness: float | None

    megapixels: float
    face_prominence: float | None

    blur_evidence: bool
    underexposure_evidence: bool
    overexposure_evidence: bool

    technical_defect_count: int


def evaluate_quality(
    *,
    face_sharpness: float | None,
    whole_mean_brightness: float,
    face_mean_brightness: float | None,
    megapixels: float,
    face_prominence: float | None,
) -> QualityEvidence:
    """
    Evaluate technical photo-quality evidence.

    Flags three types of technical defects:
    - Strong evidence of blur
    - Strong evidence of underexposure
    - Strong evidence of overexposure

    Resolution and face prominence are retained
    as measurements, but do not currently
    contribute to defect counts.

    Missing face measurements are not treated
    as evidence of blur or overexposure.

    This function does not calculate an overall
    aesthetic-quality score.
    """

    # ----------------------------------------
    # Blur evidence
    # ----------------------------------------

    blur_evidence = (
        face_sharpness is not None
        and face_sharpness
        < BLUR_EVIDENCE_THRESHOLD
    )

    # ----------------------------------------
    # Underexposure evidence
    # ----------------------------------------

    underexposure_evidence = (
        whole_mean_brightness
        < UNDEREXPOSURE_EVIDENCE_THRESHOLD
    )

    # ----------------------------------------
    # Overexposure evidence
    # ----------------------------------------

    overexposure_evidence = (
        face_mean_brightness is not None
        and face_mean_brightness
        > OVEREXPOSURE_EVIDENCE_THRESHOLD
    )

    # ----------------------------------------
    # Count detected technical defects
    # ----------------------------------------

    technical_defect_count = sum(
        (
            blur_evidence,
            underexposure_evidence,
            overexposure_evidence,
        )
    )

    return QualityEvidence(
        face_sharpness=face_sharpness,
        whole_mean_brightness=(
            whole_mean_brightness
        ),
        face_mean_brightness=(
            face_mean_brightness
        ),
        megapixels=megapixels,
        face_prominence=face_prominence,
        blur_evidence=blur_evidence,
        underexposure_evidence=(
            underexposure_evidence
        ),
        overexposure_evidence=(
            overexposure_evidence
        ),
        technical_defect_count=(
            technical_defect_count
        ),
    )


def quality_tier(
    evidence: QualityEvidence,
) -> int:
    """
    Return a conservative technical-quality tier.

    Lower is better.

    Tier 0:
        No detected technical defects.

    Tier 1:
        One detected technical defect.

    Tier 2:
        Two or more detected technical defects.

    This is not an overall aesthetic-quality score.

    Photos within the same tier are considered
    tied by the current v0.6 quality policy.
    """

    return min(
        evidence.technical_defect_count,
        2,
    )
