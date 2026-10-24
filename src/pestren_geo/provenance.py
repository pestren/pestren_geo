"""Every number this library produces comes wrapped in how it was made.

WHY A MEASUREMENT IS NOT A FLOAT.

A slope of 4.2 degrees is not a fact. "4.2 degrees, computed by Horn's method on
a 10 m grid whose source was 10 m, from USGS 3DEP, on 2026-10-24" is a fact,
because a reader can disagree with it specifically. The float alone cannot be
checked, cannot be compared against another figure honestly, and cannot be cited.

This matters more than it sounds. The same ground measured on a 1 m grid and a
10 m grid gives genuinely different slopes — not two estimates of one truth, but
two different quantities. Publishing either as a bare number invites a comparison
that is not valid.

WHY THE LIBRARY REFUSES RATHER THAN CAVEATS.

A caveat nobody reads is worse than an absence. Where the support scale makes a
measurement meaningless, the functions here raise or return `Measurement.absent`
with a reason, instead of returning a number with a footnote. An absence is a
usable answer; a wrong number that looks right is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone


@dataclass(frozen=True)
class Source:
    """Where the underlying observation came from."""

    name: str
    url: str | None = None
    acquired: date | None = None
    licence: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.acquired:
            d["acquired"] = self.acquired.isoformat()
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class Measurement:
    """One number, with everything needed to argue with it.

    `method` names the algorithm, not the intent — "horn_slope" rather than
    "steepness", so two implementations can be compared.

    `support_scale_m` is the resolution of the SOURCE, which may be coarser than
    the grid the number was computed on. It is the field that most often reveals
    a measurement should not have been made.
    """

    value: float | None
    unit: str
    method: str
    support_scale_m: float | None = None
    gsd_m: float | None = None
    crs: str | None = None
    uncertainty: float | None = None
    sources: tuple[Source, ...] = ()
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    absent_because: str | None = None

    @classmethod
    def absent(cls, unit: str, method: str, because: str, **kw) -> "Measurement":
        """A refusal, which is a result and should be returned like one."""
        return cls(value=None, unit=unit, method=method, absent_because=because, **kw)

    @property
    def is_present(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        d = {
            "value": self.value,
            "unit": self.unit,
            "method": self.method,
            "support_scale_m": self.support_scale_m,
            "gsd_m": self.gsd_m,
            "crs": self.crs,
            "uncertainty": self.uncertainty,
            "computed_at": self.computed_at.isoformat(),
            "sources": [s.to_dict() for s in self.sources],
        }
        if self.absent_because:
            d["absent_because"] = self.absent_because
        return {k: v for k, v in d.items() if v not in (None, [], ())}

    def __str__(self) -> str:
        if not self.is_present:
            return f"<absent: {self.absent_because}>"
        at = f" @{self.support_scale_m:g}m" if self.support_scale_m else ""
        return f"{self.value:g} {self.unit}{at} ({self.method})"


@dataclass(frozen=True)
class Delta:
    """The difference between two measurements of the same thing.

    This exists as its own type because it is the shape of the question this
    library was built for: a record made at one time against ground observed at
    another. A `Delta` is never an assertion that either side is wrong — both
    may be correct observations of different moments, and usually are.
    """

    a: Measurement
    b: Measurement
    label_a: str = "a"
    label_b: str = "b"

    def __post_init__(self) -> None:
        if self.a.unit != self.b.unit:
            raise ValueError(f"cannot difference {self.a.unit} against {self.b.unit}")

    @property
    def difference(self) -> float | None:
        if not (self.a.is_present and self.b.is_present):
            return None
        return self.b.value - self.a.value

    @property
    def percent(self) -> float | None:
        d = self.difference
        if d is None or not self.a.value:
            return None
        return 100.0 * d / self.a.value

    def to_dict(self) -> dict:
        return {
            self.label_a: self.a.to_dict(),
            self.label_b: self.b.to_dict(),
            "difference": self.difference,
            "difference_percent": round(self.percent, 2) if self.percent is not None else None,
            "unit": self.a.unit,
            "means": (
                f"The {self.label_a} and {self.label_b} figures differ. This states "
                "that two observations disagree and by how much. It does not "
                "establish that either is incorrect, and it is not a survey."
            ),
        }
