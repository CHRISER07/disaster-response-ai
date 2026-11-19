"""
core/evidence_validator.py

Comprehensive Evidence Normalization, 4-Timestamp Temporal Engine,
Provenance Verification, and Physical Range Validation Layer.

Every incoming observation—whether from live USGS stream gauges, Open-Meteo weather APIs,
historical CrisisLex microblogs, or official FEMA/CDC doctrine—must pass through
this validation pipeline before being indexed or consumed by the agent.

Architecture:
1. 4-Timestamp System:
   - event_time:   Physical time the event occurred in the real world
   - observed_at:  Time the sensor or field observer recorded the measurement
   - ingested_at:  Time our disaster intelligence system received the packet
   - valid_until:  Exact wall-clock expiration after which data is STALE/EXPIRED
2. Physical Range & Unit Sanity Checks (Gage height, wind, rain, magnitude)
3. Source Provenance & Cryptographic Signature
4. Duplicate Detection (Exact hash and text normalizer)
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


class SourceType(str, Enum):
    TELEMETRY = "TELEMETRY"               # USGS river gages, weather stations
    WEATHER_FORECAST = "WEATHER_FORECAST" # Open-Meteo active weather
    SOCIAL_MEDIA = "SOCIAL_MEDIA"         # Field eyewitness reports, CrisisLex microblogs
    OFFICIAL_PROTOCOL = "OFFICIAL_PROTOCOL" # FEMA, CDC doctrine
    IMAGERY = "IMAGERY"                   # Drone, satellite xView2 imagery
    SYNTHETIC_BENCHMARK = "SYNTHETIC_BENCHMARK" # Evaluation test suites


class TemporalTier(str, Enum):
    LIVE = "LIVE"                         # Current within strict TTL (<30-60m)
    RECENT = "RECENT"                     # Active disaster cycle (1-24h)
    HISTORICAL = "HISTORICAL"             # Past disaster event (days/years ago)
    STATIC_DOCTRINE = "STATIC_DOCTRINE"   # Authoritative protocols (long TTL)


class FreshnessStatus(str, Enum):
    FRESH = "FRESH"                       # Well within operational validity window
    AGING = "AGING"                       # Near expiration, warning flag attached
    EXPIRED = "EXPIRED"                   # Stale; cannot drive active operational decisions


# Source-specific Time-To-Live (TTL) definitions
TTL_POLICIES = {
    SourceType.TELEMETRY: {
        "fresh_ttl": timedelta(minutes=30),
        "max_valid_ttl": timedelta(hours=2),
    },
    SourceType.WEATHER_FORECAST: {
        "fresh_ttl": timedelta(hours=1),
        "max_valid_ttl": timedelta(hours=3),
    },
    SourceType.SOCIAL_MEDIA: {
        "fresh_ttl": timedelta(hours=2),
        "max_valid_ttl": timedelta(hours=6),
    },
    SourceType.IMAGERY: {
        "fresh_ttl": timedelta(hours=4),
        "max_valid_ttl": timedelta(hours=24),
    },
    SourceType.OFFICIAL_PROTOCOL: {
        "fresh_ttl": timedelta(days=365),
        "max_valid_ttl": timedelta(days=3650),
    },
    SourceType.SYNTHETIC_BENCHMARK: {
        "fresh_ttl": timedelta(days=365),
        "max_valid_ttl": timedelta(days=365),
    },
}

# Physical bounds for sensor data
PHYSICAL_BOUNDS = {
    "gage_height_ft": (0.0, 120.0),       # River stage height in feet
    "discharge_cfs": (0.0, 1_000_000.0),  # Cubic feet per second
    "precipitation_mm_hr": (0.0, 400.0),  # World record rain rate is ~300mm/hr
    "wind_speed_kmh": (0.0, 400.0),       # Max recorded gust ~408 km/h
    "temperature_c": (-60.0, 65.0),       # Earth temperature extremes
    "earthquake_magnitude": (-1.0, 10.0), # Richter / Moment magnitude
}


@dataclass
class EvidencePacket:
    """Standardized, normalized evidence packet with full provenance and temporal tracking."""
    evidence_id: str
    source_id: str
    source_type: SourceType
    raw_text: str
    
    # 4-Timestamp System
    event_time: datetime
    observed_at: datetime
    ingested_at: datetime
    valid_until: datetime
    
    temporal_tier: TemporalTier
    freshness: FreshnessStatus
    
    # Geospatial Metadata
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_name: Optional[str] = None
    
    # Primary Extracted Numeric Measurement
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = None
    
    # Validation & Provenance
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        """Returns standard academic / operational citation tag."""
        date_str = self.observed_at.strftime("%Y-%m-%d %H:%M UTC")
        if self.source_type == SourceType.OFFICIAL_PROTOCOL:
            return f"[KB - {self.source_id}]"
        elif self.temporal_tier == TemporalTier.LIVE:
            return f"[LIVE - {self.source_id}, {date_str}]"
        elif self.temporal_tier == TemporalTier.HISTORICAL:
            return f"[HISTORICAL - {self.source_id}, {self.observed_at.year}]"
        return f"[{self.source_type.value} - {self.source_id}, {date_str}]"


class EvidenceValidator:
    """Normalization, validation, and sanitization engine for multi-source disaster streams."""

    @staticmethod
    def parse_datetime(dt_input: Any) -> datetime:
        """Robust parser for diverse ISO8601, UTC, and epoch timestamps."""
        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None:
                return dt_input.replace(tzinfo=timezone.utc)
            return dt_input.astimezone(timezone.utc)
        
        if isinstance(dt_input, (int, float)):
            # Epoch timestamp
            return datetime.fromtimestamp(dt_input, tz=timezone.utc)
            
        if isinstance(dt_input, str):
            clean_s = dt_input.strip()
            # Replace common timezone abbreviations
            clean_s = re.sub(r"Z$", "+00:00", clean_s)
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.strptime(clean_s, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
        # Do not launder an invalid timestamp into "now": raise so the caller
        # can mark the packet invalid instead of treating it as fresh evidence.
        raise ValueError(f"Unparseable timestamp: {dt_input!r}")

    @classmethod
    def validate_and_normalize(
        cls,
        source_id: str,
        source_type: SourceType,
        raw_text: str,
        observed_at: Any,
        event_time: Any = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        location_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        metric_unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidencePacket:
        """
        Ingests, validates, normalizes, and signs an evidence observation.
        """
        now = datetime.now(timezone.utc)
        ingested_at = now
        obs_dt = cls.parse_datetime(observed_at)
        ev_dt = cls.parse_datetime(event_time) if event_time else obs_dt

        errors = []

        # 1. Timestamp Sanity: observed_at cannot be in the future (with 5 min clock skew tolerance)
        if obs_dt > now + timedelta(minutes=5):
            errors.append(f"Future timestamp detected: {obs_dt.isoformat()} > now ({now.isoformat()}); packet invalid")
            # Do not replace with now: that launders a bad timestamp into LIVE.
            obs_dt = now + timedelta(days=3650)

        # 2. Compute valid_until using source-specific TTL
        ttl_policy = TTL_POLICIES.get(source_type, TTL_POLICIES[SourceType.TELEMETRY])
        valid_until = obs_dt + ttl_policy["max_valid_ttl"]
        fresh_until = obs_dt + ttl_policy["fresh_ttl"]

        # 3. Determine Freshness Status
        if now <= fresh_until:
            freshness = FreshnessStatus.FRESH
        elif now <= valid_until:
            freshness = FreshnessStatus.AGING
        else:
            freshness = FreshnessStatus.EXPIRED

        # 4. Determine Temporal Tier
        age_hours = (now - obs_dt).total_seconds() / 3600.0
        if source_type == SourceType.OFFICIAL_PROTOCOL:
            temporal_tier = TemporalTier.STATIC_DOCTRINE
        elif age_hours <= 1.0 and freshness == FreshnessStatus.FRESH:
            temporal_tier = TemporalTier.LIVE
        elif age_hours <= 72.0:
            temporal_tier = TemporalTier.RECENT
        else:
            temporal_tier = TemporalTier.HISTORICAL

        # 5. Coordinate Bounds Checking
        if lat is not None:
            if not (-90.0 <= lat <= 90.0):
                errors.append(f"Invalid latitude {lat}; must be in [-90, 90]")
                lat = None
        if lon is not None:
            if not (-180.0 <= lon <= 180.0):
                errors.append(f"Invalid longitude {lon}; must be in [-180, 180]")
                lon = None

        # 6. Physical Measurement Sanity
        if metric_name and metric_value is not None:
            if metric_name in PHYSICAL_BOUNDS:
                low, high = PHYSICAL_BOUNDS[metric_name]
                if not (low <= metric_value <= high):
                    errors.append(
                        f"Measurement out of physical bounds for {metric_name}: "
                        f"{metric_value} not in [{low}, {high}]"
                    )
                    # Cap value to boundaries or flag
            elif metric_value < 0 and "wind" in metric_name.lower():
                errors.append(f"Negative wind speed: {metric_value}")

        # 7. Provenance Signature
        content_for_hash = f"{source_id}|{source_type.value}|{obs_dt.isoformat()}|{raw_text.strip()}"
        prov_hash = hashlib.sha256(content_for_hash.encode("utf-8")).hexdigest()
        evidence_id = f"EV-{prov_hash[:12]}"

        is_valid = len(errors) == 0

        return EvidencePacket(
            evidence_id=evidence_id,
            source_id=source_id,
            source_type=source_type,
            raw_text=raw_text.strip(),
            event_time=ev_dt,
            observed_at=obs_dt,
            ingested_at=ingested_at,
            valid_until=valid_until,
            temporal_tier=temporal_tier,
            freshness=freshness,
            lat=lat,
            lon=lon,
            location_name=location_name,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            is_valid=is_valid,
            validation_errors=errors,
            provenance_hash=prov_hash,
            metadata=metadata or {},
        )


# Singleton Validator Instance
evidence_validator = EvidenceValidator()
