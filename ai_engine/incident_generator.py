import os
import re
import math
import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================

DATA_DIR = "data"

POTHOLE_INPUT_FILE = os.path.join(
    DATA_DIR,
    "smart_detection_results.csv"
)

WATERLOGGING_INPUT_FILE = os.path.join(
    DATA_DIR,
    "waterlogging_results.csv"
)

POTHOLE_OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "pothole_incidents.csv"
)

WATERLOGGING_OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "waterlogging_incidents.csv"
)

COMBINED_OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "incidents.csv"
)


# =========================================================
# INCIDENT SETTINGS
# =========================================================

MAX_FRAME_GAP = 3


# =========================================================
# LOCATION SETTINGS
# =========================================================
#
# IMPORTANT:
# These are DEMO coordinates because the current video
# pipeline does not provide real GPS coordinates.
#
# Replace these values with the actual project/demo area
# coordinates when you know the deployment location.
#
# These values are intentionally marked as DEMO in the
# generated CSV so they are not confused with real GPS.
# =========================================================

DEFAULT_LATITUDE = 28.6139
DEFAULT_LONGITUDE = 77.2090

LOCATION_SOURCE = "DEMO_SIMULATED"
LOCATION_LABEL = "Demo Location"


# =========================================================
# GET FRAME NUMBER
# =========================================================

def get_frame_number(frame_name):

    try:

        match = re.search(
            r"(\d+)(?=\.[^.]+$)",
            str(frame_name)
        )

        if match:

            return int(
                match.group(1)
            )

        return None

    except Exception:

        return None


# =========================================================
# SAFE FLOAT
# =========================================================

def safe_float(value, default=None):

    try:

        if pd.isna(value):

            return default

        return float(value)

    except Exception:

        return default


# =========================================================
# SAFE LOCATION
# =========================================================

def get_location_from_row(row, incident_index=0):

    """
    Get latitude/longitude from detection data if available.

    Priority:
    1. Latitude / Longitude
    2. lat / lon
    3. latitude / longitude_lowercase
    4. DEMO coordinates

    If actual coordinates are not present, small deterministic
    offsets are applied around the demo center so multiple
    incidents appear as separate points on the map.
    """

    latitude = None
    longitude = None
    source = None
    label = None

    latitude_columns = [
        "Latitude",
        "latitude",
        "Lat",
        "lat"
    ]

    longitude_columns = [
        "Longitude",
        "longitude",
        "Lon",
        "lon"
    ]

    # -----------------------------------------------------
    # FIND LATITUDE
    # -----------------------------------------------------

    for column in latitude_columns:

        if column in row.index:

            value = safe_float(
                row[column]
            )

            if value is not None:

                latitude = value
                break

    # -----------------------------------------------------
    # FIND LONGITUDE
    # -----------------------------------------------------

    for column in longitude_columns:

        if column in row.index:

            value = safe_float(
                row[column]
            )

            if value is not None:

                longitude = value
                break

    # -----------------------------------------------------
    # REAL GPS AVAILABLE
    # -----------------------------------------------------

    if (
        latitude is not None
        and longitude is not None
    ):

        return (
            latitude,
            longitude,
            "GPS_METADATA",
            "GPS Location"
        )

    # -----------------------------------------------------
    # DEMO LOCATION
    # -----------------------------------------------------
    #
    # Small deterministic offset prevents all incidents
    # from appearing exactly on top of each other.
    #
    # This is NOT real GPS.
    # -----------------------------------------------------

    index = max(
        0,
        int(incident_index) - 1
    )

    # Grid-like deterministic offset
    row_offset = (index % 5) - 2
    col_offset = (index // 5) % 5 - 2

    latitude = (
        DEFAULT_LATITUDE
        + row_offset * 0.0015
    )

    longitude = (
        DEFAULT_LONGITUDE
        + col_offset * 0.0015
    )

    return (
        round(latitude, 6),
        round(longitude, 6),
        LOCATION_SOURCE,
        LOCATION_LABEL
    )


# =========================================================
# GET CONFIDENCE SAFELY
# =========================================================

def get_confidence(row):

    # New detector
    if "best_confidence" in row.index:

        value = row["best_confidence"]

        if pd.notna(value):

            try:

                return float(value)

            except Exception:

                pass

    # Old detector
    if "confidence" in row.index:

        value = row["confidence"]

        if pd.notna(value):

            try:

                return float(value)

            except Exception:

                pass

    return 0.0


# =========================================================
# GENERATE POTHOLE INCIDENTS
# =========================================================

def generate_pothole_incidents():

    print("\n" + "=" * 60)
    print("POTHOLE INCIDENT GENERATION")
    print("=" * 60)

    if not os.path.exists(
        POTHOLE_INPUT_FILE
    ):

        print(
            f"Detection file not found: "
            f"{POTHOLE_INPUT_FILE}"
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            POTHOLE_INPUT_FILE
        )

    except Exception as error:

        print(
            f"ERROR reading pothole CSV: {error}"
        )

        return pd.DataFrame()

    if df.empty:

        print(
            "Pothole detection CSV is empty."
        )

        return pd.DataFrame()

    print(
        f"Total pothole frame records: {len(df)}"
    )

    # -----------------------------------------------------
    # STATUS CHECK
    # -----------------------------------------------------

    if "status" not in df.columns:

        print(
            "ERROR: 'status' column not found."
        )

        return pd.DataFrame()

    if "frame" not in df.columns:

        print(
            "ERROR: 'frame' column not found."
        )

        return pd.DataFrame()

    # -----------------------------------------------------
    # VALID STATUSES
    # -----------------------------------------------------

    valid_statuses = [
        "POTHOLE_DETECTED",
        "VALID_POTHOLE"
    ]

    valid_df = df[
        df["status"].astype(str).isin(
            valid_statuses
        )
    ].copy()

    if valid_df.empty:

        print(
            "No pothole detections found."
        )

        return pd.DataFrame()

    # -----------------------------------------------------
    # FRAME NUMBER
    # -----------------------------------------------------

    valid_df["frame_number"] = (
        valid_df["frame"]
        .apply(get_frame_number)
    )

    valid_df = valid_df.dropna(
        subset=["frame_number"]
    )

    if valid_df.empty:

        print(
            "No valid frame numbers found."
        )

        return pd.DataFrame()

    valid_df["frame_number"] = (
        valid_df["frame_number"]
        .astype(int)
    )

    valid_df = valid_df.sort_values(
        "frame_number"
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # GROUP FRAMES
    # -----------------------------------------------------

    incidents = []

    current_group = []

    previous_frame = None

    for _, row in valid_df.iterrows():

        current_frame = int(
            row["frame_number"]
        )

        if previous_frame is None:

            current_group.append(
                row
            )

        elif (
            current_frame - previous_frame
            <= MAX_FRAME_GAP
        ):

            current_group.append(
                row
            )

        else:

            incidents.append(
                current_group
            )

            current_group = [
                row
            ]

        previous_frame = current_frame

    if current_group:

        incidents.append(
            current_group
        )

    # -----------------------------------------------------
    # CREATE INCIDENT DATA
    # -----------------------------------------------------

    incident_data = []

    for index, group in enumerate(
        incidents,
        start=1
    ):

        incident_id = (
            f"POTHOLE_{index:03d}"
        )

        best_detection = max(
            group,
            key=get_confidence
        )

        best_confidence = get_confidence(
            best_detection
        )

        frame_numbers = [
            int(item["frame_number"])
            for item in group
        ]

        # ---------------------------------------------
        # RAW DETECTIONS
        # ---------------------------------------------

        total_potholes = 0

        for item in group:

            if (
                "potholes_in_frame"
                in item.index
            ):

                value = item[
                    "potholes_in_frame"
                ]

                if pd.notna(value):

                    try:

                        total_potholes += int(
                            value
                        )

                    except Exception:

                        total_potholes += 1

                else:

                    total_potholes += 1

            else:

                total_potholes += 1

        # ---------------------------------------------
        # EVIDENCE
        # ---------------------------------------------

        evidence_image = None

        if (
            "evidence_image"
            in best_detection.index
        ):

            evidence_image = (
                best_detection[
                    "evidence_image"
                ]
            )

        # ---------------------------------------------
        # LOCATION
        # ---------------------------------------------

        latitude, longitude, location_source, location_label = (
            get_location_from_row(
                best_detection,
                index
            )
        )

        # ---------------------------------------------
        # INCIDENT
        # ---------------------------------------------

        incident_data.append(
            {
                "incident_id": incident_id,

                "incident_type": "POTHOLE",

                "status": "POTHOLE_INCIDENT",

                "severity": "MEDIUM",

                "latitude": latitude,

                "longitude": longitude,

                "location_source": location_source,

                "location_label": location_label,

                "first_frame": min(
                    frame_numbers
                ),

                "last_frame": max(
                    frame_numbers
                ),

                "frames_grouped": len(
                    group
                ),

                "raw_detections": (
                    total_potholes
                ),

                "best_confidence": round(
                    best_confidence,
                    2
                ),

                "best_frame": (
                    best_detection[
                        "frame"
                    ]
                ),

                "evidence_image": (
                    evidence_image
                )
            }
        )

    result_df = pd.DataFrame(
        incident_data
    )

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    result_df.to_csv(
        POTHOLE_OUTPUT_FILE,
        index=False
    )

    print(
        f"Pothole incidents generated: "
        f"{len(result_df)}"
    )

    print(
        f"Saved: {POTHOLE_OUTPUT_FILE}"
    )

    return result_df


# =========================================================
# GET WATERLOGGING RISK
# =========================================================

def get_waterlogging_risk(row):

    risk = row.get(
        "Risk_Level",
        row.get(
            "risk",
            "LOW"
        )
    )

    if pd.isna(risk):

        return "LOW"

    return str(
        risk
    ).upper()


# =========================================================
# GET WATERLOGGING SCORE
# =========================================================

def get_waterlogging_score(row):

    possible_columns = [
        "Score",
        "Max_Score",
        "max_score",
        "Waterlogging_Score"
    ]

    for column in possible_columns:

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                try:

                    return float(value)

                except Exception:

                    pass

    return 0.0


# =========================================================
# GENERATE WATERLOGGING INCIDENTS
# =========================================================

def generate_waterlogging_incidents():

    print("\n" + "=" * 60)
    print("WATERLOGGING INCIDENT GENERATION")
    print("=" * 60)

    if not os.path.exists(
        WATERLOGGING_INPUT_FILE
    ):

        print(
            f"Waterlogging file not found: "
            f"{WATERLOGGING_INPUT_FILE}"
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            WATERLOGGING_INPUT_FILE
        )

    except Exception as error:

        print(
            f"ERROR reading waterlogging CSV: "
            f"{error}"
        )

        return pd.DataFrame()

    if df.empty:

        print(
            "Waterlogging CSV is empty."
        )

        return pd.DataFrame()

    print(
        f"Total waterlogging records: "
        f"{len(df)}"
    )

    # -----------------------------------------------------
    # DETECTION COLUMN
    # -----------------------------------------------------

    detection_column = None

    possible_columns = [
        "Waterlogging_Detected",
        "waterlogging_detected",
        "Detected",
        "Status"
    ]

    for column in possible_columns:

        if column in df.columns:

            detection_column = column

            break

    if detection_column is None:

        print(
            "Waterlogging detection column "
            "not found."
        )

        return pd.DataFrame()

    # -----------------------------------------------------
    # KEEP DETECTED FRAMES
    # -----------------------------------------------------

    detected_df = df[
        df[detection_column]
        .astype(str)
        .str.upper()
        .isin([
            "YES",
            "TRUE",
            "DETECTED",
            "WATERLOGGING_DETECTED"
        ])
    ].copy()

    if detected_df.empty:

        print(
            "No waterlogging detected frames."
        )

        return pd.DataFrame()

    # -----------------------------------------------------
    # FRAME NUMBER
    # -----------------------------------------------------

    if "Frame" in detected_df.columns:

        detected_df["frame_number"] = pd.to_numeric(
            detected_df["Frame"],
            errors="coerce"
        )

    elif "frame" in detected_df.columns:

        detected_df["frame_number"] = pd.to_numeric(
            detected_df["frame"],
            errors="coerce"
        )

    else:

        detected_df["frame_number"] = range(
            len(detected_df)
        )

    detected_df = detected_df.dropna(
        subset=["frame_number"]
    )

    detected_df["frame_number"] = (
        detected_df["frame_number"]
        .astype(int)
    )

    detected_df = detected_df.sort_values(
        "frame_number"
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # GROUP DETECTIONS
    # -----------------------------------------------------

    incidents = []

    current_group = []

    previous_frame = None

    for _, row in detected_df.iterrows():

        current_frame = int(
            row["frame_number"]
        )

        if previous_frame is None:

            current_group.append(
                row
            )

        elif (
            current_frame - previous_frame
            <= MAX_FRAME_GAP
        ):

            current_group.append(
                row
            )

        else:

            incidents.append(
                current_group
            )

            current_group = [
                row
            ]

        previous_frame = current_frame

    if current_group:

        incidents.append(
            current_group
        )

    # -----------------------------------------------------
    # CREATE INCIDENT DATA
    # -----------------------------------------------------

    incident_data = []

    for index, group in enumerate(
        incidents,
        start=1
    ):

        incident_id = (
            f"WATERLOGGING_{index:03d}"
        )

        # ---------------------------------------------
        # BEST FRAME
        # ---------------------------------------------

        best_detection = max(
            group,
            key=get_waterlogging_score
        )

        max_score = get_waterlogging_score(
            best_detection
        )

        risk = get_waterlogging_risk(
            best_detection
        )

        frame_numbers = [
            int(
                item["frame_number"]
            )
            for item in group
        ]

        # ---------------------------------------------
        # TIMESTAMP
        # ---------------------------------------------

        timestamp = ""

        if "Timestamp" in best_detection.index:

            timestamp = best_detection[
                "Timestamp"
            ]

        elif "timestamp" in best_detection.index:

            timestamp = best_detection[
                "timestamp"
            ]

        # ---------------------------------------------
        # EVIDENCE
        # ---------------------------------------------

        evidence_image = ""

        possible_evidence_columns = [
            "Evidence_Image",
            "evidence_image",
            "Evidence",
            "Image"
        ]

        for column in possible_evidence_columns:

            if column in best_detection.index:

                value = best_detection[
                    column
                ]

                if pd.notna(value):

                    evidence_image = str(
                        value
                    )

                    break

        # ---------------------------------------------
        # SEVERITY
        # ---------------------------------------------

        if risk == "HIGH":

            severity = "HIGH"

        elif risk == "MEDIUM":

            severity = "MEDIUM"

        else:

            severity = "LOW"

        # ---------------------------------------------
        # LOCATION
        # ---------------------------------------------

        latitude, longitude, location_source, location_label = (
            get_location_from_row(
                best_detection,
                index
            )
        )

        # ---------------------------------------------
        # INCIDENT RECORD
        # ---------------------------------------------

        incident_data.append(
            {
                "incident_id": incident_id,

                "incident_type": (
                    "WATERLOGGING"
                ),

                "status": (
                    "WATERLOGGING_INCIDENT"
                ),

                "severity": severity,

                "risk_level": risk,

                "latitude": latitude,

                "longitude": longitude,

                "location_source": location_source,

                "location_label": location_label,

                "first_frame": min(
                    frame_numbers
                ),

                "last_frame": max(
                    frame_numbers
                ),

                "frames_grouped": len(
                    group
                ),

                "max_waterlogging_score": round(
                    max_score,
                    2
                ),

                "timestamp": timestamp,

                "best_frame": (
                    best_detection.get(
                        "Frame",
                        best_detection.get(
                            "frame",
                            ""
                        )
                    )
                ),

                "evidence_image": (
                    evidence_image
                )
            }
        )

    result_df = pd.DataFrame(
        incident_data
    )

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    result_df.to_csv(
        WATERLOGGING_OUTPUT_FILE,
        index=False
    )

    print(
        f"Waterlogging incidents generated: "
        f"{len(result_df)}"
    )

    print(
        f"Saved: "
        f"{WATERLOGGING_OUTPUT_FILE}"
    )

    return result_df


# =========================================================
# COMBINE ALL INCIDENTS
# =========================================================

def combine_incidents(
    pothole_df,
    waterlogging_df
):

    print("\n" + "=" * 60)
    print("COMBINING INCIDENTS")
    print("=" * 60)

    frames = []

    if (
        pothole_df is not None
        and not pothole_df.empty
    ):

        frames.append(
            pothole_df
        )

    if (
        waterlogging_df is not None
        and not waterlogging_df.empty
    ):

        frames.append(
            waterlogging_df
        )

    if not frames:

        print(
            "No incidents available."
        )

        return pd.DataFrame()

    combined_df = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    # -----------------------------------------------------
    # SORT BY INCIDENT TYPE
    # -----------------------------------------------------

    if "incident_type" in combined_df.columns:

        combined_df = combined_df.sort_values(
            "incident_type"
        ).reset_index(
            drop=True
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    combined_df.to_csv(
        COMBINED_OUTPUT_FILE,
        index=False
    )

    print(
        f"Total incidents: "
        f"{len(combined_df)}"
    )

    print(
        f"Combined report saved: "
        f"{COMBINED_OUTPUT_FILE}"
    )

    return combined_df


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    pothole_df,
    waterlogging_df,
    combined_df
):

    print("\n")
    print("=" * 60)
    print("INCIDENT GENERATION SUMMARY")
    print("=" * 60)

    pothole_count = (
        len(pothole_df)
        if pothole_df is not None
        else 0
    )

    waterlogging_count = (
        len(waterlogging_df)
        if waterlogging_df is not None
        else 0
    )

    total_count = (
        len(combined_df)
        if combined_df is not None
        else 0
    )

    print(
        f"Pothole incidents: "
        f"{pothole_count}"
    )

    print(
        f"Waterlogging incidents: "
        f"{waterlogging_count}"
    )

    print(
        f"Total incidents: "
        f"{total_count}"
    )

    # -----------------------------------------------------
    # LOCATION SUMMARY
    # -----------------------------------------------------

    if (
        combined_df is not None
        and not combined_df.empty
    ):

        print("-" * 60)
        print("LOCATION INTELLIGENCE")

        if "location_source" in combined_df.columns:

            gps_count = len(
                combined_df[
                    combined_df[
                        "location_source"
                    ]
                    == "GPS_METADATA"
                ]
            )

            demo_count = len(
                combined_df[
                    combined_df[
                        "location_source"
                    ]
                    == "DEMO_SIMULATED"
                ]
            )

            print(
                f"GPS incidents: {gps_count}"
            )

            print(
                f"Demo incidents: {demo_count}"
            )

    print("-" * 60)

    if (
        waterlogging_df is not None
        and not waterlogging_df.empty
    ):

        print(
            "\nWATERLOGGING INCIDENTS"
        )

        for _, row in waterlogging_df.iterrows():

            print()

            print(
                f"ID: "
                f"{row.get('incident_id', '')}"
            )

            print(
                f"Severity: "
                f"{row.get('severity', '')}"
            )

            print(
                f"Risk: "
                f"{row.get('risk_level', '')}"
            )

            print(
                f"Latitude: "
                f"{row.get('latitude', '')}"
            )

            print(
                f"Longitude: "
                f"{row.get('longitude', '')}"
            )

            print(
                f"Location Source: "
                f"{row.get('location_source', '')}"
            )

            print(
                f"Frames: "
                f"{row.get('first_frame', '')}"
                f" - "
                f"{row.get('last_frame', '')}"
            )

            print(
                f"Max Score: "
                f"{row.get('max_waterlogging_score', '')}"
            )

    print()

    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("URBAN INTELLIGENCE PLATFORM")
    print("SMART INCIDENT GENERATION")
    print("=" * 60)

    # -----------------------------------------------------
    # GENERATE POTHOLE INCIDENTS
    # -----------------------------------------------------

    pothole_df = (
        generate_pothole_incidents()
    )

    # -----------------------------------------------------
    # GENERATE WATERLOGGING INCIDENTS
    # -----------------------------------------------------

    waterlogging_df = (
        generate_waterlogging_incidents()
    )

    # -----------------------------------------------------
    # COMBINE
    # -----------------------------------------------------

    combined_df = combine_incidents(
        pothole_df,
        waterlogging_df
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print_summary(
        pothole_df,
        waterlogging_df,
        combined_df
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    main()