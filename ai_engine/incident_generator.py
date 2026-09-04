import os
import pandas as pd


# ==========================================
# FILE PATHS
# ==========================================

INPUT_FILE = "data/smart_detection_results.csv"

OUTPUT_FILE = "data/pothole_incidents.csv"


# ==========================================
# SETTINGS
# ==========================================

# Agar detections ke frame numbers me
# itna ya isse kam gap hai,
# to unhe same incident maana jayega.
MAX_FRAME_GAP = 2


# ==========================================
# GET FRAME NUMBER
# ==========================================

def get_frame_number(frame_name):

    try:
        number_part = (
            frame_name
            .split("_")[-1]
            .split(".")[0]
        )

        return int(number_part)

    except Exception:
        return None


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("=" * 55)
    print("URBAN INTELLIGENCE - INCIDENT GENERATION")
    print("=" * 55)

    # Check input file
    if not os.path.exists(INPUT_FILE):

        print(
            f"ERROR: Detection results not found -> "
            f"{INPUT_FILE}"
        )

        return

    # Read smart detection results
    df = pd.read_csv(INPUT_FILE)

    print(
        f"Total frame records found: "
        f"{len(df)}"
    )

    # Keep only valid pothole detections
    valid_df = df[
        df["status"] == "VALID_POTHOLE"
    ].copy()

    # No detections
    if valid_df.empty:

        print(
            "No valid pothole detections found."
        )

        return

    # Extract frame numbers
    valid_df["frame_number"] = (
        valid_df["frame"]
        .apply(get_frame_number)
    )

    # Remove invalid frame numbers
    valid_df = valid_df.dropna(
        subset=["frame_number"]
    )

    # Sort detections
    valid_df = valid_df.sort_values(
        "frame_number"
    )

    print(
        f"Valid detections found: "
        f"{len(valid_df)}"
    )


    # ==========================================
    # GROUP DETECTIONS INTO INCIDENTS
    # ==========================================

    incidents = []

    current_group = []

    previous_frame = None


    for _, row in valid_df.iterrows():

        current_frame = int(
            row["frame_number"]
        )

        # First detection
        if previous_frame is None:

            current_group.append(row)

        # Same incident
        elif (
            current_frame - previous_frame
            <= MAX_FRAME_GAP
        ):

            current_group.append(row)

        # New incident
        else:

            incidents.append(
                current_group
            )

            current_group = [row]

        previous_frame = current_frame


    # Add last group
    if len(current_group) > 0:

        incidents.append(
            current_group
        )


    # ==========================================
    # CREATE INCIDENT REPORT
    # ==========================================

    incident_data = []

    for index, group in enumerate(
        incidents,
        start=1
    ):

        incident_id = (
            f"POTHOLE_{index:03d}"
        )

        # Best detection by confidence
        best_detection = max(
            group,
            key=lambda item: float(
                item["confidence"]
            )
        )

        # Frame numbers
        frame_numbers = [
            int(item["frame_number"])
            for item in group
        ]

        incident_data.append(
            {
                "incident_id": incident_id,

                "status": "POTHOLE_DETECTED",

                "first_frame": min(
                    frame_numbers
                ),

                "last_frame": max(
                    frame_numbers
                ),

                "frames_detected": len(
                    group
                ),

                "best_confidence": round(
                    float(
                        best_detection[
                            "confidence"
                        ]
                    ),
                    2
                ),

                "best_frame": best_detection[
                    "frame"
                ],

                "evidence_image": best_detection[
                    "evidence_image"
                ]
            }
        )


    # ==========================================
    # SAVE INCIDENT REPORT
    # ==========================================

    incidents_df = pd.DataFrame(
        incident_data
    )

    incidents_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ==========================================
    # PRINT SUMMARY
    # ==========================================

    print("\n" + "=" * 55)

    print("INCIDENT GENERATION COMPLETED")

    print("=" * 55)

    print(
        f"Total detections: "
        f"{len(valid_df)}"
    )

    print(
        f"Unique pothole incidents: "
        f"{len(incident_data)}"
    )

    print(
        f"Incident report saved: "
        f"{OUTPUT_FILE}"
    )

    print("\nINCIDENTS:")

    for incident in incident_data:

        print(
            f"\n{incident['incident_id']}"
        )

        print(
            f"Frames: "
            f"{incident['first_frame']} "
            f"to "
            f"{incident['last_frame']}"
        )

        print(
            f"Best confidence: "
            f"{incident['best_confidence']}%"
        )

        print(
            f"Best frame: "
            f"{incident['best_frame']}"
        )

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()