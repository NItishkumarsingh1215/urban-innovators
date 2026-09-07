# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )
    print(
        "🌧️ WATERLOGGING VIDEO DETECTOR"
    )
    print(
        "======================================\n"
    )

    UPLOAD_DIR = Path("uploads/road_videos")
    VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

    video_files = [
        file for file in UPLOAD_DIR.iterdir()
        if file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if len(video_files) == 0:
        print("ERROR: No road video found in uploads/road_videos!")
        video = input("Enter road video path manually: ").strip()
    else:
        video = video_files[0]
        print(f"Auto-detected video: {video}")

    result = analyze_video(
        video
    )

    print(
        "\n========== RESULT =========="
    )

    print(
        "Status:",
        result["success"]
    )

    print(
        "Risk:",
        result["risk"]
    )

    print(
        "Maximum Score:",
        result["max_score"]
    )

    print(
        "Analyzed Frames:",
        result["analyzed_frames"]
    )

    print(
        "Waterlogging Frames:",
        result["waterlogging_frames"]
    )

    print(
        "Detection Percentage:",
        result.get(
            "detection_percentage",
            0
        ),
        "%"
    )

    print(
        "CSV:",
        result["csv_path"]
    )

    print(
        "Evidence:",
        result["evidence_dir"]
    )