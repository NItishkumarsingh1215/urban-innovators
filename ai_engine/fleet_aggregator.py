import pandas as pd
from pathlib import Path
from datetime import datetime

def aggregate_fleet_data(output_csv="data/fleet_summary.csv"):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Simulated fleet of public transport buses acting as mobile sensing units
    fleet_buses = [
        {"Bus_ID": "BUS-DL01-101", "Route": "Route 12 (Connaught Place - AIIMS)", "Driver": "Rajesh Kumar", "Status": "Active"},
        {"Bus_ID": "BUS-DL01-102", "Route": "Route 45 (ISBT - Dwarka)", "Driver": "Amit Sharma", "Status": "Active"},
        {"Bus_ID": "BUS-DL01-103", "Route": "Route 22 (Karol Bagh - Nehru Place)", "Driver": "Suresh Singh", "Status": "Active"},
        {"Bus_ID": "BUS-DL01-104", "Route": "Route 89 (Lajpat Nagar - Rohini)", "Driver": "Vikram Malhotra", "Status": "Active"}
    ]

    # Read existing incidents if available
    incidents_path = Path("data/incidents.csv")
    incidents_count = 0
    if incidents_path.exists():
        try:
            df_inc = pd.read_csv(incidents_path)
            incidents_count = len(df_inc)
        except Exception:
            pass

    # Generate aggregated fleet intelligence records
    fleet_records = []
    for i, bus in enumerate(fleet_buses):
        # Distribute detected events across the active bus fleet
        simulated_issues = max(2, (incidents_count + i * 3) % 15)
        fleet_records.append({
            "Bus_ID": bus["Bus_ID"],
            "Route": bus["Route"],
            "Driver": bus["Driver"],
            "Status": bus["Status"],
            "Issues_Detected": simulated_issues,
            "Last_Sync_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Fleet_Health": "Optimal" if simulated_issues < 8 else "Needs Attention"
        })

    df_fleet = pd.DataFrame(fleet_records)
    df_fleet.to_csv(output_csv, index=False)
    return {"success": True, "message": "Fleet data aggregated successfully.", "csv_path": str(output_csv)}

if __name__ == "__main__":
    aggregate_fleet_data()
    print("Fleet Aggregation test completed.")