import pandas as pd
from pathlib import Path
from datetime import datetime

def analyze_od_and_delays(output_csv="data/od_delay_results.csv"):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Origin-Destination (OD) Matrix & Route Delay Data for Public Transport Fleet
    od_routes = [
        {"Route_ID": "RT-101", "Origin": "Connaught Place", "Destination": "AIIMS", "Standard_Duration_Min": 30, "Current_Delay_Min": 12, "Congestion_Index": "High", "OD_Volume": "1,450 daily trips"},
        {"Route_ID": "RT-102", "Origin": "ISBT Kashmiri Gate", "Destination": "Dwarka Sector 21", "Standard_Duration_Min": 45, "Current_Delay_Min": 5, "Congestion_Index": "Medium", "OD_Volume": "2,100 daily trips"},
        {"Route_ID": "RT-103", "Origin": "Karol Bagh", "Destination": "Nehru Place", "Standard_Duration_Min": 40, "Current_Delay_Min": 18, "Congestion_Index": "Critical", "OD_Volume": "1,800 daily trips"},
        {"Route_ID": "RT-104", "Origin": "Lajpat Nagar", "Destination": "Rohini Sector 18", "Standard_Duration_Min": 50, "Current_Delay_Min": 8, "Congestion_Index": "Low", "OD_Volume": "950 daily trips"}
    ]

    df = pd.DataFrame(od_routes)
    df["Analysis_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(output_csv, index=False)
    return {"success": True, "message": "OD and Route Delay analysis completed.", "csv_path": str(output_csv)}

if __name__ == "__main__":
    analyze_od_and_delays()
    print("OD & Route Delay test completed.")