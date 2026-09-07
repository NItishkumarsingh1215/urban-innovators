import pandas as pd
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "traffic_results.csv"

TIMINGS = {
    "Low": {"Green": 15, "Red": 45},
    "Medium": {"Green": 30, "Red": 30},
    "High": {"Green": 60, "Red": 10}
}

def run_smart_signal():
    if not CSV_PATH.exists(): return

    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty: return
        
        latest_data = df.iloc[-1]
        traffic_level = latest_data.get("Traffic_Level", "Low")
        
        # Check if Emergency is True
        is_emergency = latest_data.get("Emergency", False)
        
        if is_emergency:
            print("\n" + "🚨"*20)
            print("🚨 EMERGENCY VEHICLE DETECTED! OVERRIDING SIGNAL 🚨")
            print("🚨"*20)
            print("🟢 GREEN Light Time   : 999 sec (FORCE OPEN)")
            print("🔴 RED Light Time     : 0 sec")
            print("========================================")
        else:
            signal = TIMINGS.get(traffic_level, {"Green": 20, "Red": 40})
            print("\n" + "="*40)
            print(f"🚦 LIVE TRAFFIC LEVEL : {traffic_level.upper()}")
            print("-" * 40)
            print(f"🟢 GREEN Light Time   : {signal['Green']} sec")
            print(f"🔴 RED Light Time     : {signal['Red']} sec")
            print("========================================")
        
    except Exception as e:
        pass

if __name__ == "__main__":
    print("🚦 Smart Signal Controller Started... (Press Ctrl+C to stop)")
    while True:
        run_smart_signal()
        time.sleep(2) # Thoda fast check karega (every 2 sec)