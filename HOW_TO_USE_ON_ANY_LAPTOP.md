# 🚀 BEL Urban Intelligence Platform (SIH 26124) - Portable Guide
## Kisi Bhi Laptop Me Kaise Use Karein (Step-by-Step Instructions)

---

### 📦 1. Kisi Aur Laptop Me Kaise Copy Karein
Aap is poore folder (`SIH26124_Urban_Intelligence`) ko:
1. **Pen Drive / External Hard Drive** me copy kar sakte hain, ya
2. **ZIP file** bana kar Google Drive / Telegram / Pendrive se transfer kar sakte hain.

> **Note**: Saara detection data (`data/*.csv`) aur saari 260+ forensic photos (`evidence/`) pehle se hi folder me saved hain. Isliye kisi bhi laptop me kholne par internet na hone par bhi saare 13 tabs instantly open honge!

---

### ⚡ 2. One-Click Launch (Sabse Aasan Tarika)
Kisi bhi laptop me folder kholne ke baad:
* Simply **`RUN_PORTABLE.bat`** file par **Double-Click** karein!

**Ye script automatically kya karta hai:**
1. Check karta hai ki us laptop me Python kahan installed hai.
2. Agar folder dusre laptop se copy kiya gaya hai, to ye `venv` ke paths ko automatically repair kar deta hai bina kuch download kiye.
3. Agar us laptop me pehli baar chal raha hai aur libraries missing hain, to ye automatic setup complete kar deta hai.
4. Streamlit server start karke **default browser me `http://localhost:8501` turant open kar deta hai**.

---

### 🛠️ 3. Agar Naye Laptop Me Python Na Ho
Agar kisi laptop me bilkul bhi Python installed nahi hai:
1. [Python Official Website](https://www.python.org/downloads/) se Python download karke install karein.
2. **Important**: Install karte time **"Add python.exe to PATH"** checkbox ko zaroor tick karein.
3. Uske baad wapas aakar **`RUN_PORTABLE.bat`** par double click kar dein.

---

### 📂 4. Core Files Overview
* **`RUN_PORTABLE.bat`**: Main 1-Click launcher (har laptop ke liye).
* **`INSTALL_DEPENDENCIES.bat`**: Agar kisi machine me fresh libraries install karni ho (One-click).
* **`app.py`**: Executive Dashboard aur Streamlit frontend.
* **`data/`**: Saare 1,032 spatial incidents aur municipal registers.
* **`evidence/`**: 260+ high-resolution geo-tagged evidence frames with Lat/Lon badges.
* **`ai_engine/master_pipeline.py`**: High-precision edge pipeline (Potholes, Waterlogging, ANPR, Pedestrian, Garbage, Signals).
