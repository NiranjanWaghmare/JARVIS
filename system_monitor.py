import psutil
import os
import time

def get_system_vitals():
    """Collect current system vitals."""
    vitals = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 1),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "disk_percent": psutil.disk_usage('/').percent,
        "disk_used_gb": round(psutil.disk_usage('/').used / (1024**3), 1),
        "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 1),
        "temp_celsius": None
    }
    
    # Try to get temperature (limited on Windows)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    vitals["temp_celsius"] = entries[0].current
                    break
    except (AttributeError, Exception):
        pass
    
    # Windows fallback: WMI thermal zone (requires Admin)
    if vitals["temp_celsius"] is None and os.name == "nt":
        try:
            import subprocess
            r = subprocess.run(
                ["powershell", "-Command",
                 "(Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi).CurrentTemperature"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                # WMI returns temp in tenths of Kelvin
                kelvin_tenths = float(r.stdout.strip().splitlines()[0])
                vitals["temp_celsius"] = round((kelvin_tenths / 10) - 273.15, 1)
        except Exception:
            pass
    
    return vitals

def check_alerts(vitals, thresholds=None):
    """Check vitals against thresholds. Returns list of alert messages."""
    if thresholds is None:
        thresholds = {
            "cpu": int(os.getenv("CPU_ALERT", "85")),
            "ram": int(os.getenv("RAM_ALERT", "90")),
            "disk": int(os.getenv("DISK_ALERT", "90")),
            "temp": int(os.getenv("TEMP_ALERT", "80")),
        }
    
    alerts = []
    
    if vitals["cpu_percent"] > thresholds["cpu"]:
        alerts.append(f"CPU is running at {vitals['cpu_percent']}% (threshold: {thresholds['cpu']}%)")
    
    if vitals["ram_percent"] > thresholds["ram"]:
        alerts.append(f"RAM at {vitals['ram_percent']}% ({vitals['ram_used_gb']}/{vitals['ram_total_gb']} GB) (threshold: {thresholds['ram']}%)")
    
    if vitals["disk_percent"] > thresholds["disk"]:
        alerts.append(f"Disk usage at {vitals['disk_percent']}% ({vitals['disk_used_gb']}/{vitals['disk_total_gb']} GB) (threshold: {thresholds['disk']}%)")
    
    if vitals["temp_celsius"] and vitals["temp_celsius"] > thresholds["temp"]:
        alerts.append(f"System temperature at {vitals['temp_celsius']}°C (threshold: {thresholds['temp']}°C)")
    
    return alerts

def format_vitals_summary(vitals):
    """Format vitals into a readable summary string."""
    lines = [
        f"CPU: {vitals['cpu_percent']}%",
        f"RAM: {vitals['ram_percent']}% ({vitals['ram_used_gb']}/{vitals['ram_total_gb']} GB)",
        f"Disk: {vitals['disk_percent']}% ({vitals['disk_used_gb']}/{vitals['disk_total_gb']} GB)",
    ]
    if vitals["temp_celsius"]:
        lines.append(f"Temp: {vitals['temp_celsius']}°C")
    else:
        lines.append("Temp: Sensors unavailable")
    return "\n".join(lines)

if __name__ == "__main__":
    v = get_system_vitals()
    print(format_vitals_summary(v))
    alerts = check_alerts(v)
    if alerts:
        print("\n[!] ALERTS:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("\n[OK] All systems nominal.")
