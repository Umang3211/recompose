# Recompose: Cognitive Composting

An AI depth vision system that recognizes food waste the moment it hits the bin, estimates how much was thrown away, and turns it into feedback people actually act on.

**3rd Place, Mercer Science and Engineering Fair** · **APA Special Award for Psychology** · **Published in IJSR** ([DOI 10.21275/SR251001091332](https://dx.doi.org/10.21275/SR251001091332))

In a school cafeteria pilot, Recompose cut food waste by **36.9%**.

Live site: [recompose.vercel.app](https://recompose.vercel.app/)

## How it works

1. An **Intel RealSense** depth camera captures a color frame and a depth map of the tray or bin.
2. **Llama 3.2 Vision** (hosted on Azure AI) labels each food item in the frame.
3. The depth map gives a volume for each item; a per food density and stacking factor converts that to an estimated mass.
4. Totals roll up into kilograms wasted, CO₂ equivalent, and "trees needed to offset", shown on a live Flask dashboard and logged to CSV.
5. The dashboard is the behavioral intervention: people see the cost of what they throw away right at the bin.

## Stack

Python · OpenCV · pyrealsense2 · NumPy / pandas · Flask · Llama 3.2 Vision (Azure AI)

## Run it

```bash
pip install pyrealsense2 opencv-python numpy pandas flask requests
export AZURE_LLAMAVISION_API_KEY=your_key
python foodwaste.py
```

Needs an Intel RealSense depth camera (D400 series) plugged in over USB 3.
