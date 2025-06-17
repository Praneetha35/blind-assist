# 🦯 Blind Assist

**Blind Assist** is a vision-based assistive system designed to help visually impaired individuals detect nearby obstacles using relative depth estimation. The app uses the `depth_anything_v2` pretrained model to analyze videos and alert users when a person or object is close.



## 💡 How It Works

- The app leverages **Depth Anything v2**, a powerful vision transformer model, to estimate relative depth from video input.
- **Relative depth** refers to the depth between objects in a scene rather than their absolute distance in meters. It provides enough information to understand which objects are closer or farther from the camera.
- Using the estimated depth values, the app analyzes the scene to detect if a person or object is too close, triggering a safety alert.



## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/blind-assist.git
   cd blind-assist
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```



## ▶️ Usage

1. Place your video inside the `/example` directory.
2. Run the script:

   ```bash
   python depth_v2.py
   ```

This script will:

- Load the video from `/example`
- Run depth inference using the `depth_anything_v2` model
- Analyze the relative depth values
- Print or play alerts when a nearby obstacle is detected



## 📂 Directory Structure

```
blind-assist/
├── depth_v2.py
├── requirements.txt
├── /example
│   └── input_video.mp4
└── README.md
```



## 🧠 Model Info

The `depth_anything_v2` model is a state-of-the-art transformer that provides dense depth prediction from a single RGB frame. It is ideal for real-time monocular depth estimation tasks and works well in a wide variety of environments.



## 🚧 Future Improvements

- Real-time camera feed support
- Audio feedback for alerts
- Integration with wearable hardware (e.g., Raspberry Pi + speakers)
