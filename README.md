# Minecraft SLAM Project

This project explores SLAM methods inside Minecraft. The goal is to build an open-source, zero-budget setup for testing different SLAM algorithms without relying on drones, cameras, or real-world (seen-millions-of-times) datasets. Minecraft works as a controlled proof of concept where we can study how SLAM behaves in many types of environments.

We plan to build multiple worlds and also provide code and instructions so you can create your own. Because Minecraft gives us exact, noiseless camera poses at every frame, we avoid things like camera calibration and sensor noise. This makes experiments simpler, but also less realistic. The visual style of Minecraft is far from real imagery, so results won’t always generalize to real scenes. Still, it works well as a learning platform, a place to compare algorithms, and a way to try ideas quickly before moving to real data.

SPACE FOR GIF EXAMPLES WHENEVER WE HAVE THEM READY

## Folder Structure

```bash
.
├── pose_extraction/   # Tools and scripts to extract and process camera poses and frames from Minecraft
├── sfm/   # Classic computer vision Structure from Motion experiments before diving into modern SLAM
└── slam/              # SLAM algorithms, experiments, and evaluation code
```