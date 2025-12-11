# Minecraft 3D Vision Project

This project explores a couple of 3D computer-vision pipelines inside Minecraft: Structure from Motion (SfM) and Visual Simultaneous Localization and Mapping (Visual SLAM). The goal is to build an open-source, zero-budget setup for experimenting with these algorithms without relying on drones, physical cameras, or overused real-world datasets. Minecraft serves as a controlled proof-of-concept environment where we can study how different 3D reconstruction and tracking techniques behave across many types of scenes.

We plan to build multiple worlds and provide tools, code, and instructions so you can easily create your own. Because Minecraft provides exact, noiseless camera poses at every frame, we can bypass challenges like camera calibration and sensor noise. This simplifies experimentation, though it also makes the setup less realistic. Having said that, the stylized graphics of Minecraft differ significantly from real imagery, which imposes additional complexity when using algorithms projected for real-world scenes and makes it harder for our results to generalize to those. Still, it works well as a learning platform, a place to compare algorithms, and a fast way to prototype ideas before moving to real data.



## Folder Structure

```bash
.
├── pose_extraction/  # Tools and scripts to extract camera poses and frames from Minecraft
├── sfm/              # Classic computer vision Structure from Motion experiments
└── slam/             # Visual SLAM experiments
```