# Pose Extraction

This folder contains everything related to building the Minecraft environments, creating missions (the scenarios from which we extract data), and running the full data-extraction pipeline.

## Environment Setup

We use [Malmo](https://github.com/microsoft/malmo/tree/0.36.0), a project from Microsoft designed as a programmable Minecraft environment. Malmo runs a local Minecraft server and lets you create agents that interact with the world, which is useful for AI experiments, especially reinforcement learning although we are not using it for RL in this project.

Missions in Malmo are defined by writing an XML description of the world. The structure of these missions follows the XML schema from [this documentation](https://microsoft.github.io/malmo/0.30.0/Schemas/Mission.html). If you want to write your own mission, we recommend checking the `small_corridor` notebook as a first example since it's the simplest one we created. With this schema, you can configure almost anything about the environment. Once the mission is running, Malmo provides a set of classes and functions (see [the main documentation](https://microsoft.github.io/malmo/0.30.0/Documentation/index.html)) that let you pull information directly from Minecraft through Python, as long as your mission specifies the right settings. Notice that the Python API itself is just a wrapper around the C++ library, so the official documentation is written in C++.

For installation, we suggest following the official instructions in the [Malmo installation docs](https://github.com/microsoft/malmo/tree/0.36.0/scripts/python-wheel), but our strong recommendation is to use the [Docker container](https://hub.docker.com/r/andkram/malmo) setup as Malmo is relatively old and can be tricky to install due to software compatibility issues. The container already comes configured and tends to work without headaches.

## Experiments

Our goal is to explore multiple environments and understand how different SLAM methods behave across them. We will expand this section later once we have more worlds available, along with information about why each scenario was created and what aspects it is intended to test.

SPACE FOR GIF WITH MULTIPLE SCENARIOS AND SHORT EXPLANATIONS