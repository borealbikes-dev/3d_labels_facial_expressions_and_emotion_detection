# 3D Labels For Facial Expressions and Emotion Detection

## Overview

This is a sample project that demonstrates how to use the Foxglove Studio 3D labels feature to annotate facial expressions on images generated by the Holoscene Edge ebike.

## Setup

1. Install the [Foxglove Studio](https://foxglove.dev/studio/) desktop application.
2. Clone this repository.
3. Build the project using colcon: `colcon build --packages-select facial_expression_recognition`
4. Source the workspace: `source install/setup.bash`
5. Run the project: `ros2 launch facial_expression_recognition 3d_marker_publisher_node.launch.py`

## Results

Image with markers on facial features:

![Alt text](https://github.com/borealbikes-dev/3d_labels_facial_expressions_foxglove/blob/facial_markers/results/published_marker.gif)

Emotion Detection 

![Alt text](https://github.com/borealbikes-dev/3d_labels_facial_expressions_foxglove/blob/main/results/emotions_result.gif)

Plot the count of emotions using Foxglove Data Analyzer and Jupyter Notebook

![Alt text](https://github.com/borealbikes-dev/3d_labels_facial_expressions_foxglove/blob/main/results/emotion_counts.png)

Jupyter Notebook - [Link](https://github.com/borealbikes-dev/3d_labels_facial_expressions_foxglove/blob/main/facial_expression_recognition/facial_expression_recognition/foxglove_data_analyzer.ipynb)

Plot a path on the map based on emotion

![Alt text](https://github.com/borealbikes-dev/3d_labels_facial_expressions_foxglove/blob/main/results/path_emotion.png)
