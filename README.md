# AR Darts

AR Darts is a Raspberry Pi–powered augmented reality dartboard system that overlays digital game visuals onto a physical dartboard using projection and computer vision.

This project is an open foundation for building modular AR sports experiences.

## What It Does (Current State)
	•	501 (double-out)
	•	Around-the-World
	•	Click-to-hit simulation
	•	Projection-ready 3:2 layout
	•	Local calibration system (persisted)
	•	Flask backend for event logging

## Tech Stack

	•	Raspberry Pi 5
	•	Raspberry Pi Camera Module 3 (Wide)
	•	Python (Flask)
	•	Vanilla JavaScript
	•	HTML5 Canvas
	•	OpenCV (vision tracking in progress)

## Roadmap

	•	ArUco marker-based perspective correction
	•	Vision-based dart detection
	•	Flight-to-tip vector estimation
	•	Advanced scoring validation
	•	Additional AR game modes
	•	Hardware enclosure iterations

## Vision

AR Darts aims to become an open platform for experimenting with augmented reality overlays in physical sports environments.

Developers, makers, and hardware enthusiasts are welcome to fork, experiment, and contribute.

## Running Locally

```bash

pip install -r requirements.txt
python app.py
