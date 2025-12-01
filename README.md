# Neurolabs Technical Assessment
This project contains my solution for the Neurolabs Implementation Engineer take-home task. The script interacts with the Neurolabs Image Recognition API, runs inference on two sets of images (Cooler and Ambient), saves the raw results, visualises detections for cooler images, and generates a couple of basic analysis charts based on the detected products.

I’ve tried to keep the setup minimal and the workflow straightforward so it can be run end-to-end without any extra configuration.

## How to run the project

1. Set up a Python environment
2. Install dependencies (use pip install -r requirements.txt)
3. Add your API key (create a .env file in the project root and add your api key)
4. Run the script

## Output locations

Raw JSON result files:
output/json/

Bounding box images for cooler task:
output/images/

Charts:
output/charts/

## Notes

Images are submitted one-at-a-time because the staging API has a strict rate limit.

Catalog items are pulled once and joined with product detections using product UUIDs.

Only the cooler images include bounding box visualisation since that was part of the task requirements.
