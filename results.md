# Results

* Connected to the Neurolabs API directly using requests.

* Loaded the two CSV files containing the cooler and ambient image URLs.

* Looked up the task UUIDs for the “Cooler” and “Ambient” tasks.

* Submitted each image URL individually. I avoided batch submission because the staging API rate-limits quite aggressively, and individual requests were the most reliable option.

* Retrieved the results using the per-image /results/{result_uuid} endpoint and saved all responses to JSON files.

* For cooler images, I downloaded each image and drew the bounding boxes and labels that came back from the model.

* Pulled catalog item data and joined it with the detections so I could generate product-level insights.

* Finally, I generated two simple charts:

  * A pie chart showing how often each product was detected.

  * A bar chart showing highest vs. lowest confidence detections.

* Everything is written to the output/ folder for easy inspection.


## Assumptions

* Some API results come back as “IN_PROGRESS”, so I added a small retry loop to wait for the final output.

* I assumed only cooler images needed bounding box visualisation (based on the task description).


## Limitations

* The staging API rate limits pretty hard, so submissions have to be slow and sequential.

* If a particular image never becomes “PROCESSED”, the script will eventually skip it.

* The analysis is intentionally basic. With more time, I’d expand the charts and maybe explore multi-image comparisons or confidence distributions per brand/category.
