from __future__ import annotations
import os
import time
import json
import csv
from pathlib import Path
from typing import List, Any, Optional, Dict

import requests
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from dotenv import load_dotenv


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("NEUROLABS_API_KEY")
if not API_KEY:
    raise RuntimeError("NEUROLABS_API_KEY missing in .env")

BASE_URL = "https://staging.api.neurolabs.ai/v2"

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"

COOLER_CSV = INPUT_DIR / "cooler.csv"
AMBIENT_CSV = INPUT_DIR / "ambient.csv"

OUTPUT_DIR = BASE_DIR / "output"
JSON_DIR = OUTPUT_DIR / "json"
IMG_DIR = OUTPUT_DIR / "images"
CHART_DIR = OUTPUT_DIR / "charts"

for d in (JSON_DIR, IMG_DIR, CHART_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def save_json(obj: Any, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("Saved:", path)


# ----------------------------------------------------------------------
# API Client
# ----------------------------------------------------------------------

class NeurolabsClient:
    #Very small client, retries are handled in main()

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-API-Key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _get(self, path: str):
        url = f"{self.base_url}{path}"
        r = requests.get(url, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        url = f"{self.base_url}{path}"
        r = requests.post(url, headers=self.headers, json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_tasks(self):
        resp = self._get("/image-recognition/tasks?limit=50&offset=0")
        return resp.get("items", [])

    def submit_url(self, task_uuid: str, url: str):
        return self._post(f"/image-recognition/tasks/{task_uuid}/urls", {"urls": [url]})

    def get_result(self, task_uuid: str, result_uuid: str):
        return self._get(f"/image-recognition/tasks/{task_uuid}/results/{result_uuid}")

    def list_catalog_items(self):
        return self._get("/catalog-items?limit=1000").get("items", [])


# ----------------------------------------------------------------------
# Result processing
# ----------------------------------------------------------------------

class ResultProcessor:
    #Responsible for reading CSVs, extracting detections, visualising and charting

    # CSV file extraction
    @staticmethod
    def read_csv_urls(path: Path) -> List[str]:
        urls = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = (row.get("url") or "").strip()
                if url.startswith("<") and url.endswith(">"):
                    url = url[1:-1].strip()
                if url:
                    urls.append(url)
        return urls

    @staticmethod
    def extract_annotations(result: dict) -> List[dict]:
        coco = result.get("coco", {})
        return coco.get("annotations", [])

    @staticmethod
    def extract_image_url(result: dict) -> Optional[str]:
        coco = result.get("coco", {})
        images = coco.get("images") or []
        if images:
            return images[0].get("file_name") or result.get("image_url")
        return result.get("image_url")

    # Image loading / boxes
    @staticmethod
    def load_image(url: str) -> Optional[np.ndarray]:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            arr = np.frombuffer(r.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def draw_boxes(img: np.ndarray, annotations: List[dict]):
        for ann in annotations:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Chart / analysis
    def build_charts(self, json_dir: Path, client: NeurolabsClient):
        detection_rows: List[Dict[str, Any]] = []
        category_rows: List[Dict[str, Any]] = []

        # Collect detections and category info
        for path in json_dir.glob("*_result_*.json"):
            result = json.loads(path.read_text())
            coco = result.get("coco", {})
            anns = coco.get("annotations", [])
            cats = coco.get("categories", [])
            images = coco.get("images", [])

            image_url = None
            if images:
                image_url = images[0].get("file_name") or result.get("image_url")

            # Build catalog-like rows from categories
            for c in cats:
                neu = c.get("neurolabs") or {}
                category_rows.append(
                    {
                        "category_id": c.get("id"),
                        "category_name": c.get("name"),
                        "product_uuid": neu.get("productUuid"),
                        "product_name": neu.get("name") or c.get("name"),
                    }
                )

            # Detection rows
            for ann in anns:
                neu = ann.get("neurolabs") or {}
                score = neu.get("score")
                detection_rows.append(
                    {
                        "category_id": ann.get("category_id"),
                        "score": score,
                        "image_url": image_url,
                    }
                )

        if not detection_rows:
            print("No detections found; skipping charts.")
            return

        det_df = pd.DataFrame(detection_rows)
        cat_df = pd.DataFrame(category_rows).drop_duplicates(subset=["category_id"])

        # Try to enrich with catalog-items
        api_catalog_df = pd.DataFrame()
        try:
            api_items = client.list_catalog_items()
            if api_items:
                api_catalog_df = pd.DataFrame(api_items)
        except Exception:
            pass

        if not api_catalog_df.empty and "uuid" in api_catalog_df.columns:
            cat_df = cat_df.merge(
                api_catalog_df,
                left_on="product_uuid",
                right_on="uuid",
                how="left",
                suffixes=("", "_api"),
            )

        # Join detections with catalog/category info
        det_df = det_df.merge(cat_df, on="category_id", how="left")

        # Final label preference: product_name -> category_name -> "Unknown"
        det_df["label"] = (
            det_df.get("product_name")
            .fillna(det_df.get("category_name"))
            .fillna("Unknown")
        )

        # ------------------------------------------------------------------
        # Chart 1: Product distribution (pie chart)
        # ------------------------------------------------------------------
        label_counts = det_df["label"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        label_counts.plot.pie(autopct="%1.1f%%", ylabel="", ax=ax1)
        fig1.savefig(CHART_DIR / "product_distribution.png", bbox_inches="tight")
        plt.close(fig1)
        print("Saved:", CHART_DIR / "product_distribution.png")

        # ------------------------------------------------------------------
        # Chart 2: Average confidence per product (bar chart)
        # ------------------------------------------------------------------
        df_conf = det_df.dropna(subset=["score"])
        if df_conf.empty:
            print("No confidence scores found; skipping confidence chart.")
            return

        mean_conf = (
            df_conf.groupby("label")["score"]
            .mean()
            .sort_values(ascending=True)  # ascending so low -> high
        )

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        mean_conf.plot.bar(ax=ax2)
        ax2.set_xlabel("Product")
        ax2.set_ylabel("Average detection confidence")
        ax2.set_title("Average confidence by product")
        plt.xticks(rotation=45, ha="right")
        fig2.tight_layout()
        fig2.savefig(CHART_DIR / "product_confidence.png", bbox_inches="tight")
        plt.close(fig2)
        print("Saved:", CHART_DIR / "product_confidence.png")

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

def main():
    print("Project root:", BASE_DIR)

    client = NeurolabsClient(BASE_URL, API_KEY)
    proc = ResultProcessor()

    # ------------------------------------------------------------------
    # 1) Retrieve tasks
    # ------------------------------------------------------------------
    print("\n1) Retrieving tasks...")
    mapping = {"cooler": None, "ambient": None}
    for t in client.list_tasks():
        name = (t.get("name") or "").lower()
        if "cooler" in name:
            mapping["cooler"] = t["uuid"]
        if "ambient" in name:
            mapping["ambient"] = t["uuid"]

    print("Task UUIDs:", mapping)
    if None in mapping.values():
        raise RuntimeError("Could not locate required tasks in account.")

    # ------------------------------------------------------------------
    # 2) Load CSVs
    # ------------------------------------------------------------------
    print("\n2) Loading CSV files...")
    cooler_urls = proc.read_csv_urls(COOLER_CSV)
    ambient_urls = proc.read_csv_urls(AMBIENT_CSV)
    print(f"Cooler URLs: {len(cooler_urls)} Ambient URLs: {len(ambient_urls)}")

    # ------------------------------------------------------------------
    # 3) Submit URLs individually (avoids rate limit)
    # ------------------------------------------------------------------
    def submit_individual(urls: List[str], task_uuid: str, label: str) -> List[str]:
        print(f"\nSubmitting {label} URLs individually...")
        result_ids: List[str] = []

        for url in urls:
            while True:
                try:
                    resp = client.submit_url(task_uuid, url)
                    # API returns a list of result UUIDs
                    if isinstance(resp, list) and resp:
                        result_ids.append(resp[0])
                    break
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        print("429 received; waiting 5 seconds before retry...")
                        time.sleep(5)
                        continue
                    raise
        print(f"{label.capitalize()} result UUIDs:", result_ids)
        return result_ids

    cooler_ids = submit_individual(cooler_urls, mapping["cooler"], "cooler")
    ambient_ids = submit_individual(ambient_urls, mapping["ambient"], "ambient")

    save_json(cooler_ids, JSON_DIR / "cooler_submission.json")
    save_json(ambient_ids, JSON_DIR / "ambient_submission.json")

    # ------------------------------------------------------------------
    # 4) Fetch results
    # ------------------------------------------------------------------
    print("\n4) Fetching results...")

    def fetch_results(task_uuid: str, ids: List[str], prefix: str) -> List[Path]:
        paths: List[Path] = []
        for rid in ids:
            result = client.get_result(task_uuid, rid)
            out = JSON_DIR / f"{prefix}_result_{rid}.json"
            save_json(result, out)
            paths.append(out)
            time.sleep(0.5)
        return paths

    cooler_files = fetch_results(mapping["cooler"], cooler_ids, "cooler")
    ambient_files = fetch_results(mapping["ambient"], ambient_ids, "ambient")

    # ------------------------------------------------------------------
    # 5) Visualise cooler detections
    # ------------------------------------------------------------------
    print("\n5) Visualising cooler results...")

    for p in cooler_files:
        obj = json.loads(p.read_text())
        anns = proc.extract_annotations(obj)
        img_url = proc.extract_image_url(obj)
        img = proc.load_image(img_url)
        if img is None:
            continue
        proc.draw_boxes(img, anns)
        out = IMG_DIR / (p.stem + ".jpg")
        cv2.imwrite(str(out), img)
        print("Saved:", out)

    # ------------------------------------------------------------------
    # 6) Charts / analysis
    # ------------------------------------------------------------------
    print("\n6) Generating charts...")
    proc.build_charts(JSON_DIR, client)

    print("\nDone. Outputs saved under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
