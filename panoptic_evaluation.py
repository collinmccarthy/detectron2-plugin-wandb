import contextlib
import io
import itertools
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict

from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.evaluation.panoptic_evaluation import _print_panoptic_results

try:
    from panopticapi.evaluation import pq_compute_multi_core
except ImportError as e:
    raise ImportError(
        f"Install panopticapi via `pip install panopticapi` and re-run.\n{e}"
    )

logger = logging.getLogger(__name__)


def pq_compute(
    gt_json_file, pred_json_file, gt_folder=None, pred_folder=None, partial_eval=False
):
    """Modified from panopticapi.evaluation.py, pq_compute() to support partial evaluation"""

    start_time = time.time()
    with open(gt_json_file, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_file, "r") as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace(".json", "")
    if pred_folder is None:
        pred_folder = pred_json_file.replace(".json", "")
    categories = {el["id"]: el for el in gt_json["categories"]}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception(
            "Folder {} with ground truth segmentations doesn't exist".format(gt_folder)
        )
    if not os.path.isdir(pred_folder):
        raise Exception(
            "Folder {} with predicted segmentations doesn't exist".format(pred_folder)
        )

    pred_annotations = {el["image_id"]: el for el in pred_json["annotations"]}
    matched_annotations_list = []
    for gt_ann in gt_json["annotations"]:
        image_id = gt_ann["image_id"]
        # Update: add partial_eval
        if image_id in pred_annotations:
            matched_annotations_list.append((gt_ann, pred_annotations[image_id]))
        elif not partial_eval:
            raise Exception("no prediction for the image with id: {}".format(image_id))

    pq_stat = pq_compute_multi_core(
        matched_annotations_list, gt_folder, pred_folder, categories
    )

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(
            categories, isthing=isthing
        )
        if name == "All":
            results["per_class"] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print(
            "{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
                name,
                100 * results[name]["pq"],
                100 * results[name]["sq"],
                100 * results[name]["rq"],
                results[name]["n"],
            )
        )

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


class PartialCOCOPanopticEvaluator(COCOPanopticEvaluator):
    """
    Support partial evaluation for debugging runs
    """

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            # Update: call our version of pq_compute above instead of panopticapi
            # from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                    partial_eval=True,  # Added
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results
