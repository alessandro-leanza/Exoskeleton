import csv
import os
import time
from typing import Dict, Any

from .roi_selector import crop_roi
from .output_parser import parse_qwen_output


class PerceptionPipeline:
    """
    YOLO → ROI → Qwen2.5-VL pipeline
    """

    def __init__(self, qwen_model, csv_path="pipeline_log.csv"):
        self.qwen = qwen_model
        
        ##------------parameters----------##
        ##VLM is invoked only when the conditions are satisfied:
        self.conf_threshold = 0.7
        self.stable_frames_required = 2
        self.iou_threshold = 0.7
        
        ##-----------state---------------##
        self.prev_bbox = None
        self.stable_count = 0
        self.frame_index = 0
        #for bbox-trigger
        self.last_inferred_bbox = None
        #for grasp-trigger
        self.vlm_triggered_in_current_grasp = False
        
        #-----------count vlm calling times--------------
        self.total_vlm_call_count = 0
        self.total_run_count = 0
        
        #----------exporting .csv log----------------
        self.csv_path = csv_path
        self._init_csv()
        
        
    def bbox_iou(self, boxA, boxB):
        
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        inter = max(0, xB-xA) * max(0, yB-yA)
        
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

        union = areaA + areaB - inter
        
        if union == 0:
            return 0
        return inter / union

        

    def run_bbox_trigger_mode(self, frame, detections, image_path="") -> Dict[str, Any]:
        self.total_run_count += 1
        vlm_call_count_this_run = 0
        triggered_this_run = False
        skip_reason = None
    
        grasped_found = False
        
        self.frame_index += 1
        
        pipeline_start = time.time()
        
        results = []

        for det in detections:
        
            bbox = det["bbox"]
            confidence = det.get("confidence", 0.0)
            cls = det.get("class", None)
            
            #----------gate 1: run only when grasped------
            if cls != "grasped":
                skip_reason = "not grasped"
                continue
            grasped_found = True
            #----------gate 2: run only when confident-----
            if confidence < self.conf_threshold:
                skip_reason = "low confidence"
                continue
            #----------gate 3: run only when bbox stable------
            if self.prev_bbox is not None:
                
                iou = self.bbox_iou(bbox, self.prev_bbox)
                
                if iou > self.iou_threshold:
                    self.stable_count +=1
                else:
                    self.stable_count = 0
            else:
                self.stable_count = 0
                
            self.prev_bbox = bbox
            
            if self.stable_count < self.stable_frames_required:
                skip_reason = "unstable bounded box"
                continue
            
            #----------gate 4: recent inference------------
            if self.last_inferred_bbox is not None:
                iou_last = self.bbox_iou(bbox, self.last_inferred_bbox)
                if iou_last > self.iou_threshold:
                    skip_reason = "same object"
                    continue
            
            #roi
            roi = crop_roi(frame, bbox)
            if roi is None:
                continue

            qwen_output = self.run_qwen(roi)
            parsed = parse_qwen_output(qwen_output["text"])
             #------------running log-----------------
            vlm_call_count_this_run += 1
            self.total_vlm_call_count += 1
            triggered_this_run = True
            skip_reason = None
            

            results.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "object_name": parsed.get("object_name", ""),
                    "weight_level": parsed.get("weight_level", ""),
                    "fragility_level": parsed.get("fragility_level", ""),
                    "json_ok": parsed.get("json_ok", False),
                    "raw_output": parsed.get("raw_text", ""),
                    "vlm_ms": qwen_output["vlm_ms"],
                    "processor_ms": qwen_output["processor_ms"],
                    "generate_ms": qwen_output["generate_ms"],
                    "decode_ms": qwen_output["decode_ms"],
                }
            )
            self.last_inferred_bbox = bbox
            
           
        #reset parameters when no grasped objects found
        if not grasped_found:
            self.prev_bbox = None
            self.stable_count = 0
            self.last_inferred_bbox = None
            self.vlm_triggered_in_current_grasp = False


        pipeline_latency_ms = (time.time() - pipeline_start) * 1000
        first = results[0] if results else {}
        self.csv_writer.writerow([
            "bbox_trigger",
            pipeline_latency_ms,
            vlm_call_count_this_run,
            self.total_vlm_call_count,
            triggered_this_run,
            skip_reason,
            len(results),
            image_path,
            first.get("object_name", ""),
            first.get("weight_level", ""),
            first.get("fragility_level", ""),
            first.get("json_ok", ""),
            first.get("vlm_ms", ""),
        ])
        self.csv_file.flush()

        return {
            "mode": "bbox_trigger",
            "pipeline_latency_ms": pipeline_latency_ms,
            "vlm_call_count_this_run": vlm_call_count_this_run,
            "total_vlm_call_count": self.total_vlm_call_count,
            "triggered_this_run": triggered_this_run,
            "skip_reason": skip_reason,
            "num_objects": len(results),
            "results": results,
        }
    
    def run_grasp_episode_trigger_mode(self, frame, detections, image_path="") -> Dict[str, Any]:
        self.total_run_count += 1
        vlm_call_count_this_run = 0
        triggered_this_run = False
        skip_reason = None
        self.frame_index += 1
        pipeline_start = time.time()

        results = []

        grasped_found = False

        for det in detections:
            bbox = det["bbox"]
            confidence = det.get("confidence", 0.0)
            cls = det.get("class", None)

            # only consider grasped
            if cls != "grasped":
                continue

            grasped_found = True

            # Gate 1: confidence
            if confidence < self.conf_threshold:
                continue

            # Gate 2: stability
            if self.prev_bbox is not None:
                iou_prev = self.bbox_iou(bbox, self.prev_bbox)
                if iou_prev > self.iou_threshold:
                    self.stable_count += 1
                else:
                    self.stable_count = 0
            else:
                self.stable_count = 0

            self.prev_bbox = bbox

            if self.stable_count < self.stable_frames_required:
                continue

            # Gate 3: already triggered in this grasp episode?
            if self.vlm_triggered_in_current_grasp:
                continue

            roi = crop_roi(frame, bbox)
            if roi is None:
                continue

            qwen_output = self.run_qwen(roi)
            parsed = parse_qwen_output(qwen_output["text"])
            
             #------------running log-----------------
            vlm_call_count_this_run += 1
            self.total_vlm_call_count += 1
            triggered_this_run = True
            skip_reason = None

            results.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "object_name": parsed.get("object_name", ""),
                    "weight_level": parsed.get("weight_level", ""),
                    "fragility_level": parsed.get("fragility_level", ""),
                    "json_ok": parsed.get("json_ok", False),
                    "raw_output": parsed.get("raw_text", ""),
                    "vlm_ms": qwen_output["vlm_ms"],
                    "processor_ms": qwen_output["processor_ms"],
                    "generate_ms": qwen_output["generate_ms"],
                    "decode_ms": qwen_output["decode_ms"],
                }
            )

            # mark that VLM has been triggered in this grasp episode
            self.vlm_triggered_in_current_grasp = True

        # if no grasped object is found, reset grasp episode
        if not grasped_found:
            self.prev_bbox = None
            self.stable_count = 0
            self.vlm_triggered_in_current_grasp = False

        pipeline_latency_ms = (time.time() - pipeline_start) * 1000.0
        first = results[0] if results else {}
        self.csv_writer.writerow([
            "grasp_episode",
            pipeline_latency_ms,
            vlm_call_count_this_run,
            self.total_vlm_call_count,
            triggered_this_run,
            skip_reason,
            len(results),
            image_path,
            first.get("object_name", ""),
            first.get("weight_level", ""),
            first.get("fragility_level", ""),
            first.get("json_ok", ""),
            first.get("vlm_ms", ""),
        ])
        self.csv_file.flush()
        return {
            "mode": "grasp_episode",
            "pipeline_latency_ms": pipeline_latency_ms,
            "vlm_call_count_this_run": vlm_call_count_this_run,
            "total_vlm_call_count": self.total_vlm_call_count,
            "triggered_this_run": triggered_this_run,
            "skip_reason": skip_reason,
            "num_objects": len(results),
            "results": results,
        }
        
    def _init_csv(self):
        file_exists = os.path.isfile(self.csv_path)

        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        if not file_exists:
            self.csv_writer.writerow([
                "mode",
                "pipeline_latency_ms",
                "vlm_call_count_this_run",
                "total_vlm_call_count",
                "triggered_this_run",
                "skip_reason",
                "num_objects",
                "image_path",
                "object_name",
                "weight_level",
                "fragility_level",
                "json_ok",
                "vlm_ms",
                ])
                
                
    def close(self):
        if hasattr(self, "csv_file"):
            self.csv_file.close()    
            
    
    def run(self, frame, detections, mode="grasp_episode", image_path=""):
        if mode == "bbox":
            return self.run_bbox_trigger_mode(frame, detections, image_path=image_path)
        elif mode == "grasp_episode":
            return self.run_grasp_episode_trigger_mode(frame, detections, image_path=image_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        

    def run_qwen(self, image) -> Dict[str, Any]:
        response, vlm_ms, proc_ms, gen_ms, dec_ms = self.qwen.infer(
            image,
            max_new_tokens=64,
        )

        return {
            "text": response,
            "vlm_ms": vlm_ms,
            "processor_ms": proc_ms,
            "generate_ms": gen_ms,
            "decode_ms": dec_ms,
        }
